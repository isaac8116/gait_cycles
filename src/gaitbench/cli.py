"""Command-line interface: ``gaitbench <command> ...``.

Every command reads and writes files under a *data root* laid out by the path
helpers in :mod:`gaitbench.formats`::

    <data>/keypoints/<clip>__<backend>.npz
    <data>/events/<clip>__<backend>__<detector>.csv
    <data>/labels/<clip>__<rater>.csv          (synthetic truth: rater "truth")

Commands, in pipeline order:

* ``simulate``         synthetic keypoints + truth labels (harness self-test)
* ``import-openpose``  OpenPose JSON directory -> keypoints
* ``pose``             video -> keypoints with a CPU pose model through myogait
* ``decimate``         keep every n-th keypoint frame (240 fps -> 60/30 fps copies)
* ``detect``           keypoints -> events, one file per detector
* ``score``            events vs labels -> matches, metrics, temporal parameters
* ``agree``            events vs a reference detector, when there is no truth
* ``plot``             foot-signal diagnostic plot per clip
* ``run``              detect + score (or agree) from a YAML config, into results/runs/<run_id>/
* ``probe`` / ``cfr``  video facts / constant-frame-rate copy
* ``label``            frame-stepping labeller
* ``realsense-modes`` / ``realsense-record`` / ``realsense-export``  Intel RealSense D455 capture
* ``list-detectors`` / ``env``  what is installed

Relative paths (also inside configs) are resolved against the current
directory, so run gaitbench from the repository root.

Optional parts (myogait, OpenCV, the labeller, RealSense) are imported only by
the commands that need them, so ``gaitbench --help`` works on a bare install.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import __version__
from .formats import (
    FormatError,
    Keypoints,
    events_path,
    frames_to_times,
    keypoints_path,
    labels_path,
    read_events,
    read_labels,
    safe_name,
    write_events,
    write_labels,
)

logger = logging.getLogger("gaitbench")

TRUTH_RATER = "truth"
SYNTHETIC_BACKEND = "synthetic"
DEFAULT_TOLERANCE_MS = 100.0
DEFAULT_RESULTS_ROOT = Path("results") / "runs"

# Columns printed in summary tables (whichever exist).
_SUMMARY_COLUMNS: tuple[str, ...] = (
    "backend", "ref_detector", "detector", "event", "n_truth", "n_pred", "recall", "precision", "f1",
    "bias_ms", "sd_ms", "mae_ms", "mae_frames", "p_within_1_frame", "p_within_2_frames",
)

_CONFIG_KEYS = {
    "run_id", "description", "data_root", "results_root", "simulate", "import_openpose", "clips", "backends",
    "detectors", "detector_params", "truth_rater", "reference_detector", "tolerance_ms", "restrict_to_truth_span",
    "plots", "skip_existing",
}


class CLIError(RuntimeError):
    """A problem the user can fix (missing files, bad arguments); printed without a traceback."""


# ---------------------------------------------------------------------------
# data-root files
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataFile:
    """A file in the data root, with the name parts parsed from its file name.

    The parts are filename-safe forms (:func:`gaitbench.formats.safe_name`);
    the real names are read from the file contents where they are needed.
    """

    path: Path
    clip: str
    backend: str = ""
    detector: str = ""


def keypoint_files(
    data: str | Path, clips: Sequence[str] | None = None, backends: Sequence[str] | None = None
) -> list[DataFile]:
    """Keypoints files under ``<data>/keypoints``, optionally filtered by clip and backend."""
    out = []
    for path in sorted((Path(data) / "keypoints").glob("*__*.npz")):
        clip, backend = path.stem.rsplit("__", 1)
        out.append(DataFile(path, clip, backend))
    return [f for f in out if _keep(f.clip, clips) and _keep(f.backend, backends)]


def event_files(
    data: str | Path,
    clips: Sequence[str] | None = None,
    backends: Sequence[str] | None = None,
    detectors: Callable[[str], bool] | None = None,
) -> list[DataFile]:
    """Events files under ``<data>/events``; ``detectors`` filters on the safe detector name."""
    out = []
    for path in sorted((Path(data) / "events").glob("*__*__*.csv")):
        clip, backend, detector = path.stem.rsplit("__", 2)
        out.append(DataFile(path, clip, backend, detector))
    return [
        f for f in out
        if _keep(f.clip, clips) and _keep(f.backend, backends) and (detectors is None or detectors(f.detector))
    ]


def label_files(data: str | Path, rater: str, clips: Sequence[str] | None = None) -> list[DataFile]:
    """Labels files of one rater under ``<data>/labels``."""
    suffix = "__" + safe_name(rater) + ".csv"
    out = []
    for path in sorted((Path(data) / "labels").glob("*" + suffix)):
        out.append(DataFile(path, path.name[: -len(suffix)], safe_name(rater)))
    return [f for f in out if _keep(f.clip, clips)]


def _keep(part: str, wanted: Sequence[str] | None) -> bool:
    return not wanted or part in {safe_name(w) for w in wanted}


# ---------------------------------------------------------------------------
# detector names
# ---------------------------------------------------------------------------


def expand_detectors(spec: str | Sequence[str] | None) -> list[str]:
    """Turn a detector spec into detector names to run.

    ``spec`` is a comma-separated string or a list whose items are detector
    names or the groups ``all`` (every detector of
    :func:`gaitbench.detectors.list_detectors`), ``ours`` (zeni, stenum) and
    ``myogait`` (every ``myogait:<method>``). Order is kept, duplicates dropped.
    """
    from .detectors import MYOGAIT_PREFIX, list_detectors

    items = _spec_items(spec) or ["ours"]
    available = list_detectors()
    names: list[str] = []
    for item in items:
        if item == "all":
            names += available
        elif item == "ours":
            names += [d for d in available if not d.startswith(MYOGAIT_PREFIX)]
        elif item == "myogait":
            mg = [d for d in available if d.startswith(MYOGAIT_PREFIX)]
            if not mg:
                raise CLIError("No myogait detectors: myogait is not installed (pip install 'gaitbench[myogait]')")
            names += mg
        else:
            names.append(item)
    return list(dict.fromkeys(names))


def detector_filter(spec: str | Sequence[str] | None) -> Callable[[str], bool] | None:
    """Predicate on safe detector names for selecting existing events files (None = keep all)."""
    items = _spec_items(spec)
    if not items or "all" in items:
        return None
    exact = {safe_name(i) for i in items if i not in ("ours", "myogait")}
    ours = "ours" in items
    myogait = "myogait" in items
    mg_prefix = "myogait-"  # safe_name("myogait:<method>") starts with this

    def keep(name: str) -> bool:
        is_mg = name.startswith(mg_prefix)
        return name in exact or (myogait and is_mg) or (ours and not is_mg)

    return keep


def _spec_items(spec: str | Sequence[str] | None) -> list[str]:
    if spec is None:
        return []
    if isinstance(spec, str):
        spec = spec.split(",")
    return [str(s).strip() for s in spec if str(s).strip()]


def _known_names() -> dict[str, str]:
    """Safe name -> real detector name, so empty events files still get a proper name."""
    try:
        from .detectors import list_detectors

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return {safe_name(d): d for d in list_detectors()}
    except Exception:  # listing names is cosmetic here; never fail scoring over it
        return {}


# ---------------------------------------------------------------------------
# pipeline steps (also used by `run`)
# ---------------------------------------------------------------------------


def simulate_clip(
    data: str | Path,
    clip_id: str = "synthetic",
    *,
    fps: float = 60.0,
    pan: float | str = 0.0,
    direction: int | str = 1,
    noise_px: float = 0.0,
    duration: float = 6.0,
    seed: int = 0,
    speed: float | None = None,
    dropout: float = 0.0,
    near_side: str | None = None,
) -> tuple[Path, Path]:
    """Write one synthetic walk's keypoints (backend ``synthetic``) and truth labels (rater ``truth``).

    ``pan`` is the camera speed in m/s, or ``"follow"`` for the walking speed
    (walker stays centred). ``direction`` is +1/-1 or ``"ltr"``/``"rtl"``.
    Returns the keypoints and labels paths.
    """
    from .synthetic import WalkerParams, simulate_walk

    overrides: dict[str, Any] = {
        "clip_id": clip_id,
        "fps": float(fps),
        "duration_s": float(duration),
        "direction": _parse_direction(direction),
        "noise_px": float(noise_px),
        "dropout_prob": float(dropout),
        "seed": int(seed),
    }
    if speed is not None:
        overrides["speed_mps"] = float(speed)
    if near_side is not None:
        overrides["near_side"] = near_side
    walk_speed = overrides.get("speed_mps", WalkerParams().speed_mps)
    overrides["pan_speed_mps"] = walk_speed if str(pan).lower() == "follow" else float(pan)
    walk = simulate_walk(**overrides)
    kp_file = walk.keypoints.save(keypoints_path(data, clip_id, SYNTHETIC_BACKEND))
    lab_file = write_labels(walk.truth, labels_path(data, clip_id, TRUTH_RATER))
    return kp_file, lab_file


def import_openpose(
    data: str | Path,
    json_dir: str | Path,
    clip_id: str,
    *,
    fps: float | None = None,
    width: float | None = None,
    height: float | None = None,
    video: str | Path | None = None,
    person: str | int = "best",
    backend: str = "openpose",
    foot_index: str | None = None,
) -> Path:
    """Convert an OpenPose JSON directory into ``<data>/keypoints/<clip>__<backend>.npz``."""
    from .io import load_openpose_dir

    if video is None and None in (fps, width, height):
        raise CLIError("import-openpose needs --fps, --width and --height, or --video to read them from")
    kp = load_openpose_dir(
        json_dir, clip_id, fps, width, height, person=person, video_path=video, backend=backend, foot_index=foot_index
    )
    return kp.save(keypoints_path(data, clip_id, backend))


def detect_all(
    data: str | Path,
    detectors: Sequence[str],
    *,
    clips: Sequence[str] | None = None,
    backends: Sequence[str] | None = None,
    params: Mapping[str, Mapping[str, Any]] | None = None,
    skip_existing: bool = False,
    echo: Callable[[str], None] = print,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every detector on every selected keypoints file and write the events files.

    A detector that raises is recorded and skipped; the others still run.
    An empty result is still written (as a header-only file) so that scoring
    counts the clip's events as missed rather than ignoring the detector.

    Returns
    -------
    done, failures:
        ``done`` has one row per written file (clip_id, backend, detector,
        n_hs, n_to, path, skipped); ``failures`` one row per failed run
        (clip_id, backend, detector, error).
    """
    from .detectors import get_detector

    files = keypoint_files(data, clips, backends)
    if not files:
        raise CLIError(f"No keypoints files in {Path(data) / 'keypoints'} match clips={clips} backends={backends}")
    params = dict(params or {})
    done, failures = [], []
    unknown_params = sorted(set(params) - set(detectors))
    if unknown_params:
        raise CLIError(f"Parameters given for detectors that are not run: {unknown_params}")
    # A typo in a name or parameter, or a missing package, fails before any work is done.
    runners: dict[str, Callable[[Keypoints], pd.DataFrame]] = {}
    for name in detectors:
        try:
            runners[name] = get_detector(name, **dict(params.get(name, {})))
        except (KeyError, TypeError, ImportError) as exc:
            raise CLIError(f"Detector {name!r}: {exc.args[0] if exc.args else exc}") from exc
    for f in files:
        kp = Keypoints.load(f.path)
        backend = str(kp.meta.get("backend", "")) or f.backend
        for name, runner in runners.items():
            out = events_path(data, kp.clip_id, backend, name)
            if skip_existing and out.exists():
                done.append({"clip_id": kp.clip_id, "backend": backend, "detector": name, "path": str(out),
                             "skipped": True})
                continue
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    events = runner(kp)
            except Exception as exc:  # one detector failing must not stop the sweep
                failures.append(_failure(kp.clip_id, backend, name, exc))
                echo(f"  FAILED {kp.clip_id} / {backend} / {name}: {type(exc).__name__}: {exc}")
                continue
            for w in caught:
                logger.info("%s / %s / %s: %s", kp.clip_id, backend, name, w.message)
            write_events(events, out)
            n_hs = int((events["event"] == "HS").sum())
            n_to = int((events["event"] == "TO").sum())
            done.append({"clip_id": kp.clip_id, "backend": backend, "detector": name, "n_hs": n_hs, "n_to": n_to,
                         "path": str(out), "skipped": False})
            echo(f"  {kp.clip_id} / {backend} / {name}: {n_hs} HS, {n_to} TO")
    return (
        pd.DataFrame(done, columns=["clip_id", "backend", "detector", "n_hs", "n_to", "path", "skipped"]),
        pd.DataFrame(failures, columns=["clip_id", "backend", "detector", "error"]),
    )


def score_events(
    data: str | Path,
    rater: str,
    *,
    tolerance_s: float = DEFAULT_TOLERANCE_MS / 1000.0,
    clips: Sequence[str] | None = None,
    backends: Sequence[str] | None = None,
    detectors: str | Sequence[str] | None = None,
    restrict_to_truth_span: bool = True,
) -> dict[str, pd.DataFrame]:
    """Score every events file of every labelled clip against one rater's labels.

    Each events file is matched on its own (:func:`gaitbench.scoring.match_events`),
    so an empty file still counts its clip's labels as missed.

    Returns
    -------
    dict
        ``matches`` (one row per TP/FP/FN), ``metrics`` (per backend,
        detector and event), ``metrics_by_side`` (also per side; near and
        far limbs differ) and ``params`` (temporal parameters vs truth, per
        backend and detector). Frame-based columns use each keypoints file's
        sample rate.
    """
    from .scoring import compare_parameters, match_events, temporal_parameters

    labels = label_files(data, rater, clips)
    if not labels:
        raise CLIError(f"No labels for rater {rater!r} in {Path(data) / 'labels'}")
    keep = detector_filter(detectors)
    names = _known_names()
    kp_cache: dict[tuple[str, str], Keypoints | None] = {}
    match_parts, preds, truths, rates = [], [], [], []
    for lab in labels:
        truth = read_labels(lab.path)
        clip_id = _clip_id(truth, lab.clip)
        truth = truth.assign(clip_id=clip_id)
        truth = _with_times(truth, lambda: _any_keypoints(data, lab.clip, kp_cache), lab.path)
        truths.append(truth)
        files = event_files(data, [lab.clip], backends, keep)
        if not files:
            warnings.warn(f"{clip_id}: labelled by {rater!r} but has no events files", stacklevel=2)
        for ef in files:
            pred, backend, detector = _read_source(ef, clip_id, names)
            kp = _keypoints(data, ef.clip, ef.backend, kp_cache)
            pred = _with_times(pred, lambda: kp, ef.path)
            m = match_events(pred, truth, tolerance_s=tolerance_s, restrict_to_truth_span=restrict_to_truth_span)
            match_parts.append(_fill_source(m, clip_id, backend, detector))
            preds.append(pred)
            rates.append((backend, clip_id, _sample_rate(kp)))
    if not match_parts:
        raise CLIError(f"No events files for the clips labelled by {rater!r}; run `gaitbench detect` first")

    matches = pd.concat(match_parts, ignore_index=True)
    metrics = _metrics_per_backend(matches, rates, ("backend", "detector", "event"))
    by_side = _metrics_per_backend(matches, rates, ("backend", "detector", "side", "event"))
    pred_params = temporal_parameters(pd.concat(preds, ignore_index=True))
    truth_params = temporal_parameters(pd.concat(truths, ignore_index=True))
    params = compare_parameters(pred_params, truth_params, tolerance_s=tolerance_s, group_by=("backend", "detector"))
    return {"matches": matches, "metrics": metrics, "metrics_by_side": by_side, "params": params}


def agree_events(
    data: str | Path,
    reference: str,
    *,
    tolerance_s: float = DEFAULT_TOLERANCE_MS / 1000.0,
    clips: Sequence[str] | None = None,
    backends: Sequence[str] | None = None,
    detectors: str | Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Agreement of every detector with a reference detector on the same keypoints (no truth needed).

    For each clip and backend, every other detector's events are matched
    against the reference's (:func:`gaitbench.scoring.agreement`). "recall" is
    then the share of reference events another detector also found, and
    ``bias_ms`` its mean offset from the reference.

    Returns
    -------
    dict
        ``matches`` and ``metrics`` (per backend, reference, detector and event).
    """
    from .scoring import agreement

    keep = detector_filter(detectors)
    ref_safe = safe_name(reference)
    names = _known_names()
    files = event_files(data, clips, backends)
    groups: dict[tuple[str, str], list[DataFile]] = {}
    for f in files:
        groups.setdefault((f.clip, f.backend), []).append(f)
    kp_cache: dict[tuple[str, str], Keypoints | None] = {}
    parts, rates = [], []
    for (clip, backend_part), group in sorted(groups.items()):
        ref_file = next((f for f in group if f.detector == ref_safe), None)
        if ref_file is None:
            warnings.warn(f"{clip} / {backend_part}: no events from reference {reference!r}; skipped", stacklevel=2)
            continue
        ref, backend, ref_name = _read_source(ref_file, None, names)
        clip_id = _clip_id(ref, clip)
        kp = _keypoints(data, clip, backend_part, kp_cache)
        ref = _with_times(ref.assign(clip_id=clip_id), lambda: kp, ref_file.path)
        for f in group:
            if f is ref_file or (keep is not None and not keep(f.detector)):
                continue
            pred, _, detector = _read_source(f, clip_id, names)
            m = agreement(_with_times(pred, lambda: kp, f.path), ref, tolerance_s=tolerance_s)
            m = _fill_source(m, clip_id, backend, detector)
            m["ref_detector"] = m["ref_detector"].where(m["ref_detector"] != "", ref_name)
            m["ref_backend"] = m["ref_backend"].where(m["ref_backend"] != "", backend)
            parts.append(m)
        rates.append((backend, clip_id, _sample_rate(kp)))
    if not parts:
        raise CLIError(f"Nothing to compare: no clip has events from both {reference!r} and another detector")
    matches = pd.concat(parts, ignore_index=True)
    metrics = _metrics_per_backend(matches, rates, ("backend", "ref_detector", "detector", "event"))
    return {"matches": matches, "metrics": metrics}


def plot_clips(
    data: str | Path,
    out_dir: str | Path,
    *,
    clips: Sequence[str] | None = None,
    backends: Sequence[str] | None = None,
    detectors: str | Sequence[str] | None = None,
    rater: str | None = None,
) -> list[Path]:
    """Write ``<out_dir>/<clip>__<backend>.png`` signal plots (see :func:`gaitbench.viz.plot_signals`)."""
    from .viz import plot_signals

    keep = detector_filter(detectors)
    names = _known_names()
    written = []
    for f in keypoint_files(data, clips, backends):
        kp = Keypoints.load(f.path)
        events = {}
        for ef in event_files(data, [f.clip], [f.backend], keep):
            table, _, detector = _read_source(ef, kp.clip_id, names)
            events[detector] = table
        truth = None
        if rater:
            labs = label_files(data, rater, [f.clip])
            truth = read_labels(labs[0].path).assign(clip_id=kp.clip_id) if labs else None
        out = Path(out_dir) / f"{f.clip}__{f.backend}.png"
        plot_signals(kp, events, truth=truth, out_path=out)
        written.append(out)
    return written


def decimate_keypoints(kp: Keypoints, step: int, offset: int = 0, backend: str | None = None) -> Keypoints:
    """Keep every ``step``-th frame, starting at array position ``offset``.

    ``frame_idx`` and ``time_s`` keep their source-video values, so events
    found on the copy are in the frame numbering and timeline of the original
    video and score directly against labels made on it (e.g. 240 fps
    slow-motion labels). The copy is named ``<backend>-<rate>fps`` unless
    ``backend`` is given. Keypoints are subsampled, not re-estimated: pose
    models that smooth over time (MediaPipe video mode) would give slightly
    different points on a real lower-rate video.
    """
    if step < 1 or not 0 <= offset < step:
        raise ValueError(f"need step >= 1 and 0 <= offset < step, got step={step}, offset={offset}")
    rate = _sample_rate(kp) or kp.fps
    new_rate = rate / step
    src_backend = str(kp.meta.get("backend", "")) or "unknown"
    name = backend or f"{src_backend}-{new_rate:.0f}fps"
    sel = slice(offset, None, step)
    meta = {**kp.meta, "fps": float(kp.fps) / step, "backend": name, "decimated_from": src_backend,
            "decimate_step": int(step), "decimate_offset": int(offset)}
    return Keypoints(clip_id=kp.clip_id, names=kp.names, xy=kp.xy[sel], conf=kp.conf[sel], time_s=kp.time_s[sel],
                     frame_idx=kp.frame_idx[sel], meta=meta)


def run_config(config: str | Path | Mapping[str, Any], *, run_id: str | None = None, overwrite: bool = False,
               echo: Callable[[str], None] = print) -> Path:
    """Run detect + score (or agree) as described by a YAML config; return the run directory.

    Config keys: ``run_id``, ``data_root`` (required); ``results_root``
    (default ``results/runs``); ``simulate`` (list of :func:`simulate_clip`
    keyword dicts, written first); ``import_openpose`` (list of
    :func:`import_openpose` keyword dicts); ``clips``, ``backends`` (list or
    ``all``); ``detectors`` (spec, see :func:`expand_detectors`);
    ``detector_params`` (``{detector: {param: value}}``); ``truth_rater``
    (labels to score against, or null); ``reference_detector`` (for
    agreement; used when there is no truth, or in addition to it);
    ``tolerance_ms`` (default 100); ``restrict_to_truth_span`` (default
    true); ``plots`` (write signal plots into the run directory);
    ``skip_existing`` (reuse events files already in the data root).

    The run directory gets ``config.yaml`` (the config as run, with the
    expanded detector list), ``env.json`` (:func:`gaitbench.env.snapshot`),
    ``detect.csv`` and ``failures.csv``, then with truth ``matches.csv``,
    ``metrics.csv``, ``metrics_by_side.csv``, ``params.csv``, and with a
    reference ``agreement.csv`` and ``agreement_matches.csv``. Agreement is
    kept out of ``metrics.csv`` so it is never mistaken for accuracy.
    """
    from .env import snapshot

    cfg = _load_config(config)
    run_id = run_id or cfg.get("run_id")
    if not run_id or "data_root" not in cfg:
        raise CLIError("A run config needs at least run_id and data_root")
    unknown = sorted(set(cfg) - _CONFIG_KEYS)
    if unknown:
        warnings.warn(f"Unknown config keys ignored: {unknown}", stacklevel=2)
    data = Path(cfg["data_root"])
    out_dir = Path(cfg.get("results_root") or DEFAULT_RESULTS_ROOT) / safe_name(run_id)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise CLIError(f"{out_dir} already exists; choose another run_id or pass --overwrite")
    clips = _all_or_list(cfg.get("clips"))
    backends = _all_or_list(cfg.get("backends"))
    detectors = expand_detectors(cfg.get("detectors"))
    tolerance_s = float(cfg.get("tolerance_ms", DEFAULT_TOLERANCE_MS)) / 1000.0
    reference = cfg.get("reference_detector")
    if reference and reference not in detectors:
        raise CLIError(f"reference_detector {reference!r} is not among the detectors run: {detectors}")

    for spec in cfg.get("simulate") or []:
        spec = dict(spec)
        kp_file, _ = simulate_clip(data, spec.pop("clip_id", "synthetic"), **spec)
        echo(f"simulated {kp_file}")
    for spec in cfg.get("import_openpose") or []:
        spec = dict(spec)
        echo(f"imported {import_openpose(data, spec.pop('json_dir'), spec.pop('clip_id'), **spec)}")

    # The record goes in first, so that a run that fails later still says what was attempted.
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {**cfg, "run_id": run_id, "detectors_expanded": detectors, "gaitbench_version": __version__,
              "data_root_resolved": str(data.resolve())}
    _write_yaml(record, out_dir / "config.yaml")
    (out_dir / "env.json").write_text(json.dumps(snapshot(), indent=2) + "\n")

    echo(f"detect: {len(detectors)} detector(s) on {data}")
    done, failures = detect_all(data, detectors, clips=clips, backends=backends, params=cfg.get("detector_params"),
                                skip_existing=bool(cfg.get("skip_existing", False)), echo=echo)
    done.to_csv(out_dir / "detect.csv", index=False)
    failures.to_csv(out_dir / "failures.csv", index=False)

    rater = cfg.get("truth_rater")
    if not rater and not reference:
        warnings.warn("Config has neither truth_rater nor reference_detector: only detection was run", stacklevel=2)
    if rater:
        restrict = bool(cfg.get("restrict_to_truth_span", True))
        result = score_events(data, rater, tolerance_s=tolerance_s, clips=clips, backends=backends,
                              detectors=detectors, restrict_to_truth_span=restrict)
        for key, filename in (("matches", "matches.csv"), ("metrics", "metrics.csv"),
                              ("metrics_by_side", "metrics_by_side.csv"), ("params", "params.csv")):
            result[key].to_csv(out_dir / filename, index=False)
        echo(f"\nScore against {rater!r} (tolerance {tolerance_s * 1000:g} ms):")
        echo(summary_table(result["metrics"]))
    if reference:
        result = agree_events(data, reference, tolerance_s=tolerance_s, clips=clips, backends=backends,
                              detectors=detectors)
        result["metrics"].to_csv(out_dir / "agreement.csv", index=False)
        result["matches"].to_csv(out_dir / "agreement_matches.csv", index=False)
        echo(f"\nAgreement with {reference!r} (tolerance {tolerance_s * 1000:g} ms; no ground truth):")
        echo(summary_table(result["metrics"]))
    if cfg.get("plots"):
        plot_clips(data, out_dir / "plots", clips=clips, backends=backends, detectors=detectors, rater=rater)

    if len(failures):
        echo(f"\n{len(failures)} detector run(s) failed (see {out_dir / 'failures.csv'}):")
        echo(failures.to_string(index=False, max_colwidth=100))
    echo(f"\nResults in {out_dir}")
    return out_dir


def summary_table(metrics: pd.DataFrame) -> str:
    """Compact text table of the main metric columns."""
    if metrics.empty:
        return "(no rows)"
    cols = [c for c in _SUMMARY_COLUMNS if c in metrics.columns]
    table = metrics.loc[:, cols].copy()
    for col in table.columns:
        if col.startswith(("recall", "precision", "f1", "p_within")):
            table[col] = table[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
        elif col.endswith(("_ms", "_frames")):
            table[col] = table[col].map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
    return table.to_string(index=False)


# ---------------------------------------------------------------------------
# helpers for the pipeline steps
# ---------------------------------------------------------------------------


def _failure(clip_id: str, backend: str, detector: str, exc: BaseException) -> dict[str, str]:
    return {"clip_id": clip_id, "backend": backend, "detector": detector, "error": f"{type(exc).__name__}: {exc}"}


def _clip_id(table: pd.DataFrame, fallback: str) -> str:
    ids = sorted({str(c) for c in table.get("clip_id", pd.Series(dtype=str)) if str(c)})
    if len(ids) > 1:
        raise FormatError(f"Expected one clip per file, found {ids}")
    return ids[0] if ids else fallback


def _read_source(f: DataFile, clip_id: str | None, names: Mapping[str, str]) -> tuple[pd.DataFrame, str, str]:
    """Read an events file; return it (with ``clip_id`` set) and its backend and detector names."""
    table = read_events(f.path)
    backend = _single(table, "backend") or f.backend
    detector = _single(table, "detector") or names.get(f.detector, f.detector)
    if clip_id is not None:
        found = _single(table, "clip_id")
        if found and found != clip_id:
            warnings.warn(f"{f.path.name}: clip_id {found!r} differs from the labels' {clip_id!r}; using the labels'",
                          stacklevel=3)
        table = table.assign(clip_id=clip_id)
    return table, backend, detector


def _single(table: pd.DataFrame, col: str) -> str:
    values = [v for v in table[col].astype(str).unique() if v] if col in table.columns else []
    return values[0] if len(values) == 1 else ""


def _fill_source(m: pd.DataFrame, clip_id: str, backend: str, detector: str) -> pd.DataFrame:
    """Name the source on every row; FN rows of an empty prediction table come back unnamed."""
    m = m.copy()
    for col, value in (("clip_id", clip_id), ("backend", backend), ("detector", detector)):
        current = m[col].fillna("").astype(str) if col in m.columns else pd.Series("", index=m.index)
        m[col] = current.where(current != "", value)
    return m


def _keypoints(data: str | Path, clip: str, backend: str, cache: dict) -> Keypoints | None:
    key = (clip, backend)
    if key not in cache:
        path = Path(data) / "keypoints" / f"{clip}__{backend}.npz"
        cache[key] = Keypoints.load(path) if path.exists() else None
    return cache[key]


def _any_keypoints(data: str | Path, clip: str, cache: dict) -> Keypoints | None:
    for f in keypoint_files(data, [clip]):
        return _keypoints(data, f.clip, f.backend, cache)
    return None


def _with_times(events: pd.DataFrame, get_kp: Callable[[], Keypoints | None], source: Path) -> pd.DataFrame:
    """Fill missing ``time_s`` from the keypoints' frame times (matching is done on time).

    ``get_kp`` is only called when a time is missing, so keypoints are not loaded needlessly.
    """
    missing = events["time_s"].isna().to_numpy()
    if not missing.any():
        return events
    kp = get_kp()
    if kp is None:
        raise CLIError(f"{source}: {int(missing.sum())} events have no time_s and there are no keypoints to look it up")
    events = events.copy()
    events.loc[missing, "time_s"] = frames_to_times(events.loc[missing, "frame"].to_numpy(), kp)
    return events


def _sample_rate(kp: Keypoints | None) -> float | None:
    if kp is None:
        return None
    if kp.n_frames > 1:
        return float(1.0 / np.median(np.diff(kp.time_s)))
    return float(kp.fps)


def _metrics_per_backend(matches: pd.DataFrame, rates: Iterable[tuple[str, str, float | None]],
                         group_by: Sequence[str]) -> pd.DataFrame:
    """event_metrics with each backend's own frame rate (a decimated copy has a lower one)."""
    from .scoring import event_metrics

    by_backend: dict[str, dict[str, float | None]] = {}
    for backend, clip_id, rate in rates:
        by_backend.setdefault(backend, {})[clip_id] = rate
    parts = []
    for backend, sub in matches.groupby("backend", sort=True):
        clip_rates = by_backend.get(backend, {})
        fps: dict[str, float] | None = {c: float(r) for c, r in clip_rates.items() if r}
        if not fps or set(sub["clip_id"]) - set(fps):
            fps = None
        parts.append(event_metrics(sub, fps=fps, group_by=group_by))
    return pd.concat(parts, ignore_index=True) if parts else event_metrics(matches, group_by=group_by)


def _parse_direction(value: int | str) -> int:
    text = str(value).strip().lower()
    if text in ("1", "+1", "ltr", "left_to_right", "left-to-right"):
        return 1
    if text in ("-1", "rtl", "right_to_left", "right-to-left"):
        return -1
    raise CLIError(f"direction must be +1/-1 or ltr/rtl, got {value!r}")


def _all_or_list(value: Any) -> list[str] | None:
    if value is None or value == "all" or value == ["all"]:
        return None
    if isinstance(value, str):
        return _spec_items(value)
    return [str(v) for v in value]


def _load_config(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    import yaml

    path = Path(config)
    if not path.is_file():
        raise CLIError(f"Config file not found: {path}")
    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise CLIError(f"{path}: expected a mapping at the top level")
    return cfg


def _write_yaml(obj: Mapping[str, Any], path: Path) -> None:
    import yaml

    path.write_text(yaml.safe_dump(json.loads(json.dumps(obj, default=str)), sort_keys=False))


def _write_tables(tables: Mapping[str, pd.DataFrame], out_dir: Path, names: Mapping[str, str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in names.items():
        tables[key].to_csv(out_dir / filename, index=False)


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _lazy(module: str, what: str, hint: str) -> Any:
    """Import an optional gaitbench module, turning a missing dependency into a CLIError."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise CLIError(f"{what} is not available: {exc}. {hint}") from exc


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def _cmd_simulate(args: argparse.Namespace) -> int:
    kp_file, lab_file = simulate_clip(
        args.data, args.clip_id, fps=args.fps, pan=args.pan, direction=args.direction, noise_px=args.noise_px,
        duration=args.duration, seed=args.seed, speed=args.speed, dropout=args.dropout,
    )
    print(f"keypoints: {kp_file}\nlabels:    {lab_file}")
    return 0


def _cmd_import_openpose(args: argparse.Namespace) -> int:
    person: str | int = int(args.person) if str(args.person).isdigit() else args.person
    out = import_openpose(
        args.data, args.json_dir, args.clip_id, fps=args.fps, width=args.width, height=args.height,
        video=args.video, person=person, backend=args.backend,
        foot_index=None if args.foot_index == "none" else args.foot_index,
    )
    kp = Keypoints.load(out)
    print(f"{out}: {kp.n_frames} frames, {len(kp.names)} keypoints, time from {kp.meta.get('time_source')}")
    return 0


def _cmd_pose(args: argparse.Namespace) -> int:
    from .video import frame_timestamps, probe_video

    bridge = _lazy("gaitbench.myogait_bridge", "The pose stage", "pip install 'gaitbench[myogait,pose]'")
    video = Path(args.video)
    clip_id = args.clip_id or video.stem
    info = probe_video(video, timestamps=False)
    for note in info.get("notes", []):
        print(f"note: {note}")
    timestamps, n_measured = None, 0
    if not args.nominal_time:
        try:
            measured = frame_timestamps(video)
        except (RuntimeError, OSError) as exc:
            warnings.warn(f"Could not read frame timestamps ({exc}); using frame_idx / fps", stacklevel=1)
        else:
            n_measured = len(measured)
            # OpenCV can decode a frame or two more than ffprobe lists; extrapolate those
            # rather than lose the whole pose run to a length mismatch.
            step = float(np.median(np.diff(measured))) if n_measured > 1 else 1.0 / (info.get("fps_nominal") or 30.0)
            pad = max(10, n_measured // 20)
            timestamps = np.concatenate([measured, measured[-1] + step * np.arange(1, pad + 1)])
    extra: dict[str, Any] = {}
    if args.max_frames:
        extra["max_frames"] = args.max_frames
    kp = bridge.extract_keypoints(video, model=args.backend, clip_id=clip_id, timestamps=timestamps, **extra)
    if timestamps is not None and kp.n_frames and kp.frame_idx.max() >= n_measured:
        warnings.warn(f"{video.name}: decoder gave frames beyond the {n_measured} timestamped ones; "
                      "their times are extrapolated", stacklevel=1)
    kp.meta["video_probe"] = {k: info.get(k) for k in ("fps_nominal", "n_frames", "duration_s", "codec", "rotation",
                                                        "is_hdr", "device", "backend")}
    out = kp.save(keypoints_path(args.data, clip_id, args.backend))
    per_frame = kp.meta.get("pose_s_per_frame")
    print(f"{out}: {kp.n_frames} frames" + (f", {1000 * per_frame:.1f} ms/frame" if per_frame else ""))
    return 0


def _cmd_decimate(args: argparse.Namespace) -> int:
    path = keypoints_path(args.data, args.clip, args.backend)
    if not path.exists():
        raise CLIError(f"No keypoints file {path}")
    kp = decimate_keypoints(Keypoints.load(path), args.step, args.offset, args.out_backend)
    out = kp.save(keypoints_path(args.data, kp.clip_id, kp.meta["backend"]))
    print(f"{out}: {kp.n_frames} frames at {kp.fps:g} fps (backend {kp.meta['backend']!r})")
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    detectors = expand_detectors(args.detectors)
    params = json.loads(args.params) if args.params else None
    done, failures = detect_all(args.data, detectors, clips=args.clip, backends=args.backend, params=params,
                                skip_existing=args.skip_existing)
    written = int((~done["skipped"].astype(bool)).sum()) if len(done) else 0
    print(f"\n{written} events file(s) written, {len(failures)} failure(s)")
    if len(failures):
        print(failures.to_string(index=False, max_colwidth=100))
    if len(failures) and (args.strict or not len(done)):
        return 1
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    result = score_events(args.data, args.rater, tolerance_s=args.tolerance_ms / 1000.0, clips=args.clip,
                          backends=args.backend, detectors=args.detectors,
                          restrict_to_truth_span=not args.all_predictions)
    out = Path(args.out) if args.out else Path(args.data) / "results" / f"score__{safe_name(args.rater)}"
    _write_tables(result, out, {"matches": "matches.csv", "metrics": "metrics.csv",
                                "metrics_by_side": "metrics_by_side.csv", "params": "params.csv"})
    print(f"Score against {args.rater!r} (tolerance {args.tolerance_ms:g} ms):")
    print(summary_table(result["metrics"]))
    print(f"\nWritten to {out}")
    return 0


def _cmd_agree(args: argparse.Namespace) -> int:
    result = agree_events(args.data, args.reference, tolerance_s=args.tolerance_ms / 1000.0, clips=args.clip,
                          backends=args.backend, detectors=args.detectors)
    out = Path(args.out) if args.out else Path(args.data) / "results" / f"agree__{safe_name(args.reference)}"
    _write_tables(result, out, {"matches": "agreement_matches.csv", "metrics": "agreement.csv"})
    print(f"Agreement with {args.reference!r} (tolerance {args.tolerance_ms:g} ms; no ground truth):")
    print(summary_table(result["metrics"]))
    print(f"\nWritten to {out}")
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    out = Path(args.out) if args.out else Path(args.data) / "plots"
    written = plot_clips(args.data, out, clips=args.clip, backends=args.backend, detectors=args.detectors,
                         rater=args.rater)
    if not written:
        raise CLIError(f"No keypoints files in {Path(args.data) / 'keypoints'} to plot")
    for path in written:
        print(path)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    run_config(args.config, run_id=args.run_id, overwrite=args.overwrite)
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    from .video import frame_timestamps, probe_video

    info = probe_video(args.video, backend=args.backend, timestamps=not args.no_timestamps)
    _print_json(info)
    if args.timestamps_csv:
        ts = frame_timestamps(args.video, backend=args.backend)
        pd.DataFrame({"frame": np.arange(len(ts)), "time_s": ts}).to_csv(args.timestamps_csv, index=False)
        print(f"{len(ts)} frame times written to {args.timestamps_csv}", file=sys.stderr)
    return 0


def _cmd_cfr(args: argparse.Namespace) -> int:
    from .video import cfr_command

    cmd = cfr_command(args.src, args.dst, args.fps, crf=args.crf, audio=not args.no_audio)
    if not args.run:
        print(" ".join(_quote(c) for c in cmd))
        return 0
    if shutil.which(cmd[0]) is None:
        raise CLIError("ffmpeg not found on PATH (macOS: brew install ffmpeg)")
    return subprocess.run(cmd).returncode


def _cmd_label(args: argparse.Namespace) -> int:
    app = _lazy("gaitbench.labelling.app", "The labeller", "It needs OpenCV: pip install 'gaitbench[video]'")
    video = Path(args.video)
    clip_id = args.clip_id or video.stem
    if args.out:
        out_csv = Path(args.out)
    elif args.data:
        out_csv = labels_path(args.data, clip_id, args.rater)
    else:
        raise CLIError("Give --data (labels go to <data>/labels/) or --out")
    kp_path = args.keypoints
    if kp_path is None and args.data and args.backend:
        kp_path = keypoints_path(args.data, clip_id, args.backend)
    app.run_labeller(video, out_csv, args.rater, clip_id=clip_id, keypoints_path=kp_path)
    print(f"labels: {out_csv}")
    return 0


def _realsense() -> Any:
    return _lazy("gaitbench.capture.realsense", "RealSense capture",
                 "Install pyrealsense2 (Linux/Windows) or pyrealsense2-macosx (Apple Silicon): "
                 "pip install 'gaitbench[realsense]'")


def _cmd_realsense_modes(args: argparse.Namespace) -> int:
    modes = _realsense().list_modes()
    if isinstance(modes, pd.DataFrame):
        print(modes.to_string(index=False))
    elif isinstance(modes, list) and modes and isinstance(modes[0], Mapping):
        print(pd.DataFrame(modes).to_string(index=False))
    elif isinstance(modes, Mapping):
        _print_json(modes)
    else:
        for mode in modes or []:
            print(mode)
    return 0


def _cmd_realsense_record(args: argparse.Namespace) -> int:
    rs = _realsense()
    default_w, default_h, default_fps = rs.DEFAULT_COLOR
    fps = args.fps or default_fps
    color = (args.width or default_w, args.height or default_h, fps)
    # Depth at the colour frame rate, so every colour frame has a depth frame.
    depth = None if args.no_depth else (rs.DEFAULT_DEPTH[0], rs.DEFAULT_DEPTH[1], fps)
    result = rs.record(args.out, args.duration, color=color, depth=depth, clip_id=args.clip_id, serial=args.serial)
    print(result)
    return 0


def _cmd_realsense_export(args: argparse.Namespace) -> int:
    paths = _realsense().bag_to_video(args.bag, args.out_dir)
    if isinstance(paths, Mapping):
        for key, value in paths.items():
            print(f"{key}: {value}")
    else:
        for p in paths if isinstance(paths, (list, tuple)) else [paths]:
            print(p)
    return 0


def _cmd_list_detectors(args: argparse.Namespace) -> int:
    from .detectors import list_detectors

    for name in list_detectors():
        print(name)
    return 0


def _cmd_env(args: argparse.Namespace) -> int:
    from .env import snapshot

    _print_json(snapshot())
    return 0


def _quote(text: str) -> str:
    import shlex

    return shlex.quote(text)


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser for :func:`main`."""
    parser = argparse.ArgumentParser(
        prog="gaitbench",
        description="Run open-source gait event detectors on the same keypoints and score them.",
        epilog="Data root layout: <data>/keypoints, <data>/events, <data>/labels. Relative paths are "
               "resolved against the current directory.",
    )
    parser.add_argument("--version", action="version", version=f"gaitbench {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="more log output (-vv: debug, tracebacks)")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    def data_arg(p: argparse.ArgumentParser, required: bool = True) -> None:
        p.add_argument("--data", required=required, help="data root (holds keypoints/, events/, labels/)")

    def selection(p: argparse.ArgumentParser, detectors_default: str | None) -> None:
        p.add_argument("--clip", action="append", help="clip id (repeatable; default: all clips found)")
        p.add_argument("--backend", action="append", help="pose backend (repeatable; default: all)")
        p.add_argument("--detectors", default=detectors_default,
                       help="comma list of detector names and/or groups: all, ours (zeni, stenum), myogait")

    def tolerance(p: argparse.ArgumentParser) -> None:
        p.add_argument("--tolerance-ms", type=float, default=DEFAULT_TOLERANCE_MS,
                       help=f"match tolerance (default {DEFAULT_TOLERANCE_MS:g} ms)")

    p = sub.add_parser("simulate", help="write a synthetic walk's keypoints and truth labels")
    data_arg(p)
    p.add_argument("--clip-id", default="synthetic")
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--pan", default="0", help="camera speed in m/s, or 'follow' to keep the walker centred")
    p.add_argument("--direction", default="1", help="+1/ltr (left to right in the image) or -1/rtl")
    p.add_argument("--noise-px", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.0, help="probability a point is missing")
    # 6 s, not the simulator's 5 s: at 5 s a heel strike falls on the last frame, where
    # no peak-based detector can find it, so recall would look worse than it is.
    p.add_argument("--duration", type=float, default=6.0, help="seconds (default 6)")
    p.add_argument("--speed", type=float, default=None, help="walking speed in m/s (default 1.3)")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_simulate)

    p = sub.add_parser("import-openpose", help="convert a directory of OpenPose JSON files to keypoints")
    data_arg(p)
    p.add_argument("--json-dir", required=True)
    p.add_argument("--clip-id", required=True)
    p.add_argument("--fps", type=float)
    p.add_argument("--width", type=float)
    p.add_argument("--height", type=float)
    p.add_argument("--video", help="source video: fills missing fps/size and gives real frame times")
    p.add_argument("--person", default="best", help="'best' (track the walker) or an index into 'people'")
    p.add_argument("--backend", default="openpose")
    p.add_argument("--foot-index", choices=("none", "big_toe", "mean_toes"), default="none",
                   help="also write *_FOOT_INDEX from the toes (default: no)")
    p.set_defaults(func=_cmd_import_openpose)

    p = sub.add_parser("pose", help="run a CPU pose model on a video (through myogait)")
    data_arg(p)
    p.add_argument("--video", required=True)
    p.add_argument("--clip-id", help="default: the video file name without extension")
    p.add_argument("--backend", default="mediapipe", help="myogait model: mediapipe, rtmw, yolo, ...")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--nominal-time", action="store_true",
                   help="use frame_idx / fps instead of the video's real frame timestamps")
    p.set_defaults(func=_cmd_pose)

    p = sub.add_parser("decimate", help="keep every n-th keypoint frame (e.g. 240 fps -> 60 fps copy)")
    data_arg(p)
    p.add_argument("--clip", required=True)
    p.add_argument("--backend", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--out-backend", help="backend name of the copy (default <backend>-<rate>fps)")
    p.set_defaults(func=_cmd_decimate)

    p = sub.add_parser("detect", help="run detectors on keypoints files and write events files")
    data_arg(p)
    selection(p, "ours")
    p.add_argument("--params", help='detector parameters as JSON, e.g. \'{"zeni": {"cutoff_hz": 5}}\'')
    p.add_argument("--skip-existing", action="store_true", help="keep events files that already exist")
    p.add_argument("--strict", action="store_true", help="exit 1 if any detector fails")
    p.set_defaults(func=_cmd_detect)

    p = sub.add_parser("score", help="score events against labels (matches, metrics, temporal parameters)")
    data_arg(p)
    p.add_argument("--rater", default=TRUTH_RATER, help="labels to use as truth ('truth' for synthetic clips)")
    selection(p, None)
    tolerance(p)
    p.add_argument("--all-predictions", action="store_true",
                   help="also count predictions outside the labelled time span as false positives")
    p.add_argument("--out", help="results directory (default <data>/results/score__<rater>)")
    p.set_defaults(func=_cmd_score)

    p = sub.add_parser("agree", help="agreement of every detector with a reference detector (no truth)")
    data_arg(p)
    p.add_argument("--reference", default="zeni", help="reference detector (default zeni)")
    selection(p, None)
    tolerance(p)
    p.add_argument("--out", help="results directory (default <data>/results/agree__<reference>)")
    p.set_defaults(func=_cmd_agree)

    p = sub.add_parser("plot", help="foot-signal plot with events, one PNG per clip and backend")
    data_arg(p)
    selection(p, None)
    p.add_argument("--rater", help="also draw this rater's labels (e.g. truth)")
    p.add_argument("--out", help="output directory (default <data>/plots)")
    p.set_defaults(func=_cmd_plot)

    p = sub.add_parser("run", help="detect + score/agree from a YAML config into results/runs/<run_id>/")
    p.add_argument("config")
    p.add_argument("--run-id", help="override the config's run_id")
    p.add_argument("--overwrite", action="store_true", help="reuse an existing run directory")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("probe", help="print video facts (size, fps, VFR, rotation) as JSON")
    p.add_argument("video")
    p.add_argument("--backend", choices=("auto", "ffprobe", "opencv"), default="auto")
    p.add_argument("--no-timestamps", action="store_true", help="skip the per-frame VFR check")
    p.add_argument("--timestamps-csv", help="also write every frame's time to this CSV")
    p.set_defaults(func=_cmd_probe)

    p = sub.add_parser("cfr", help="print (or --run) the ffmpeg command for a constant-frame-rate copy")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--fps", required=True)
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--run", action="store_true", help="run ffmpeg instead of printing the command")
    p.set_defaults(func=_cmd_cfr)

    p = sub.add_parser("label", help="label heel strikes and toe offs frame by frame")
    p.add_argument("--video", required=True)
    p.add_argument("--rater", required=True, help="your rater id, e.g. initials")
    data_arg(p, required=False)
    p.add_argument("--out", help="labels CSV (default <data>/labels/<clip>__<rater>.csv)")
    p.add_argument("--clip-id")
    p.add_argument("--keypoints", help="keypoints file used only to centre the foot zoom (never drawn)")
    p.add_argument("--backend", help="use <data>/keypoints/<clip>__<backend>.npz for the foot zoom")
    p.set_defaults(func=_cmd_label)

    p = sub.add_parser("realsense-modes", help="list the connected RealSense camera's stream modes")
    p.set_defaults(func=_cmd_realsense_modes)

    p = sub.add_parser("realsense-record", help="record one RealSense pass (colour + depth)")
    p.add_argument("--out", required=True, help="folder for the recording (.bag or .db3, by SDK version) and its sidecar")
    p.add_argument("--duration", type=float, required=True, help="seconds")
    p.add_argument("--clip-id", help="name of the recording (default: a timestamp)")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--fps", type=int)
    p.add_argument("--no-depth", action="store_true")
    p.add_argument("--serial", help="camera serial number, if several are connected")
    p.set_defaults(func=_cmd_realsense_record)

    p = sub.add_parser("realsense-export", help="export a .bag's streams to video files")
    p.add_argument("bag")
    p.add_argument("out_dir")
    p.set_defaults(func=_cmd_realsense_export)

    p = sub.add_parser("list-detectors", help="print the available detector names")
    p.set_defaults(func=_cmd_list_detectors)

    p = sub.add_parser("env", help="print versions and platform facts as JSON")
    p.set_defaults(func=_cmd_env)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``gaitbench`` console script.

    Parameters
    ----------
    argv:
        Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit status: 0 on success, 1 on an error (printed to stderr), 2 on
        bad usage (from argparse).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    try:
        return int(args.func(args) or 0)
    except (CLIError, FormatError, FileNotFoundError, ImportError, ValueError, RuntimeError, KeyError) as exc:
        if args.verbose >= 2:
            traceback.print_exc()
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"gaitbench {args.command}: error: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

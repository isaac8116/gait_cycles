"""Bridge between gaitbench keypoints and myogait's pivot pipeline.

myogait (pinned 0.9.1) works on a "pivot" dict: ``meta`` (fps, frame size),
an ``extraction`` block naming the pose model, and a ``frames`` list where
each frame holds ``frame_idx``, ``time_s`` and ``landmarks`` as
``{NAME: {"x", "y", "visibility"}}`` with x and y **normalised** by the frame
width and height. This module converts between that pivot and
:class:`~gaitbench.formats.Keypoints`, and runs myogait's validated
processing chain (``run_pipeline``: Butterworth filter, angles, sign
canonicalisation, event detection, cycle segmentation, direction filter) so
that the events scored by the harness are exactly what myogait would report.

Things this module takes care of, found by reading myogait 0.9.1:

* **Frame numbering.** ``detect_events`` maps detector output from array
  positions to ``frame_idx`` (fixed in myogait 0.8.3), but
  ``event_consensus`` does not: its frames are array positions. The bridge
  runs consensus on a positional view of the frames and maps the result to
  source frame indices itself, so every events table uses source-video frame
  indices whatever the method.
* **Missing points** are written the way ``myogait.extract`` writes them: a
  frame with no detected point has ``landmarks == {}``; a single missing
  point is ``{"x": nan, "y": nan, "visibility": 0.0}``.
* **Mirroring.** ``extract(flip_if_right=True)`` mirrors right-to-left walks
  and swaps the left/right labels. Harness keypoints must stay in
  source-image pixels with the walker's own sides, so
  :func:`extract_keypoints` extracts unflipped, and
  :func:`pivot_to_keypoints` undoes a flip if it finds one.
* **Estimated feet.** ``extract`` fills missing heel/foot-index points from
  ankle-knee geometry (visibility 0.5). They are not detector output, so
  :func:`pivot_to_keypoints` drops them by default. Otherwise a COCO-17
  backend would appear to track feet it never saw.
* **No silent installs.** gaitkit installs onnxruntime/tensorflow with pip
  on first use of ``gk_intellevent``/``gk_deepevent`` unless
  ``GAITKIT_AUTO_INSTALL_DEPS`` is off. The bridge turns it off for its own
  calls (unless the caller has set the variable) and raises
  :class:`MissingDependencyError` instead.
* **Quiet runs.** myogait and gaitkit log records produced during a call are
  captured and re-emitted on this module's logger one level lower
  (WARNING -> INFO), because most of them are about cycle averaging and
  would flood a sweep over many clips and methods. The captured messages and
  myogait's quality report are also attached to the events table as
  ``df.attrs["myogait"]``.

myogait and gaitkit are optional dependencies: importing this module never
imports them; every public function does so lazily and raises
:class:`MissingDependencyError` with an install hint if they are absent.
None of this module's state is thread-safe (myogait itself keeps module
globals during event detection).
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import time
import warnings
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .formats import CANONICAL_KEYPOINTS, FormatError, Keypoints, frames_to_times, make_events

logger = logging.getLogger(__name__)

CONSENSUS = "consensus"
# Default consensus members, chosen by the evaluation plan (E2).
DEFAULT_CONSENSUS_METHODS: tuple[str, ...] = ("zeni", "gk_bike", "gk_zeni", "gk_oconnor")

_EVENT_KEYS: dict[str, tuple[str, str]] = {
    "left_hs": ("L", "HS"),
    "right_hs": ("R", "HS"),
    "left_to": ("L", "TO"),
    "right_to": ("R", "TO"),
}

# Optional packages that individual myogait/gaitkit methods need. Used to name
# the missing package in error messages.
_METHOD_DEPENDENCIES: dict[str, str] = {
    "gk_intellevent": "onnxruntime",
    "gk_deepevent": "tensorflow (with keras and h5py)",
}

# Python package behind each myogait pose model, recorded as model provenance.
_MODEL_PACKAGES: dict[str, str] = {
    "mediapipe": "mediapipe",
    "yolo": "ultralytics",
    "rtmw": "rtmlib",
    "hrnet": "mmpose",
    "mmpose": "mmpose",
    "vitpose": "transformers",
    "sapiens": "torch",
    "detectron2": "detectron2",
    "alphapose": "alphapose",
}

# Visibility myogait assigns to geometrically estimated foot points.
_ESTIMATED_FOOT_VISIBILITY = 0.5
_ESTIMATED_FOOT_PARTS = ("HEEL", "FOOT_INDEX")
_CAPTURED_LOGGERS = ("myogait", "gaitkit")


class MissingDependencyError(RuntimeError, ImportError):
    """myogait, gaitkit or a package one of their methods needs is not installed.

    Subclasses both :class:`RuntimeError` and :class:`ImportError`, so either
    ``except`` clause catches it.
    """


class MyogaitRunError(RuntimeError):
    """myogait raised while processing a clip (for a reason other than a missing package)."""


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def is_available() -> bool:
    """Return True if myogait can be imported."""
    try:
        _require_myogait()
    except MissingDependencyError:
        return False
    return True


def myogait_version() -> str:
    """Installed myogait version (raises :class:`MissingDependencyError` if absent)."""
    return str(getattr(_require_myogait(), "__version__", "unknown"))


def list_methods() -> list[str]:
    """Event methods this bridge can run.

    Returns
    -------
    list of str
        ``myogait.list_event_methods()`` (which includes the ``gk_*`` gaitkit
        methods when gaitkit is installed) followed by ``"consensus"``.
        Methods whose own optional packages are missing (``gk_intellevent``
        needs onnxruntime, ``gk_deepevent`` needs tensorflow) are still
        listed; running them raises :class:`MissingDependencyError`.
    """
    _require_myogait()
    from myogait.events import list_event_methods

    methods = [str(m) for m in list_event_methods()]
    return methods + ([CONSENSUS] if CONSENSUS not in methods else [])


# ---------------------------------------------------------------------------
# Keypoints <-> pivot
# ---------------------------------------------------------------------------


def keypoints_to_pivot(kp: Keypoints, *, derive_foot_index: bool = True) -> dict:
    """Build a myogait pivot dict from harness keypoints.

    The pivot has the layout ``myogait.extract`` produces: ``meta`` from
    ``myogait.schema.create_empty``, an ``extraction`` block naming the
    backend as the model, and one entry in ``frames`` per keypoint frame with
    ``frame_idx`` (source-video index), ``time_s``, ``confidence`` (mean
    visibility of the points present) and ``landmarks``. Coordinates are
    divided by the frame width and height. All 33 MediaPipe names are
    written (NaN where the backend has no such point, as ``extract`` does for
    COCO models); ``LEFT_BIG_TOE`` and the other extended foot names are
    written only if the keypoints have them.

    Parameters
    ----------
    kp:
        Keypoints with ``meta['width']`` and ``meta['height']`` set.
    derive_foot_index:
        If the keypoints have toe points but no ``*_FOOT_INDEX`` (OpenPose,
        RTMW), write ``FOOT_INDEX`` as the mean of the available toes, as
        ``extract`` does for RTMW and Sapiens. gaitkit methods use
        ``FOOT_INDEX`` as the toe marker. :func:`pivot_to_keypoints` drops
        derived points again.

    Returns
    -------
    dict
        A new pivot sharing no mutable state with ``kp``. The original
        keypoint names, clip id and metadata are kept under
        ``pivot['extraction']['gaitbench']`` so the conversion can be undone
        exactly.
    """
    _require_myogait()
    from myogait.constants import MP_LANDMARK_NAMES
    from myogait.schema import create_empty

    width, height = _frame_size(kp.meta, what=f"keypoints {kp.clip_id!r}")
    backend = str(kp.meta.get("backend", "") or "unknown")

    xy = kp.xy.copy()
    conf = np.clip(np.nan_to_num(kp.conf, nan=0.0), 0.0, 1.0)
    missing = np.isnan(xy).any(axis=2)
    norm = np.empty_like(xy)
    norm[..., 0] = xy[..., 0] / width
    norm[..., 1] = xy[..., 1] / height

    names = list(kp.names)
    derived: list[str] = []
    if derive_foot_index:
        for prefix in ("LEFT_", "RIGHT_"):
            target = prefix + "FOOT_INDEX"
            toes = [j for j, n in enumerate(names) if n in (prefix + "BIG_TOE", prefix + "SMALL_TOE")]
            if target in names or not toes:
                continue
            toe_xy = np.where(missing[:, toes, None], np.nan, norm[:, toes, :])
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices: toe missing in that frame
                mid = np.nanmean(toe_xy, axis=1)
            toe_conf = np.where(missing[:, toes], np.inf, conf[:, toes]).min(axis=1)
            mid_missing = np.isnan(mid).any(axis=1)
            norm = np.concatenate([norm, mid[:, None, :]], axis=1)
            conf = np.concatenate([conf, np.where(mid_missing, 0.0, toe_conf)[:, None]], axis=1)
            missing = np.concatenate([missing, mid_missing[:, None]], axis=1)
            names.append(target)
            derived.append(target)

    mp_names = list(MP_LANDMARK_NAMES)
    extra_names = [n for n in names if n not in mp_names]
    order = mp_names + extra_names
    col = {n: j for j, n in enumerate(names)}

    frames = []
    for i in range(kp.n_frames):
        present = ~missing[i]
        if not present.any():
            # extract() writes an undetected frame as an empty landmarks dict.
            frames.append({"frame_idx": int(kp.frame_idx[i]), "time_s": float(kp.time_s[i]), "landmarks": {}, "confidence": 0.0})
            continue
        landmarks: dict[str, dict[str, float]] = {}
        for name in order:
            j = col.get(name)
            if j is None:
                if name in mp_names:
                    landmarks[name] = {"x": math.nan, "y": math.nan, "visibility": 0.0}
                continue
            if missing[i, j]:
                landmarks[name] = {"x": math.nan, "y": math.nan, "visibility": 0.0}
            else:
                landmarks[name] = {"x": float(norm[i, j, 0]), "y": float(norm[i, j, 1]), "visibility": float(conf[i, j])}
        frames.append(
            {
                "frame_idx": int(kp.frame_idx[i]),
                "time_s": float(kp.time_s[i]),
                "landmarks": landmarks,
                "confidence": float(np.mean(conf[i][present])),
            }
        )

    pivot = create_empty(
        video_path=str(kp.meta.get("source", "")),
        fps=float(kp.fps),
        width=int(round(width)),
        height=int(round(height)),
        n_frames=kp.n_frames,
    )
    pivot["frames"] = frames
    pivot["extraction"] = {
        "model": backend,
        "model_detail": f"gaitbench:{backend}",
        "keypoint_format": "mediapipe33",
        "n_landmarks": len(mp_names),
        "landmark_names": mp_names,
        "direction_detected": "unknown",
        "was_flipped": False,
        "inversions_corrected": False,
        "gaitbench": {
            "clip_id": kp.clip_id,
            "names": list(kp.names),
            "derived": derived,
            # JSON round trip: a deep copy that also proves the pivot stays serialisable.
            "meta": json.loads(json.dumps(kp.meta)),
        },
    }
    return pivot


def pivot_to_keypoints(
    pivot: Mapping[str, Any],
    clip_id: str,
    backend: str,
    *,
    keep_estimated_feet: bool = False,
) -> Keypoints:
    """Convert a myogait pivot (from ``extract`` or :func:`keypoints_to_pivot`) to keypoints.

    Parameters
    ----------
    pivot:
        Pivot dict with ``meta`` (fps, width, height) and ``frames``. Not modified.
    clip_id:
        Clip name for the keypoints.
    backend:
        Pose backend name stored in ``meta['backend']``.
    keep_estimated_feet:
        ``extract`` replaces missing heel / foot-index points by an estimate
        from ankle-knee geometry (visibility exactly 0.5, frame flagged
        ``foot_landmarks_source == 'estimated'``). By default those points are
        marked missing, because no model saw them; True keeps them.

    Returns
    -------
    Keypoints
        Pixel coordinates in the source image (a myogait left-right flip is
        undone), only names in :data:`~gaitbench.formats.CANONICAL_KEYPOINTS`,
        ``frame_idx`` and ``time_s`` from the pivot frames (``frame_idx``
        scaled back to source frames when ``extract`` subsampled with
        ``experimental={'target_fps': ...}``). ``meta`` holds fps, width,
        height, source, backend, ``tool='myogait'``, tool_version, model,
        model_detail and a ``myogait_extraction`` summary.
    """
    frames = list(pivot.get("frames") or [])
    if not frames:
        raise FormatError("Pivot has no frames (did extraction find anyone?)")
    pmeta = dict(pivot.get("meta") or {})
    extraction = dict(pivot.get("extraction") or {})
    stored = dict(extraction.get("gaitbench") or {})
    width, height = _frame_size(pmeta, what="pivot meta")
    flipped = extraction.get("was_flipped") is True

    if stored.get("names"):
        names = [n for n in stored["names"] if n in CANONICAL_KEYPOINTS]
    else:
        seen = set(extraction.get("landmark_names") or [])
        for f in frames:
            seen.update((f.get("landmarks") or {}).keys())
        names = [n for n in CANONICAL_KEYPOINTS if n in seen]
    if not names:
        raise FormatError("Pivot has no landmark that maps to a canonical keypoint name")

    n_frames, n_kp = len(frames), len(names)
    xy = np.full((n_frames, n_kp, 2), np.nan)
    conf = np.zeros((n_frames, n_kp))
    frame_idx = np.empty(n_frames, dtype=np.int64)
    time_s = np.empty(n_frames)
    fps = float(pmeta.get("fps") or 0.0)
    n_dropped = 0
    for i, f in enumerate(frames):
        frame_idx[i] = int(f.get("frame_idx", i))
        t = f.get("time_s")
        time_s[i] = float(t) if t is not None else (frame_idx[i] / fps if fps > 0 else float(i))
        lm = f.get("landmarks") or {}
        estimated = f.get("foot_landmarks_source") == "estimated"
        for j, name in enumerate(names):
            # A mirrored pivot labels the walker's left foot as RIGHT_*.
            entry = lm.get(_swap_side(name) if flipped else name)
            if not entry:
                continue
            x, y, vis = _entry_values(entry)
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            if (
                estimated
                and not keep_estimated_feet
                and name.endswith(_ESTIMATED_FOOT_PARTS)
                and vis == _ESTIMATED_FOOT_VISIBILITY
            ):
                n_dropped += 1
                continue
            if flipped:
                x = 1.0 - x
            xy[i, j] = (x * width, y * height)
            conf[i, j] = min(max(vis, 0.0), 1.0) if np.isfinite(vis) else 0.0

    stride = int(((extraction.get("experimental") or {}).get("frame_stride")) or 1)
    if stride > 1:
        # extract() numbers subsampled frames 0, 1, 2...; the source frame is stride times that.
        frame_idx = frame_idx * stride

    meta: dict[str, Any] = dict(stored.get("meta") or {})
    meta.update(
        {
            "fps": fps if fps > 0 else None,
            "width": int(round(width)),
            "height": int(round(height)),
            "source": str(pmeta.get("video_path") or meta.get("source") or ""),
            "backend": str(backend),
            "tool": "myogait",
            "tool_version": str(pivot.get("myogait_version") or _installed_myogait_version()),
            "model": str(extraction.get("model", "")),
            "model_detail": str(extraction.get("model_detail", "")),
            "myogait_extraction": _json_safe(
                {
                    k: extraction[k]
                    for k in (
                        "keypoint_format",
                        "direction_detected",
                        "was_flipped",
                        "inversions_corrected",
                        "timestamp",
                        "auxiliary_format",
                        "experimental",
                    )
                    if k in extraction
                }
            ),
            "myogait_estimated_foot_points_dropped": n_dropped,
        }
    )
    if meta["fps"] is None:
        del meta["fps"]
    return Keypoints(
        clip_id=str(clip_id),
        names=tuple(names),
        xy=xy,
        conf=conf,
        time_s=time_s,
        frame_idx=frame_idx,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Running myogait
# ---------------------------------------------------------------------------


def run_myogait(
    kp: Keypoints,
    method: str = "zeni",
    *,
    butterworth_cutoff: float = 4.0,
    min_cycle_duration_s: float = 0.8,
    max_cycle_duration_s: float = 1.6,
    direction_filter: bool = True,
    consensus_methods: Sequence[str] | None = None,
    consensus_tolerance_frames: int = 3,
) -> dict:
    """Run myogait's validated pipeline on keypoints and return its raw result.

    For a single method this is ``myogait.pipeline.run_pipeline`` on
    :func:`keypoints_to_pivot` of ``kp`` with ``analyze=False``. For
    ``method='consensus'`` the same steps run (normalize, compute_angles,
    canonicalize_angle_signs), then ``event_consensus``, then cycle
    segmentation, the direction filter and myogait's quality diagnostics on
    the consensus events.

    Parameters
    ----------
    kp:
        Keypoints of one clip.
    method:
        A name from :func:`list_methods`.
    butterworth_cutoff, min_cycle_duration_s, max_cycle_duration_s, direction_filter:
        Passed to ``run_pipeline`` (same meaning and defaults).
    consensus_methods:
        Members of the consensus (default :data:`DEFAULT_CONSENSUS_METHODS`).
    consensus_tolerance_frames:
        ``event_consensus`` clustering tolerance in frames (myogait default 3).

    Returns
    -------
    dict
        ``{"data", "cycles", "stats", "quality", "source_type"}`` as from
        ``run_pipeline``. Event and cycle frames (``data['events']``,
        ``cycles['cycles'][i]['start_frame']`` ...) are source-video frame
        indices for every method, consensus included.

    Raises
    ------
    ValueError
        Unknown method.
    MissingDependencyError
        myogait, gaitkit or a package the method needs is not installed.
    MyogaitRunError
        myogait failed on this clip for another reason.
    """
    result, _ = _run(
        kp,
        method,
        butterworth_cutoff=butterworth_cutoff,
        min_cycle_duration_s=min_cycle_duration_s,
        max_cycle_duration_s=max_cycle_duration_s,
        direction_filter=direction_filter,
        consensus_methods=consensus_methods,
        consensus_tolerance_frames=consensus_tolerance_frames,
    )
    return result


def detect_events_myogait(
    kp: Keypoints,
    method: str,
    *,
    butterworth_cutoff: float = 4.0,
    min_cycle_duration_s: float = 0.8,
    max_cycle_duration_s: float = 1.6,
    direction_filter: bool = True,
    consensus_methods: Sequence[str] | None = None,
    consensus_tolerance_frames: int = 3,
) -> pd.DataFrame:
    """Detect heel strikes and toe offs with one myogait method.

    Runs :func:`run_myogait` and converts ``data['events']`` to the harness
    events table. The direction filter acts on cycles only, so it does not
    remove events.

    Parameters
    ----------
    kp:
        Keypoints of one clip.
    method:
        A name from :func:`list_methods`, without the ``myogait:`` prefix.
    butterworth_cutoff, min_cycle_duration_s, max_cycle_duration_s, direction_filter, \
consensus_methods, consensus_tolerance_frames:
        See :func:`run_myogait`.

    Returns
    -------
    pandas.DataFrame
        Validated events with ``clip_id=kp.clip_id``, ``tool='myogait'``,
        ``tool_version`` the myogait version, ``backend=kp.meta['backend']``,
        ``detector='myogait:<method>'``, ``frame`` the source-video frame
        index, ``time_s`` from :func:`~gaitbench.formats.frames_to_times` and
        ``confidence`` as myogait reports it (prominence ratio for the
        native methods, the vote fraction for consensus).
        ``df.attrs['myogait']`` holds myogait's quality report, its captured
        log messages and any Python warnings raised during the run.

    Raises
    ------
    ValueError
        Unknown method.
    MissingDependencyError
        myogait, gaitkit or a package the method needs is not installed.
    MyogaitRunError
        myogait failed on this clip for another reason.
    """
    result, captured = _run(
        kp,
        method,
        butterworth_cutoff=butterworth_cutoff,
        min_cycle_duration_s=min_cycle_duration_s,
        max_cycle_duration_s=max_cycle_duration_s,
        direction_filter=direction_filter,
        consensus_methods=consensus_methods,
        consensus_tolerance_frames=consensus_tolerance_frames,
    )
    events = (result.get("data") or {}).get("events") or {}
    df = _events_table(events, kp, method)
    df.attrs["myogait"] = {
        "method": method,
        "quality": _json_safe(result.get("quality")),
        "n_cycles": len((result.get("cycles") or {}).get("cycles") or []),
        **captured,
    }
    logger.debug("myogait:%s on %s: %d events", method, kp.clip_id, len(df))
    return df


def extract_keypoints(
    video_path: str | Path,
    model: str = "mediapipe",
    clip_id: str | None = None,
    *,
    timestamps: Sequence[float] | np.ndarray | None = None,
    keep_estimated_feet: bool = False,
    **kwargs: Any,
) -> Keypoints:
    """Run myogait's pose stage on a video and return harness keypoints.

    Calls ``myogait.extract`` with ``flip_if_right=False`` (keypoints stay in
    source-image pixels with the walker's own sides) and
    ``show_progress=False`` unless overridden in ``kwargs``. MediaPipe runs
    on the CPU and downloads its model to ``~/.myogait/models`` on first use.

    Parameters
    ----------
    video_path:
        Video file readable by OpenCV.
    model:
        myogait pose model name (``"mediapipe"``, ``"yolo"``, ``"rtmw"``, ...).
        Also used as the backend name.
    clip_id:
        Clip name; defaults to the file stem.
    timestamps:
        Optional presentation time of every source frame (e.g. from
        ``gaitbench.video.frame_timestamps``). ``extract`` writes nominal
        times ``frame_idx / fps``, which are wrong for variable-frame-rate
        phone video; when given, ``time_s`` is looked up here by frame index.
    keep_estimated_feet:
        See :func:`pivot_to_keypoints`.
    **kwargs:
        Passed to ``myogait.extract`` (``max_frames``, ``correct_inversions``,
        ``experimental``, model options ...).

    Returns
    -------
    Keypoints
        With ``meta['backend'] = model``, myogait and model package versions,
        and ``pose_runtime_s`` / ``pose_s_per_frame`` (wall clock).

    Raises
    ------
    MissingDependencyError
        myogait or the pose model's package is not installed.
    """
    mg = _require_myogait()
    from myogait.extract import extract

    path = Path(video_path)
    kwargs.setdefault("flip_if_right", False)
    kwargs.setdefault("show_progress", False)
    t0 = time.perf_counter()
    with _captured_output() as captured:
        try:
            pivot = extract(str(path), model=model, **kwargs)
        except ImportError as exc:
            # myogait reports any MediaPipe setup failure as "not installed"; say what really failed.
            problem = _mediapipe_native_problem() if model == "mediapipe" else None
            detail = f" Underlying problem: {problem}" if problem else ""
            raise MissingDependencyError(
                f"myogait pose model {model!r} could not start: {exc}.{detail}"
            ) from exc
    elapsed = time.perf_counter() - t0
    n_decoded = int((pivot.get("meta") or {}).get("n_frames") or len(pivot.get("frames") or []))

    kp = pivot_to_keypoints(pivot, clip_id or path.stem, backend=model, keep_estimated_feet=keep_estimated_feet)
    if timestamps is not None:
        ts = np.asarray(timestamps, dtype=float)
        if kp.n_frames and (kp.frame_idx.min() < 0 or kp.frame_idx.max() >= len(ts)):
            raise ValueError(
                f"timestamps has {len(ts)} entries but extraction reached frame {int(kp.frame_idx.max())}"
            )
        kp = Keypoints(
            clip_id=kp.clip_id, names=kp.names, xy=kp.xy, conf=kp.conf,
            time_s=ts[kp.frame_idx], frame_idx=kp.frame_idx, meta={**kp.meta, "time_source": "timestamps"},
        )
    else:
        kp.meta["time_source"] = "nominal_fps"
    package = _model_package(model)
    kp.meta.update(
        {
            "source": str(path),
            "myogait_version": str(getattr(mg, "__version__", "unknown")),
            "model_package": package,
            "model_package_version": _package_version(package) if package else None,
            "n_frames_decoded": n_decoded,
            "pose_runtime_s": round(elapsed, 3),
            "pose_s_per_frame": round(elapsed / n_decoded, 5) if n_decoded else None,
            "myogait_log": captured["log"][-50:],
        }
    )
    logger.info(
        "extract_keypoints: %s with %s, %d frames in %.1f s (%.1f ms/frame)",
        path.name, model, n_decoded, elapsed, 1000 * elapsed / max(n_decoded, 1),
    )
    return kp


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _require_myogait() -> ModuleType:
    try:
        import myogait
    except ImportError as exc:
        raise MissingDependencyError(
            "myogait is not installed; install the optional extra with "
            "`pip install 'gaitbench[myogait]'` (myogait==0.9.1, gaitkit==1.4.9)"
        ) from exc
    return myogait


def _installed_myogait_version() -> str:
    try:
        return myogait_version()
    except MissingDependencyError:
        return "unknown"


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _model_package(model: str) -> str | None:
    for prefix, package in _MODEL_PACKAGES.items():
        if model.startswith(prefix):
            return package
    return None


def _mediapipe_native_problem() -> str | None:
    """Why MediaPipe's native Tasks library cannot be loaded, or None if it loads (or cannot be probed)."""
    try:
        import mediapipe  # noqa: F401
    except ImportError as exc:
        return f"mediapipe is not importable ({exc}); install with `pip install 'myogait[mediapipe]'`"
    try:
        from mediapipe.tasks.python.core import mediapipe_c_bindings
    except ImportError:
        return None
    loader = getattr(mediapipe_c_bindings, "load_raw_library", None)
    if loader is None:
        return None
    try:
        loader()
    except OSError as exc:
        return (
            f"MediaPipe's native library failed to load ({exc}); on Linux install the EGL/GLES "
            "runtime, e.g. `apt install libegl1 libgles2`"
        )
    return None


def _frame_size(meta: Mapping[str, Any], *, what: str) -> tuple[float, float]:
    try:
        width, height = float(meta["width"]), float(meta["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FormatError(f"{what} needs numeric 'width' and 'height' to convert coordinates") from exc
    if not (width > 0 and height > 0):
        raise FormatError(f"{what} has non-positive frame size {width}x{height}")
    return width, height


def _swap_side(name: str) -> str:
    if name.startswith("LEFT_"):
        return "RIGHT_" + name[5:]
    if name.startswith("RIGHT_"):
        return "LEFT_" + name[6:]
    return name


def _entry_values(entry: Mapping[str, Any]) -> tuple[float, float, float]:
    def num(key: str, default: float) -> float:
        value = entry.get(key, default)
        try:
            return float(value) if value is not None else math.nan
        except (TypeError, ValueError):
            return math.nan

    return num("x", math.nan), num("y", math.nan), num("visibility", 1.0)


def _json_safe(obj: Any) -> Any:
    """Deep copy of ``obj`` with numpy scalars/arrays turned into Python types."""
    if obj is None:
        return None
    return json.loads(json.dumps(obj, default=_json_default))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _check_method(method: str) -> None:
    from myogait.events import list_event_methods

    available = [str(m) for m in list_event_methods()]
    if method == CONSENSUS or method in available:
        return
    if method.startswith("gk_") and _package_version("gaitkit") is None:
        raise MissingDependencyError(
            f"myogait method {method!r} needs gaitkit, which is not installed (`pip install gaitkit==1.4.9`)"
        )
    raise ValueError(f"Unknown myogait event method {method!r}. Available: {available + [CONSENSUS]}")


def _run(
    kp: Keypoints,
    method: str,
    *,
    butterworth_cutoff: float,
    min_cycle_duration_s: float,
    max_cycle_duration_s: float,
    direction_filter: bool,
    consensus_methods: Sequence[str] | None,
    consensus_tolerance_frames: int,
) -> tuple[dict, dict]:
    _require_myogait()
    _check_method(method)
    members = list(consensus_methods) if consensus_methods is not None else list(DEFAULT_CONSENSUS_METHODS)
    if method == CONSENSUS:
        if not members:
            raise ValueError("consensus_methods must name at least one method")
        for m in members:
            if m == CONSENSUS:
                raise ValueError("'consensus' cannot be a member of itself")
            _check_method(m)

    pivot = keypoints_to_pivot(kp)
    with _captured_output() as captured, _gaitkit_auto_install_off():
        try:
            if method == CONSENSUS:
                result = _run_consensus(
                    pivot,
                    members,
                    tolerance=int(consensus_tolerance_frames),
                    butterworth_cutoff=butterworth_cutoff,
                    min_cycle_duration_s=min_cycle_duration_s,
                    max_cycle_duration_s=max_cycle_duration_s,
                    direction_filter=direction_filter,
                )
            else:
                from myogait.pipeline import run_pipeline

                result = run_pipeline(
                    pivot,
                    event_method=method,
                    butterworth_cutoff=butterworth_cutoff,
                    min_cycle_duration_s=min_cycle_duration_s,
                    max_cycle_duration_s=max_cycle_duration_s,
                    analyze=False,
                    direction_filter=direction_filter,
                    show_progress=False,
                )
        except ImportError as exc:
            raise _dependency_error(method, members, exc) from exc
        except Exception as exc:
            raise MyogaitRunError(
                f"myogait method {method!r} failed on clip {kp.clip_id!r}: {type(exc).__name__}: {exc}"
            ) from exc

    quality = result.get("quality") or {}
    logger.info(
        "myogait:%s on %s: %d left / %d right cycles kept%s",
        method,
        kp.clip_id,
        quality.get("n_cycles_left", 0),
        quality.get("n_cycles_right", 0),
        "; " + "; ".join(quality["warnings"]) if quality.get("warnings") else "",
    )
    return result, captured


def _run_consensus(
    pivot: dict,
    methods: list[str],
    *,
    tolerance: int,
    butterworth_cutoff: float,
    min_cycle_duration_s: float,
    max_cycle_duration_s: float,
    direction_filter: bool,
) -> dict:
    """``run_pipeline`` with ``event_consensus`` in place of ``detect_events``.

    Mirrors ``myogait.pipeline._run_steps`` of myogait 0.9.1 for a non-C3D
    source, using only public functions plus the optional ``_diagnose``.
    """
    from myogait.angles import canonicalize_angle_signs, compute_angles
    from myogait.cycles import segment_cycles
    from myogait.events import event_consensus
    from myogait.normalize import normalize
    from myogait.pipeline import enforce_flexion_positive, filter_cycles_by_direction

    data = normalize(pivot, filters=["butterworth"], butterworth_cutoff=butterworth_cutoff)
    data = compute_angles(data, calibrate=False)
    data = canonicalize_angle_signs(data)

    # event_consensus returns array positions (it skips the frame_idx remap that
    # detect_events does). Run it where position == frame_idx, then map back.
    view = _positional_view(data)
    event_consensus(
        view,
        methods=list(methods),
        tolerance=tolerance,
        min_cycle_duration=min(0.6, min_cycle_duration_s),  # what run_pipeline passes to detect_events
    )
    events = dict(view["events"])
    source_idx = [int(f.get("frame_idx", i)) for i, f in enumerate(data["frames"])]
    fps = float(events.get("fps") or data["meta"].get("fps") or 30.0)
    for key in _EVENT_KEYS:
        mapped = []
        for ev in events.get(key, []) or []:
            pos = int(ev["frame"])
            if not 0 <= pos < len(source_idx):
                logger.warning("consensus event at position %d is outside the %d frames; dropped", pos, len(source_idx))
                continue
            mapped.append({**ev, "frame": source_idx[pos], "time": round(source_idx[pos] / fps, 4)})
        events[key] = mapped
    data["events"] = events

    cycles_all = segment_cycles(
        data, n_points=101, min_duration=min_cycle_duration_s, max_duration=max_cycle_duration_s
    )
    if direction_filter:
        cycles = filter_cycles_by_direction(data, cycles_all)
    else:
        cycles = cycles_all
        enforce_flexion_positive(cycles.get("cycles", []))
    try:
        from myogait.pipeline import _diagnose
    except ImportError:
        quality: dict | None = None
    else:
        quality = _diagnose(data, cycles_all, cycles)
    return {"data": data, "cycles": cycles, "stats": None, "quality": quality, "source_type": "json"}


def _positional_view(data: dict) -> dict:
    """Shallow copy of ``data`` whose frames (and angle frames) are numbered 0..N-1."""
    view = dict(data)
    view["frames"] = [{**f, "frame_idx": i} for i, f in enumerate(data["frames"])]
    angles = data.get("angles")
    if isinstance(angles, dict) and len(angles.get("frames") or []) == len(data["frames"]):
        view["angles"] = {**angles, "frames": [{**af, "frame_idx": i} for i, af in enumerate(angles["frames"])]}
    return view


def _events_table(events: Mapping[str, Any], kp: Keypoints, method: str) -> pd.DataFrame:
    known = set(kp.frame_idx.tolist())
    records = []
    for key, (side, event) in _EVENT_KEYS.items():
        for ev in events.get(key, []) or []:
            frame = int(ev["frame"])
            if frame not in known:
                logger.warning(
                    "myogait:%s returned frame %d, which is not a frame of %s", method, frame, kp.clip_id
                )
            conf = ev.get("confidence")
            records.append(
                {
                    "side": side,
                    "event": event,
                    "frame": frame,
                    "confidence": float(conf) if conf is not None else math.nan,
                }
            )
    defaults = {
        "clip_id": kp.clip_id,
        "tool": "myogait",
        "tool_version": _installed_myogait_version(),
        "backend": str(kp.meta.get("backend", "")),
        "detector": f"myogait:{method}",
    }
    if records:
        times = frames_to_times([r["frame"] for r in records], kp)
        for r, t in zip(records, times):
            r["time_s"] = float(t)
    return make_events(records, defaults=defaults)


def _dependency_error(method: str, members: list[str], exc: ImportError) -> MissingDependencyError:
    involved = [method] + (members if method == CONSENSUS else [])
    hints = [f"{m} needs {_METHOD_DEPENDENCIES[m]}" for m in involved if m in _METHOD_DEPENDENCIES]
    missing = getattr(exc, "name", None)
    what = f"the package {missing!r}" if missing else "an optional package"
    hint = f" ({'; '.join(hints)})" if hints else ""
    return MissingDependencyError(
        f"myogait method {method!r} needs {what}, which is not installed{hint}. Original error: {exc}"
    )


class _ListHandler(logging.Handler):
    def __init__(self, records: list[logging.LogRecord]) -> None:
        super().__init__(level=logging.DEBUG)
        self.records = records

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextlib.contextmanager
def _captured_output() -> Iterator[dict]:
    """Capture myogait/gaitkit log records and Python warnings during a call.

    Yields a dict that is filled on exit with ``log`` (messages) and
    ``python_warnings``. Records are re-emitted on this module's logger one
    level lower; warnings are logged at DEBUG.
    """
    records: list[logging.LogRecord] = []
    handler = _ListHandler(records)
    saved = []
    for name in _CAPTURED_LOGGERS:
        lg = logging.getLogger(name)
        saved.append((lg, lg.propagate, lg.level))
        lg.addHandler(handler)
        lg.propagate = False
        lg.setLevel(logging.INFO)
    out: dict[str, list[str]] = {"log": [], "python_warnings": []}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield out
    finally:
        for lg, propagate, level in saved:
            lg.removeHandler(handler)
            lg.propagate = propagate
            lg.setLevel(level)
        for rec in records:
            msg = f"[{rec.name}] {rec.getMessage()}"
            out["log"].append(f"{rec.levelname}: {msg}")
            level = logging.WARNING if rec.levelno >= logging.ERROR else (
                logging.INFO if rec.levelno >= logging.WARNING else logging.DEBUG
            )
            logger.log(level, "%s", msg)
        seen: set[str] = set()
        for w in caught:
            text = f"{w.category.__name__}: {w.message} ({Path(w.filename).name}:{w.lineno})"
            if text not in seen:
                seen.add(text)
                out["python_warnings"].append(text)
                logger.debug("python warning during myogait call: %s", text)


@contextlib.contextmanager
def _gaitkit_auto_install_off() -> Iterator[None]:
    """Stop gaitkit from pip-installing onnxruntime/tensorflow, unless the caller chose otherwise."""
    key = "GAITKIT_AUTO_INSTALL_DEPS"
    if key in os.environ:
        yield
        return
    os.environ[key] = "0"
    try:
        yield
    finally:
        os.environ.pop(key, None)


__all__ = [
    "CONSENSUS",
    "DEFAULT_CONSENSUS_METHODS",
    "MissingDependencyError",
    "MyogaitRunError",
    "detect_events_myogait",
    "extract_keypoints",
    "is_available",
    "keypoints_to_pivot",
    "list_methods",
    "myogait_version",
    "pivot_to_keypoints",
    "run_myogait",
]

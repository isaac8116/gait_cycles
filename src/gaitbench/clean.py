"""Clean pose keypoints before event detection: leg-swap correction.

The problem
-----------
In a side view the two legs overlap every time they pass each other. Pose
models (OpenPose, MediaPipe) then sometimes give the left leg's points the
``RIGHT_`` labels and the right leg's points the ``LEFT_`` labels, for one
frame or a few. Sometimes they put *both* legs' labels on the same physical
leg instead, which loses the other leg for that frame. A coordinate-based
detector reads either error as a foot jumping to the other foot's position
and back, and finds false events there. On the GPJATK sample
(``gpjatk_p01s01_sag_left__openpose``) this happens near t = 1.4, 2.5, 2.7
and 3.1 s.

What :func:`fix_leg_swaps` does, in plain words
-----------------------------------------------
1. **Leg chains.** For every frame, each label (``LEFT_``/``RIGHT_``) has a
   chain of points: knee, ankle, heel and toes (``cost_parts``). Hips are
   not used to decide, because in a side view the two hips sit on top of
   each other. Points below ``min_conf`` count as missing.
2. **Motion cost.** For three frames in a row, each leg's points are
   predicted at constant velocity from the two earlier frames, using the
   real frame times (iPhone video can have a variable frame rate). Image
   coordinates are used as they are: a steady pan is a constant velocity
   and drops out, and a trolley's speed changes add well under the noise
   per frame. A point costs ``(miss / allowed) ** 2``, capped at 1. The
   allowed miss is ``outlier_px`` (4 times the clip's typical miss), larger
   for low-confidence points (divided by ``sqrt(confidence)``) and across
   skipped frames. A missing point costs 1, like an outlier, so every point
   has to be explained either way. The cost of a frame triple is the sum
   over both legs divided by the number of chain parts. A whole chain on
   the wrong leg therefore costs 1 per leg per frame triple, and an
   isolated swapped frame about 6.
3. **Two states, one best path.** Each frame is either *kept* (labels as the
   pose model gave them) or *swapped*. A Viterbi pass finds the sequence of
   states with the lowest total motion cost plus ``switch_penalty`` for
   every change of state. Every point of the frame triple counts, so the
   decision uses the whole leg chain. When the ankles coincide as the legs
   cross, the knees and feet still separate the two labellings. The
   switching penalty stops the labels flip-flopping on noise. A genuine
   crossing never causes a switch, because switching there costs the
   penalty and gains nothing. Frames where the two chains lie on top of
   each other (``merge_px``) are left out of this pass and bridged, since
   they cannot tell the labellings apart. Each is then decided on its own
   between its labelled neighbours.
4. **Which labelling is "right".** Swapping every frame gives exactly the
   same motion cost. A cost per second of swapped frames
   (``swap_cost_per_s``) settles it in favour of the labelling that agrees
   with the pose model on most frames (majority). That is also what fixes
   a swap at the very start of a clip. It also keeps the start or end of a
   clip from being flipped wholesale where the evidence is weak.
5. **Both labels on one leg** (``fix_merged``). After the first pass, the
   function looks for short runs of frames where the two chains lie on top
   of each other. For each run, every leg's path is interpolated across the
   run from the frames on either side. A leg that lands far from its path
   (more than ``dup_px``) while the other leg stays on its own is a copy of
   the other leg, and its chain points are removed (NaN, confidence 0). The
   detectors' gap filling then bridges the frame. Standing with the feet
   together, and genuine crossings, stay on their paths and are left alone.
   The swap pass is then run again without the removed points.

The output has the ``LEFT_``/``RIGHT_`` versions of every part in ``parts``
(hips included) exchanged, xy and confidence together, on the swapped
frames. Removed duplicates are missing. Nothing else changes.

Limits (see the tests for the cases that are checked)
-----------------------------------------------------
* Evidence shrinks with noise. On the synthetic walker at 25-240 fps with
  up to 4 px noise on a 217 px leg, every injected swap is fixed. With
  5 px noise and the far leg at confidence 0.3 throughout, about 1 in 5
  is left in place. Those are swaps of only the first frame (one frame
  triple of evidence) or frames where the legs are within about a
  quarter of a leg length of each other. A clean clip is still left as it
  is. But two swaps a few frames apart on either side of a crossing may be
  merged into one run, which relabels the crossing frames between them,
  where the legs are close anyway.
* A swap that starts and ends exactly where the legs cross shows only as a
  kink in each leg's velocity. That kink is small at high frame rates and
  is lost in heavy noise.
* If the pose model is wrong on *more* than half of the frames, the
  majority rule picks the wrong labelling for the whole clip.
* Swaps of part of the chain (feet swapped but knees not) are fixed as all
  or nothing, whichever the majority of the chain points supports.
* A run of duplicated frames at the very start or end of the clip has no
  frames on one side to interpolate from, so it is left alone. A copy that
  lands on the other leg's path for longer than ``max_merge_s`` is not
  removed.
* Long gaps (no leg points for more than ``max_span_s``) cut the evidence.
  The labelling then carries on unchanged across the gap.

File naming
-----------
:func:`clean_keypoints_file` writes a new keypoints file. Its backend is
the input backend with the suffix ``-clean`` (:data:`CLEAN_SUFFIX`), passed
through :func:`gaitbench.formats.safe_name` and never doubled:
``openpose`` -> ``openpose-clean`` and ``mediapipe-60fps`` ->
``mediapipe-60fps-clean``. By default the file goes next to the input as
``<clip>__<backend>-clean.npz``, so the pipeline (``keypoint_files``) treats
it as one more backend, and its events and scores sit beside the uncleaned
ones. ``meta['cleaned']`` records what was done: tool, version, steps,
source backend and path, and per step the parameters used and the frames
changed.

Example
-------
>>> from gaitbench.clean import fix_leg_swaps, clean_keypoints_file
>>> fixed, report = fix_leg_swaps(kp)                                   # doctest: +SKIP
>>> report[report.action == "swap"]                                     # doctest: +SKIP
>>> clean_keypoints_file("data/x/keypoints/clip__openpose.npz")         # doctest: +SKIP
"""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import __version__
from .formats import Keypoints, safe_name

LEG_PARTS: tuple[str, ...] = ("HIP", "KNEE", "ANKLE", "HEEL", "FOOT_INDEX", "BIG_TOE", "SMALL_TOE")
# Parts that are swapped but never used to decide: the hips overlap in a side view.
_NOT_FOR_COST: frozenset[str] = frozenset({"HIP"})
CLEAN_SUFFIX = "-clean"
REPORT_COLUMNS: tuple[str, ...] = (
    "frame", "time_s", "cost_keep", "cost_swap", "action", "dropped", "separation_px",
)

# Typical peak foot acceleration in walking, in leg lengths per s^2 (about 3 g
# for a 0.9 m leg). Only used to widen the allowed miss across missing frames.
_FOOT_ACCEL_LEGS_PER_S2 = 30.0
# Floor of the noise estimate, as a fraction of the leg length. Noise-free
# (synthetic) data would otherwise make every small model error an outlier.
_NOISE_FLOOR_LEG_FRACTION = 0.02
# outlier_px = this many times the noise of a constant-velocity prediction.
_OUTLIER_NOISE_MULTIPLE = 4.0
# Cost of a missing point in a frame triple: the same as an outlier. Every
# point must be explained, so a labelling cannot get cheaper by splitting a
# leg's track where it moves awkwardly (e.g. a foot landing as it enters the
# image), and labellings with more or fewer complete triples compare fairly.
_MISSING_POINT_COST = 1.0
# Frame steps within this fraction of the median count as "no frame missing"
# when the noise is estimated.
_REGULAR_STEP_TOLERANCE = 0.25


# ---------------------------------------------------------------------------
# Leg-swap correction
# ---------------------------------------------------------------------------


def fix_leg_swaps(
    kp: Keypoints,
    *,
    parts: Sequence[str] = LEG_PARTS,
    cost_parts: Sequence[str] | None = None,
    min_conf: float = 0.1,
    switch_penalty: float = 0.5,
    swap_cost_per_s: float = 1.0,
    outlier_px: float | None = None,
    max_span_s: float = 0.25,
    fix_merged: bool = True,
    merge_px: float | None = None,
    dup_px: float | None = None,
    max_merge_s: float = 0.15,
) -> tuple[Keypoints, pd.DataFrame]:
    """Find frames where the pose model swapped the leg labels and swap them back.

    See the module docstring for the method in plain words.

    Parameters
    ----------
    kp:
        Side-view keypoints of one clip. Not modified.
    parts:
        Leg parts whose ``LEFT_``/``RIGHT_`` versions are exchanged on a
        swapped frame. Parts missing from the file are ignored.
    cost_parts:
        Parts used to decide. Default: ``parts`` without ``HIP``.
    min_conf:
        Points below this confidence count as missing when deciding. They
        are still swapped with the rest of their leg. A point's allowed
        miss grows as ``1 / sqrt(confidence)``.
    switch_penalty:
        Cost of changing state between two frames. The unit is one leg
        chain wholly inconsistent in one frame triple, so a well-separated
        one-frame swap is worth about 6 and a swap of the first or last
        frame about 2. A one-frame swap is corrected when it is worth more
        than twice this.
    swap_cost_per_s:
        Cost per second of output frames whose labels are exchanged: the
        prior that the pose model usually labels the legs right. It picks
        the labelling that agrees with the model on more frames when two are
        equally smooth (a swap at the very start of the clip). It also stops
        the start or end of a clip from being flipped wholesale on weak
        evidence.
    outlier_px:
        Prediction miss at which a point counts as fully inconsistent.
        Default: ``4`` times the median miss of a constant-velocity
        prediction over the clip, with a floor of 2 % of the leg length.
    max_span_s:
        Frame triples spanning more than this (because of missing frames)
        are not used: constant-velocity prediction means nothing over long
        gaps.
    fix_merged:
        Also remove a leg's chain where the pose model put both legs'
        labels on the other leg (see the module docstring).
    merge_px:
        The chains count as lying on top of each other when the median
        distance between matching points is at most this. Such frames are
        decided one by one after the main pass, and are checked for copies.
        Default: the larger of the noise and 3 % of the leg length.
    dup_px:
        A leg counts as a copy of the other leg when its points are at
        least this far from the leg's path. Default: the larger of
        ``2 * outlier_px`` and 20 % of the leg length.
    max_merge_s:
        Longest run of overlapping frames checked for copies.

    Returns
    -------
    (Keypoints, pandas.DataFrame)
        The corrected keypoints (a new object, with
        ``meta['cleaned']['leg_swaps']`` summarising what was changed), and
        a report with one row per frame:

        * ``frame``, ``time_s``: the frame;
        * ``cost_keep``, ``cost_swap``: motion cost of the frame triples
          that contain this frame, with its labels as the pose model gave
          them or swapped (every other frame as finally labelled). NaN for
          frames with no usable leg point. A large ``cost_keep`` with
          ``action == "keep"`` marks a frame worth a look;
        * ``action``: ``"swap"`` if the output has this frame's labels
          exchanged, else ``"keep"``;
        * ``dropped``: ``"L"`` or ``"R"`` if that leg's chain was removed
          as a copy of the other leg, else ``""`` (sides are the output's);
        * ``separation_px``: median distance between the two legs' chain
          points after correction (NaN if no point has both sides).
    """
    if min(switch_penalty, swap_cost_per_s, min_conf, max_merge_s) < 0 or max_span_s <= 0:
        raise ValueError("switch_penalty, swap_cost_per_s, min_conf and max_merge_s must be >= 0, "
                         "and max_span_s > 0")
    swap_parts = [p for p in dict.fromkeys(str(p).upper() for p in parts)
                  if kp.has("LEFT_" + p) and kp.has("RIGHT_" + p)]
    wanted_cost = [str(p).upper() for p in cost_parts] if cost_parts is not None else \
        [p for p in swap_parts if p not in _NOT_FOR_COST]
    decide = [p for p in dict.fromkeys(wanted_cost) if kp.has("LEFT_" + p) and kp.has("RIGHT_" + p)]
    n = kp.n_frames
    t = kp.time_s
    if not decide or n < 3:
        if not decide:
            looked_for = wanted_cost if cost_parts is not None else \
                [str(p).upper() for p in parts if str(p).upper() not in _NOT_FOR_COST]
            warnings.warn(f"{kp.clip_id}: no LEFT_/RIGHT_ pair of {looked_for} in the file; "
                          "leg swaps not checked", stacklevel=2)
        decide = []

    P, W = _leg_arrays(kp, decide, min_conf)  # (n, 2, m, 2), (n, 2, m)
    leg_px = _leg_length_px(kp, min_conf)
    dt_med = float(np.median(np.diff(t))) if n > 1 else 1.0
    noise_px = _prediction_noise(P, W, t, dt_med)
    noise_eff = max(noise_px if np.isfinite(noise_px) else 0.0, _NOISE_FLOOR_LEG_FRACTION * leg_px)
    r0 = float(outlier_px) if outlier_px is not None else _OUTLIER_NOISE_MULTIPLE * noise_eff
    merge_eff = float(merge_px) if merge_px is not None else max(noise_eff, 0.03 * leg_px)
    dup_eff = float(dup_px) if dup_px is not None else max(2.0 * r0, 0.2 * leg_px)
    accel_px = _FOOT_ACCEL_LEGS_PER_S2 * leg_px
    # A tiny floor keeps a mirror-image labelling from tying when swap_cost_per_s is 0.
    prior_per_frame = max(swap_cost_per_s * dt_med, 1e-9)

    def label_frames(P_: np.ndarray, W_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Swap state and ``(cost_keep, cost_swap)`` of every frame.

        Frames where the two chains lie on top of each other say little about
        the labelling, and one where both labels sit on one leg would let the
        labelling flip for free. So the Viterbi pass leaves them out (the
        frame triples bridge them with constant-velocity prediction over the
        real time gap), and each is then decided on its own between its
        already-labelled neighbours, where it cannot start a lasting flip.
        """
        state, costs = np.zeros(n, dtype=bool), np.full((n, 2), np.nan)
        has = (W_ > 0).any(axis=(1, 2))
        overlap = np.nan_to_num(_separation(P_, W_), nan=np.inf) <= merge_eff
        idx = np.flatnonzero(has & ~overlap)
        if len(idx) < 3:
            return state, costs
        C = _triplet_costs(P_[idx], W_[idx], t[idx], r0, accel_px, max_span_s, dt_med)
        s = _viterbi(C, switch_penalty, prior_per_frame)
        state = _spread_states(s, idx, n)
        costs[idx] = _local_costs(C, s)
        for f in np.flatnonzero(has & overlap):
            pos = int(np.searchsorted(idx, f))
            frames = np.concatenate([idx[max(0, pos - 2):pos], [f], idx[pos:pos + 2]])
            k = min(pos, 2)  # position of f in `frames`
            if len(frames) < 3:
                continue
            Csub = _triplet_costs(P_[frames], W_[frames], t[frames], r0, accel_px, max_span_s, dt_med)
            local = _local_costs(Csub, state[frames].astype(int))[k]
            neighbours = [int(state[frames[k - 1]])] if k > 0 else []
            neighbours += [int(state[frames[k + 1]])] if k + 1 < len(frames) else []
            total = [local[v] + switch_penalty * sum(v != s_ for s_ in neighbours) + v * prior_per_frame
                     for v in (0, 1)]
            if total[1 - int(state[f])] < total[int(state[f])]:
                state[f] = not state[f]
            costs[f] = local
        return state, costs

    state, costs = label_frames(P, W)
    drop_label = np.full(n, -1)  # original label (0 = LEFT_, 1 = RIGHT_) removed as a copy, or -1
    if fix_merged and len(decide) and n >= 3:
        Q, V = _apply_state(P, state), _apply_state(W, state)
        for frame, track in _find_copies(Q, V, t, merge_eff, dup_eff, max_merge_s, max_span_s):
            drop_label[frame] = track ^ int(state[frame])
        if (drop_label >= 0).any():
            P2, W2 = P.copy(), W.copy()
            rows = np.flatnonzero(drop_label >= 0)
            W2[rows, drop_label[rows]] = 0.0
            P2[rows, drop_label[rows]] = np.nan
            state, costs = label_frames(P2, W2)

    # Build the output: swap every part in `parts`, then remove copies.
    xy, conf = kp.xy.copy(), kp.conf.copy()
    rows = np.flatnonzero(state)
    for part in swap_parts:
        li, ri = kp.index("LEFT_" + part), kp.index("RIGHT_" + part)
        xy[rows, li], xy[rows, ri] = kp.xy[rows, ri], kp.xy[rows, li]
        conf[rows, li], conf[rows, ri] = kp.conf[rows, ri], kp.conf[rows, li]
    dropped = np.full(n, "", dtype=object)
    for frame in np.flatnonzero(drop_label >= 0):
        side = "LR"[int(drop_label[frame]) ^ int(state[frame])]
        for part in decide:
            j = kp.index(("LEFT_" if side == "L" else "RIGHT_") + part)
            xy[frame, j] = np.nan
            conf[frame, j] = 0.0
        dropped[frame] = side

    separation = _separation(_apply_state(P, state), _apply_state(W, state))
    report = pd.DataFrame({
        "frame": kp.frame_idx.astype(np.int64),
        "time_s": t,
        # State 0 is the model's own labelling, so column 0 is "keep".
        "cost_keep": costs[:, 0],
        "cost_swap": costs[:, 1],
        "action": np.where(state, "swap", "keep"),
        "dropped": dropped.astype(str),
        "separation_px": separation,
    }, columns=list(REPORT_COLUMNS))

    summary = {
        "n_frames": int(n),
        "n_swapped": int(state.sum()),
        "swapped_frames": _frame_runs(kp.frame_idx, state),
        "n_dropped": int((drop_label >= 0).sum()),
        "dropped_frames": {side: _frame_runs(kp.frame_idx, dropped == side) for side in ("L", "R")},
        "params": {
            "parts": list(swap_parts), "cost_parts": list(decide), "min_conf": float(min_conf),
            "switch_penalty": float(switch_penalty), "swap_cost_per_s": float(swap_cost_per_s),
            "outlier_px": _round(r0), "max_span_s": float(max_span_s),
            "fix_merged": bool(fix_merged), "merge_px": _round(merge_eff), "dup_px": _round(dup_eff),
            "max_merge_s": float(max_merge_s), "noise_px": _round(noise_px), "leg_px": _round(leg_px),
        },
    }
    meta = dict(kp.meta)
    meta["cleaned"] = {**dict(meta.get("cleaned") or {}), "leg_swaps": summary}
    out = Keypoints(clip_id=kp.clip_id, names=kp.names, xy=xy, conf=conf, time_s=kp.time_s.copy(),
                    frame_idx=kp.frame_idx.copy(), meta=meta)
    return out, report


# ---------------------------------------------------------------------------
# Cleaning files
# ---------------------------------------------------------------------------

STEPS: dict[str, Callable[..., tuple[Keypoints, pd.DataFrame]]] = {
    "leg_swaps": fix_leg_swaps,
}


def cleaned_backend_name(backend: str) -> str:
    """Backend name of a cleaned copy: ``openpose`` -> ``openpose-clean`` (never doubled)."""
    name = safe_name(backend or "unknown")
    return name if name.endswith(CLEAN_SUFFIX) else name + CLEAN_SUFFIX


def clean_keypoints(
    kp: Keypoints,
    steps: Sequence[str] = ("leg_swaps",),
    *,
    step_params: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Keypoints, pd.DataFrame]:
    """Run cleaning steps in order on keypoints in memory.

    Parameters
    ----------
    kp:
        Keypoints of one clip. Not modified.
    steps:
        Names from :data:`STEPS`, run in the order given.
    step_params:
        Keyword arguments per step, e.g. ``{"leg_swaps": {"min_conf": 0.2}}``.
        Checked before anything runs, so a typo fails at once.

    Returns
    -------
    (Keypoints, pandas.DataFrame)
        The cleaned keypoints, with ``meta['backend']`` renamed by
        :func:`cleaned_backend_name` and ``meta['cleaned']`` filled in, and
        the steps' reports stacked, with ``clip_id`` and ``step`` columns in
        front.
    """
    steps = [str(s) for s in ([steps] if isinstance(steps, str) else steps)]
    params = {str(k): dict(v or {}) for k, v in (step_params or {}).items()}
    unknown = sorted(set(steps) - set(STEPS)) + sorted(set(params) - set(STEPS))
    if unknown:
        raise ValueError(f"Unknown cleaning step(s) {sorted(set(unknown))}; valid steps: {list(STEPS)}")
    if not steps:
        raise ValueError("No cleaning steps given")
    for step in steps:
        accepted = [p for p in inspect.signature(STEPS[step]).parameters if p != "kp"]
        bad = sorted(set(params.get(step, {})) - set(accepted))
        if bad:
            raise TypeError(f"Cleaning step {step!r} does not take {bad}; it takes {accepted}")

    previous = dict(kp.meta.get("cleaned") or {})
    source_backend = str(previous.get("source_backend") or kp.meta.get("backend", "") or "unknown")
    out, reports = kp, []
    for step in steps:
        out, report = STEPS[step](out, **params.get(step, {}))
        report = report.copy()
        report.insert(0, "step", step)
        report.insert(0, "clip_id", kp.clip_id)
        reports.append(report)

    meta = dict(out.meta)
    meta["cleaned"] = {
        **dict(meta.get("cleaned") or {}),
        "tool": "gaitbench.clean",
        "tool_version": __version__,
        "steps": list(previous.get("steps") or []) + steps,
        "source_backend": source_backend,
    }
    meta["backend"] = cleaned_backend_name(str(kp.meta.get("backend", "")) or "unknown")
    cleaned = Keypoints(clip_id=out.clip_id, names=out.names, xy=out.xy, conf=out.conf, time_s=out.time_s,
                        frame_idx=out.frame_idx, meta=meta)
    return cleaned, pd.concat(reports, ignore_index=True)


def clean_keypoints_file(
    in_path: str | Path,
    out_path: str | Path | None = None,
    steps: Sequence[str] = ("leg_swaps",),
    *,
    step_params: Mapping[str, Mapping[str, Any]] | None = None,
    report_path: str | Path | None = None,
) -> pd.DataFrame:
    """Clean one keypoints file and write the result as a new backend.

    Parameters
    ----------
    in_path:
        A keypoints ``.npz`` (:meth:`gaitbench.formats.Keypoints.save`).
    out_path:
        Where to write the cleaned file. Default: next to the input, as
        ``<clip>__<backend>-clean.npz``. ``<clip>`` comes from the input
        file name and ``<backend>`` from ``meta['backend']`` (see the module
        docstring). Writing over the input is refused.
    steps, step_params:
        See :func:`clean_keypoints`.
    report_path:
        If given, the report is also written there as CSV.

    Returns
    -------
    pandas.DataFrame
        The report of :func:`clean_keypoints`.
    """
    in_path = Path(in_path)
    kp = Keypoints.load(in_path)
    cleaned, report = clean_keypoints(kp, steps, step_params=step_params)
    cleaned.meta["cleaned"]["source_path"] = str(in_path)
    if out_path is None:
        # Keep the clip part of the input name, as the pipeline finds files by name.
        clip = in_path.stem.rsplit("__", 1)[0] if "__" in in_path.stem else safe_name(kp.clip_id)
        out_path = in_path.with_name(f"{clip}__{cleaned.meta['backend']}.npz")
    out_path = Path(out_path)
    if out_path.resolve() == in_path.resolve():
        raise ValueError(f"Refusing to overwrite the input {in_path}; give a different out_path")
    cleaned.save(out_path)
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_path, index=False)
    return report


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _leg_arrays(kp: Keypoints, parts: Sequence[str], min_conf: float) -> tuple[np.ndarray, np.ndarray]:
    """``P[frame, label, part, xy]`` and weights ``W[frame, label, part]`` (0 = missing)."""
    n, m = kp.n_frames, len(parts)
    P = np.full((n, 2, m, 2), np.nan)
    W = np.zeros((n, 2, m))
    for label, prefix in enumerate(("LEFT_", "RIGHT_")):
        for j, part in enumerate(parts):
            xy, c = kp.point(prefix + part), kp.confidence(prefix + part)
            ok = np.isfinite(xy).all(axis=1) & (c >= min_conf) & (c > 0)
            P[ok, label, j] = xy[ok]
            W[ok, label, j] = c[ok]
    return P, W


def _leg_length_px(kp: Keypoints, min_conf: float) -> float:
    """Median hip-ankle distance (or twice a thigh or shank) over the clip, in pixels."""
    for a, b, factor in (("HIP", "ANKLE", 1.0), ("KNEE", "ANKLE", 2.0), ("HIP", "KNEE", 2.0)):
        values = []
        for prefix in ("LEFT_", "RIGHT_"):
            if kp.has(prefix + a) and kp.has(prefix + b):
                d = np.linalg.norm(kp.masked_point(prefix + a, min_conf) - kp.masked_point(prefix + b, min_conf), axis=1)
                values.append(d[np.isfinite(d)] * factor)
        values = np.concatenate(values) if values else np.array([])
        if values.size and np.median(values) > 0:
            return float(np.median(values))
    height = kp.meta.get("height")
    return float(0.25 * float(height)) if height else 100.0


def _triplet_miss(P: np.ndarray, W: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Constant-velocity prediction miss for frames (k-2, k-1, k) in every combination of states.

    State 1 means the frame's labels are exchanged, so leg ``l`` takes the
    points of label ``l ^ state``. Returns ``miss[a, b, c, k-2, leg, part]``
    in pixels (NaN where a point is missing) and the matching weights (the
    smallest confidence of the three points, 0 where one is missing).
    """
    K, _, m, _ = P.shape
    miss = np.full((2, 2, 2, max(K - 2, 0), 2, m), np.nan)
    weight = np.zeros_like(miss)
    if K < 3:
        return miss, weight
    order = (np.array([0, 1]), np.array([1, 0]))
    Ps, Ws = [P[:, o] for o in order], [W[:, o] for o in order]
    ratio = ((t[2:] - t[1:-1]) / (t[1:-1] - t[:-2]))[:, None, None, None]
    for a in (0, 1):
        for b in (0, 1):
            x0, x1, w01 = Ps[a][:-2], Ps[b][1:-1], np.minimum(Ws[a][:-2], Ws[b][1:-1])
            for c in (0, 1):
                weight[a, b, c] = np.minimum(w01, Ws[c][2:])
                with np.errstate(invalid="ignore"):
                    miss[a, b, c] = np.linalg.norm(Ps[c][2:] - x1 - (x1 - x0) * ratio, axis=-1)
    return miss, weight


def _prediction_noise(P: np.ndarray, W: np.ndarray, t: np.ndarray, dt_med: float) -> float:
    """Median miss of a constant-velocity prediction over regular frame triples.

    Each triple is scored in whichever labelling fits it best, so the swaps
    being corrected do not inflate the estimate.
    """
    if len(t) < 3 or P.shape[2] == 0:
        return float("nan")
    dt = np.diff(t)
    regular = np.abs(dt / dt_med - 1.0) <= _REGULAR_STEP_TOLERANCE
    miss, weight = _triplet_miss(P, W, t)
    miss = miss.reshape(8, len(t) - 2, -1)
    valid = weight.reshape(8, len(t) - 2, -1) > 0
    count = valid.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_miss = np.where(valid, miss, 0.0).sum(axis=2) / count  # (8, K-2); NaN where nothing valid
    usable = np.isfinite(mean_miss).any(axis=0) & regular[1:] & regular[:-1]
    if not usable.any():
        return float("nan")
    best = np.argmin(np.where(np.isfinite(mean_miss), mean_miss, np.inf), axis=0)
    cols = np.flatnonzero(usable)
    chosen = miss[best[cols], cols]
    return float(np.median(chosen[valid[best[cols], cols]]))


def _triplet_costs(P: np.ndarray, W: np.ndarray, t: np.ndarray, r0: float, accel_px: float,
                   max_span_s: float, dt_med: float) -> np.ndarray:
    """``C[k, a, b, c]``: motion cost of frames (k-2, k-1, k) in states (a, b, c); 0 for k < 2."""
    K, _, m, _ = P.shape
    C = np.zeros((K, 2, 2, 2))
    if K < 3 or m == 0:
        return C
    miss, weight = _triplet_miss(P, W, t)
    dt1, dt2 = t[1:-1] - t[:-2], t[2:] - t[1:-1]
    # Across skipped frames the allowed miss widens twice over: the velocity,
    # measured over the short step dt1, is extrapolated over the long step
    # dt2 (its noise grows with r = dt2 / dt1; the miss is x2 - (1 + r) x1 + r x0,
    # so its variance is 1 + (1 + r)^2 + r^2 times a point's, against 6 for
    # equal steps), and a typical acceleration adds a * dt2 * (dt1 + dt2) / 2.
    r = dt2 / dt1
    noise_gain = np.sqrt((1.0 + (1.0 + r) ** 2 + r**2) / 6.0)
    excess = np.maximum(0.0, dt2 * (dt1 + dt2) / 2.0 - dt_med**2)
    scale = (r0 * np.maximum(noise_gain, 1.0) + accel_px * excess)[:, None, None]
    usable = (t[2:] - t[:-2]) <= max_span_s + 1e-9
    with np.errstate(invalid="ignore", divide="ignore"):
        # A weak point is allowed a larger miss (tolerance / sqrt(confidence)),
        # but a point that misses by the allowed distance costs 1 whatever its
        # confidence. Scaling costs by confidence instead would make a
        # labelling that mixes the legs (so mixes near-leg and weaker far-leg
        # points in one track) look cheap. A missing point costs as much as an
        # outlier, so a labelling gains nothing by losing points.
        tolerance = scale / np.sqrt(np.where(weight > 0, weight, 1.0))
        rho = np.where(weight > 0, np.minimum((miss / tolerance) ** 2, 1.0), _MISSING_POINT_COST)
    cost = rho.sum(axis=(4, 5)) / m  # (2, 2, 2, K-2)
    C[2:] = np.moveaxis(np.where(usable, cost, 0.0), -1, 0)
    return C


def _viterbi(C: np.ndarray, switch_penalty: float, prior: float) -> np.ndarray:
    """Lowest-cost state sequence for a second-order chain (states are frame pairs)."""
    K = C.shape[0]
    s = np.zeros(K, dtype=int)
    if K < 3:
        return s
    switch = switch_penalty * np.array([[0.0, 1.0], [1.0, 0.0]])
    swapped = prior * np.array([0.0, 1.0])
    D = switch + swapped[:, None] + swapped[None, :]  # D[s0, s1]
    back = np.zeros((K, 2, 2), dtype=np.int8)
    for k in range(2, K):
        cand = D[:, :, None] + C[k]  # cand[a, b, c]
        back[k] = np.argmin(cand, axis=0)
        D = cand.min(axis=0) + switch + swapped[None, :]
    b, c = np.unravel_index(int(np.argmin(D)), D.shape)
    s[K - 2], s[K - 1] = b, c
    for k in range(K - 1, 1, -1):
        s[k - 2] = back[k, s[k - 1], s[k]]
    return s


def _local_costs(C: np.ndarray, s: np.ndarray) -> np.ndarray:
    """``out[k, v]``: cost of the triples containing frame k, with frame k in state v."""
    K = len(s)
    out = np.zeros((K, 2))
    for k in range(K):
        for v in (0, 1):
            total = 0.0
            for j in (k, k + 1, k + 2):
                if 2 <= j < K:
                    a, b, c = (v if i == k else s[i] for i in (j - 2, j - 1, j))
                    total += C[j, a, b, c]
            out[k, v] = total
    return out


def _spread_states(s: np.ndarray, idx: np.ndarray, n: int) -> np.ndarray:
    """States on all frames: frames without leg points are swapped only between swapped frames."""
    state = np.zeros(n, dtype=bool)
    state[idx] = s.astype(bool)
    others = np.setdiff1d(np.arange(n), idx)
    if len(others) and len(idx):
        pos = np.searchsorted(idx, others)
        before = state[idx[np.clip(pos - 1, 0, len(idx) - 1)]]
        after = state[idx[np.clip(pos, 0, len(idx) - 1)]]
        state[others] = before & after
    return state


def _apply_state(A: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Exchange the label axis (axis 1) of ``A`` on frames where ``state`` is set."""
    out = A.copy()
    out[state] = A[state][:, ::-1]
    return out


def _separation(Q: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Median distance between matching points of the two legs, per frame (NaN if none)."""
    both = (V[:, 0] > 0) & (V[:, 1] > 0)
    with np.errstate(invalid="ignore"):
        d = np.where(both, np.linalg.norm(Q[:, 0] - Q[:, 1], axis=-1), np.nan)
    out = np.full(len(Q), np.nan)
    rows = both.any(axis=1)
    out[rows] = np.nanmedian(d[rows], axis=1)
    return out


def _find_copies(Q: np.ndarray, V: np.ndarray, t: np.ndarray, merge_px: float, dup_px: float,
                 max_merge_s: float, max_span_s: float) -> list[tuple[int, int]]:
    """(frame, leg) pairs where the leg's chain is a copy of the other leg's.

    ``Q``/``V`` are already in output order (leg 0 = left). A run of frames
    where the chains overlap is checked against each leg's path, linearly
    interpolated between the nearest frames on either side that have points
    for that leg and do not overlap.
    """
    n = len(t)
    sep = _separation(Q, V)
    merged = np.nan_to_num(sep, nan=np.inf) <= merge_px
    has = V > 0  # (n, 2, m)
    out: list[tuple[int, int]] = []
    for start, stop in _runs(merged):
        if t[stop - 1] - t[start] > max_merge_s:
            continue
        anchors = []
        for leg in (0, 1):
            ok = has[:, leg].any(axis=1) & ~merged
            before = [i for i in range(start - 1, -1, -1) if t[start] - t[i] <= max_span_s and ok[i]][:1]
            after = [i for i in range(stop, n) if t[i] - t[stop - 1] <= max_span_s and ok[i]][:1]
            anchors.append((before[0], after[0]) if before and after else None)
        if None in anchors:
            continue
        for frame in range(start, stop):
            dev = np.full(2, np.nan)
            for leg, (a, b) in enumerate(anchors):
                use = has[a, leg] & has[b, leg] & has[frame, leg]
                if use.any():
                    f = (t[frame] - t[a]) / (t[b] - t[a])
                    path = Q[a, leg, use] + f * (Q[b, leg, use] - Q[a, leg, use])
                    dev[leg] = float(np.median(np.linalg.norm(Q[frame, leg, use] - path, axis=-1)))
            if np.isfinite(dev).all():
                far, near = (0, 1) if dev[0] >= dev[1] else (1, 0)
                if dev[far] > dup_px and dev[near] < 0.5 * dev[far]:
                    out.append((frame, far))
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``(start, stop)`` of each run of True values."""
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _frame_runs(frame_idx: np.ndarray, mask: np.ndarray) -> list[list[int]]:
    """Runs of selected frames as ``[first_frame, last_frame]`` source-frame pairs (JSON-friendly)."""
    return [[int(frame_idx[a]), int(frame_idx[b - 1])] for a, b in _runs(mask)]


def _round(value: float) -> float | None:
    return round(float(value), 3) if np.isfinite(value) else None

"""Building blocks shared by the coordinate-based event detectors.

Zeni (2008) and Stenum (2021) both look at how far a foot point is in front
of, or behind, the pelvis along the walking direction:

* the foot is furthest **in front** of the pelvis just as it lands
  (heel strike, HS), and
* furthest **behind** the pelvis just as it lifts off (toe off, TO).

Because both quantities are measured relative to the pelvis, they do not
change when the camera pans to follow the walker. That is the property the
rail-mounted rig relies on.

Each detector is the same pipeline with a different choice of foot point:

1. pick a foot keypoint per side, falling back to the ankle if the backend
   has no heel or toe (:func:`choose_foot_point`);
2. build ``direction * (foot_x - pelvis_x)`` with low-confidence points
   removed and short gaps filled (:func:`foot_signals`);
3. smooth each continuous stretch with a zero-phase Butterworth filter;
4. pick peaks with a minimum spacing and a minimum prominence, refine them to
   a fraction of a frame and convert array positions to source-video frames
   and times (:func:`find_extrema`, :func:`positions_to_frames`).

The walking direction comes from :func:`estimate_direction` unless it is given.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

from .. import __version__
from ..formats import SIDES, Keypoints, frames_to_times, make_events

# Fallback chains for each foot point, best first. The ankle is the last
# resort: every pose model has it, but in real walking it is furthest forward
# a little after heel strike and furthest back a little before toe off,
# because the foot rotates about the heel and then the toe.
FOOT_POINT_CHAINS: dict[str, tuple[str, ...]] = {
    "heel": ("HEEL", "ANKLE"),
    "toe": ("FOOT_INDEX", "BIG_TOE", "ANKLE"),
    "ankle": ("ANKLE",),
}

FILTER_ORDER = 4
# Each end of a segment is padded by this many periods of the cutoff before
# filtering (0.5 s at 6 Hz), about as long as the filter takes to settle.
FILTER_PAD_PERIODS = 3.0
# Upper limit on the cutoff as a fraction of the Nyquist frequency. At 25 fps
# Nyquist is 12.5 Hz, so a requested 15 Hz cutoff becomes 11.25 Hz.
MAX_CUTOFF_FRACTION_OF_NYQUIST = 0.9
# A point in a chain is used only if it has at least this fraction of the
# confident frames of the best point in that chain; otherwise the next is tried.
_POINT_USABLE_FRACTION = 0.5
# Next to a segment edge only part of a peak's fall on that side is visible,
# so that side needs to fall by just this fraction of the usual prominence
# (see _edge_aware_drop). The other side must still pass in full.
EDGE_PROMINENCE_FRACTION = 0.25
# Only peaks this close to the edge are relaxed. A real event further in has
# fallen far enough by the edge to pass in full (a planted heel moves back
# relative to the pelvis at walking speed), while relaxing it would let jitter
# in a standing-still stretch at the start or end of a clip pass as an event.
# 0.2 s rather than 0.1 s: with the noise margin below, 0.2 s still gave no
# false events in standing tails or at gap edges, and keeps real clip-edge
# events at 25 fps that 0.1 s lost (review finding DET-1).
EDGE_RELAX_MAX_S = 0.2
# A censored fall ends on the outermost sample of a segment, which zero-phase
# filtering leaves almost unsmoothed, so its fall is shortened by this many
# noise standard deviations before it is relaxed (see _edge_aware_drop).
EDGE_NOISE_MARGIN = 2.0
# Each hip is interpolated across its own gaps up to this long (see pelvis_x).
# Longer than a foot's max_gap_s: a foot gap can hide an event, while a hip
# moves almost linearly over half a second.
PELVIS_MAX_GAP_S = 0.5
# Fewest confident frames a direction cue needs before it is trusted.
_MIN_DIRECTION_FRAMES = 10
# Share of frames that must agree with the median sign for a cue to count.
_MIN_DIRECTION_AGREEMENT = 0.6

TOOL_NAME = "gaitbench"


@dataclass
class FootSignal:
    """One side's ``direction * (foot_x - pelvis_x)`` signal and how it was made.

    Attributes
    ----------
    side:
        ``"L"`` or ``"R"``, the walker's own side.
    point:
        Keypoint actually used, e.g. ``"LEFT_HEEL"`` or ``"LEFT_ANKLE"`` after a fallback.
    raw:
        ``(n_frames,)`` signal in pixels after masking and gap filling; NaN in long gaps.
    smooth:
        ``(n_frames,)`` low-pass filtered signal; NaN outside usable segments.
    conf:
        ``(n_frames,)`` confidence of ``point`` in each frame (0 where missing).
    segments:
        Array slices of the continuous stretches that were filtered.
    cutoff_hz:
        Filter cutoff actually used (after clamping below Nyquist).
    """

    side: str
    point: str
    raw: np.ndarray
    smooth: np.ndarray
    conf: np.ndarray
    segments: list[slice] = field(default_factory=list)
    cutoff_hz: float = float("nan")


@dataclass
class Extremum:
    """A peak found in a :class:`FootSignal`, before conversion to an event row."""

    position: float  # fractional array position (0 = first stored frame)
    index: int  # nearest whole array position
    strength: float  # edge-aware prominence divided by the signal's robust range


# ---------------------------------------------------------------------------
# Points and signals
# ---------------------------------------------------------------------------


def pelvis_x(kp: Keypoints, min_conf: float = 0.3, max_gap_s: float | None = None) -> np.ndarray:
    """Horizontal pelvis position: the mean of the two hips.

    Parameters
    ----------
    kp:
        Keypoints of one clip.
    min_conf:
        Hip points below this confidence count as missing.
    max_gap_s:
        Each hip is interpolated across its own gaps up to this long before
        the two are averaged; None means :data:`PELVIS_MAX_GAP_S`.

    Returns
    -------
    numpy.ndarray
        ``(n_frames,)`` pixel x of the pelvis; NaN where a hip is missing for
        longer than ``max_gap_s``.

    Raises
    ------
    ValueError
        If the keypoints have neither ``LEFT_HIP`` nor ``RIGHT_HIP``.

    Notes
    -----
    In a side view the two hips are not at the same image x: perspective
    moves them apart by tens of pixels, by an amount that changes as the
    walker crosses the image. Falling back to the one visible hip frame by
    frame would add a step of half that offset to every foot signal, which
    the peak picker sees (the same reason :func:`choose_foot_point` chooses
    once per clip). So each hip is first interpolated across its own gaps,
    and a frame where one hip is still missing is a missing pelvis. The
    choice of hips is made once per clip: a file with one hip, or whose
    other hip has fewer than half its confident frames, uses that hip alone
    throughout.
    """
    names = [n for n in ("LEFT_HIP", "RIGHT_HIP") if kp.has(n)]
    if not names:
        raise ValueError(f"{kp.clip_id}: needs LEFT_HIP or RIGHT_HIP to locate the pelvis")
    counts = [int(np.sum(kp.confidence(n) >= min_conf)) for n in names]
    names = [n for n, count in zip(names, counts) if count >= _POINT_USABLE_FRACTION * max(counts)]
    max_gap_s = PELVIS_MAX_GAP_S if max_gap_s is None else max_gap_s
    max_gap_frames = int(round(max_gap_s * kp.rate)) if kp.n_frames > 1 else 0
    hips = [fill_gaps(kp.masked_point(n, min_conf)[:, 0], kp.time_s, max_gap_frames) for n in names]
    # NaN in either hip makes the mean NaN, which is the point.
    return np.mean(hips, axis=0)


def choose_foot_point(kp: Keypoints, side: str, part: str, min_conf: float = 0.3) -> str | None:
    """Pick the keypoint to use for one foot, falling back along a chain.

    Parameters
    ----------
    kp:
        Keypoints of one clip.
    side:
        ``"L"`` or ``"R"``.
    part:
        ``"heel"`` (HEEL, then ANKLE), ``"toe"`` (FOOT_INDEX, BIG_TOE, then
        ANKLE) or ``"ankle"``. A bare keypoint part such as ``"KNEE"`` is
        also accepted and used without fallback.
    min_conf:
        Confidence threshold used to count usable frames.

    Returns
    -------
    str or None
        Full keypoint name such as ``"RIGHT_FOOT_INDEX"``, or None if no
        candidate has any confident frame.

    Notes
    -----
    The choice is made once per side for the whole clip rather than frame by
    frame: mixing heel and ankle within one signal would add a step of a few
    centimetres wherever the source changes, and the peak picker would see it.
    A candidate is skipped if it is missing from the file or has fewer than
    half the confident frames of the best candidate (some backends export a
    heel that is almost always zero confidence).
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    prefix = "LEFT_" if side == "L" else "RIGHT_"
    chain = FOOT_POINT_CHAINS.get(part.lower(), (part.upper(),))
    candidates = [prefix + p for p in chain if kp.has(prefix + p)]
    counts = [int(np.sum(kp.confidence(name) >= min_conf)) for name in candidates]
    best = max(counts, default=0)
    if best == 0:
        return None
    # The best candidate always passes, so this always finds one.
    return next(name for name, count in zip(candidates, counts) if count >= _POINT_USABLE_FRACTION * best)


def fill_gaps(values: np.ndarray, time_s: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Linearly interpolate interior NaN runs of at most ``max_gap_frames`` frames.

    Interpolation is done against ``time_s`` so that a variable frame rate is
    handled correctly. Leading and trailing NaNs and longer gaps are left as
    NaN; they become breaks between segments.

    Parameters
    ----------
    values:
        ``(n_frames,)`` signal with NaN where missing.
    time_s:
        ``(n_frames,)`` frame times.
    max_gap_frames:
        Longest run of missing frames to fill.

    Returns
    -------
    numpy.ndarray
        A filled copy of ``values``.
    """
    out = np.asarray(values, dtype=float).copy()
    if max_gap_frames <= 0:
        return out
    valid = np.isfinite(out)
    if valid.sum() < 2:
        return out
    for start, stop in _runs(~valid):
        interior = start > 0 and stop < len(out)
        if interior and stop - start <= max_gap_frames:
            out[start:stop] = np.interp(time_s[start:stop], time_s[valid], out[valid])
    return out


def finite_segments(values: np.ndarray, min_length: int) -> list[slice]:
    """Slices of the continuous finite stretches at least ``min_length`` frames long."""
    return [slice(a, b) for a, b in _runs(np.isfinite(values)) if b - a >= min_length]


def effective_cutoff(cutoff_hz: float, fps: float) -> float:
    """Cutoff clamped to :data:`MAX_CUTOFF_FRACTION_OF_NYQUIST` of the Nyquist frequency."""
    if cutoff_hz <= 0:
        raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}")
    return float(min(cutoff_hz, MAX_CUTOFF_FRACTION_OF_NYQUIST * fps / 2.0))


def lowpass(values: np.ndarray, fps: float, cutoff_hz: float, order: int = FILTER_ORDER) -> np.ndarray:
    """Zero-phase low-pass Butterworth filter of one continuous segment.

    Zero phase (forward-backward filtering) matters here: an ordinary filter
    would delay every peak, and so every event, by a few frames.

    Parameters
    ----------
    values:
        ``(n,)`` finite samples of one segment.
    fps:
        Sampling rate in Hz.
    cutoff_hz:
        Requested cutoff; clamped with :func:`effective_cutoff`.
    order:
        Filter order before forward-backward doubling.

    Returns
    -------
    numpy.ndarray
        Filtered samples, same length as ``values``.
    """
    values = np.asarray(values, dtype=float)
    cutoff = effective_cutoff(cutoff_hz, fps)
    sos = butter(order, cutoff / (fps / 2.0), btype="low", output="sos")
    # Pad by a fixed time (FILTER_PAD_PERIODS cutoff periods) rather than
    # scipy's fixed 15 samples, which is only 62 ms at 240 fps and lets the
    # filter's start-up flatten the ends. Never pad less than scipy would,
    # and never more than a short segment allows.
    padlen = max(3 * (2 * len(sos) + 1), int(np.ceil(FILTER_PAD_PERIODS * fps / cutoff)))
    return sosfiltfilt(sos, values, padlen=min(padlen, len(values) - 1))


def foot_signals(
    kp: Keypoints,
    part: str,
    *,
    direction: int,
    min_conf: float = 0.3,
    max_gap_s: float = 0.1,
    cutoff_hz: float = 6.0,
    min_segment_s: float = 0.3,
) -> dict[str, FootSignal]:
    """Build ``direction * (foot_x - pelvis_x)`` for both sides.

    Parameters
    ----------
    kp:
        Keypoints of one clip.
    part:
        Foot point to use, see :func:`choose_foot_point`.
    direction:
        +1 if the walker moves left to right in the image, -1 otherwise.
        Multiplying by it makes "in front of the pelvis" positive either way.
    min_conf:
        Points below this confidence are treated as missing.
    max_gap_s:
        Gaps up to this long are interpolated; longer gaps split the signal.
    cutoff_hz:
        Low-pass cutoff (4th-order Butterworth, zero phase).
    min_segment_s:
        Continuous stretches shorter than this are dropped (too short to
        filter or to contain a whole peak).

    Returns
    -------
    dict
        ``{"L": FootSignal, "R": FootSignal}``; a side with no usable point is
        left out (with a warning).
    """
    if direction not in (1, -1):
        raise ValueError(f"direction must be +1 or -1, got {direction!r}")
    fps = kp.rate
    pelvis = pelvis_x(kp, min_conf)
    max_gap_frames = int(round(max_gap_s * fps))
    # Below about 5 samples a 4th-order filter has nothing to work with.
    min_length = max(int(np.ceil(min_segment_s * fps)), 5)
    cutoff = effective_cutoff(cutoff_hz, fps)

    signals: dict[str, FootSignal] = {}
    for side in SIDES:
        name = choose_foot_point(kp, side, part, min_conf)
        if name is None:
            warnings.warn(f"{kp.clip_id}: no usable {part} point for side {side}; side skipped", stacklevel=2)
            continue
        foot = kp.masked_point(name, min_conf)[:, 0]
        raw = fill_gaps(direction * (foot - pelvis), kp.time_s, max_gap_frames)
        segments = finite_segments(raw, min_length)
        smooth = np.full_like(raw, np.nan)
        for seg in segments:
            smooth[seg] = lowpass(raw[seg], fps, cutoff)
        signals[side] = FootSignal(
            side=side,
            point=name,
            raw=raw,
            smooth=smooth,
            conf=kp.confidence(name).copy(),
            segments=segments,
            cutoff_hz=cutoff,
        )
    return signals


# ---------------------------------------------------------------------------
# Peaks
# ---------------------------------------------------------------------------


def robust_range(values: np.ndarray) -> float:
    """Spread of a signal ignoring outliers: 95th minus 5th percentile of its finite values."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    p5, p95 = np.percentile(finite, [5, 95])
    return float(p95 - p5)


def find_extrema(
    signal: FootSignal,
    kind: str,
    fps: float,
    *,
    min_event_interval_s: float = 0.5,
    prominence_frac: float = 0.15,
    refine: bool = True,
) -> list[Extremum]:
    """Find the maxima (``kind="max"``) or minima (``kind="min"``) of a foot signal.

    Parameters
    ----------
    signal:
        Output of :func:`foot_signals` for one side.
    kind:
        ``"max"`` for heel strikes, ``"min"`` for toe offs.
    fps:
        Sampling rate (:attr:`Keypoints.rate`) used to turn
        ``min_event_interval_s`` and :data:`EDGE_RELAX_MAX_S` into samples.
    min_event_interval_s:
        Two peaks of the same kind closer than this are not both kept.
    prominence_frac:
        Minimum prominence as a fraction of the signal's robust range
        (:func:`robust_range`). The range is taken over all segments of the
        signal, so a short stretch of data cannot lower the bar for itself.
        For a peak within :data:`EDGE_RELAX_MAX_S` of a segment edge, the
        edge side needs only :data:`EDGE_PROMINENCE_FRACTION` of it once the
        fall is clearly larger than the jitter (:func:`noise_sd`, see
        :func:`_edge_aware_drop`), so events in the last few frames are not lost.
    refine:
        Fit a parabola through each peak and its two neighbours to place it
        between frames.

    Returns
    -------
    list of Extremum
        Sorted by position. Peaks are never placed on the first or last
        sample of a segment, since there is no evidence either side of them.
    """
    if kind not in ("max", "min"):
        raise ValueError(f"kind must be 'max' or 'min', got {kind!r}")
    sign = 1.0 if kind == "max" else -1.0
    spread = robust_range(signal.smooth)
    if spread <= 0:
        return []
    needed = prominence_frac * spread
    distance = max(1, int(round(min_event_interval_s * fps)))
    max_edge_samples = int(np.floor(EDGE_RELAX_MAX_S * fps + 1e-9))
    noise = noise_sd(signal)
    found: list[Extremum] = []
    for seg in signal.segments:
        y = sign * signal.smooth[seg]
        peaks, _ = find_peaks(y, distance=distance)
        for peak in peaks:
            drop = _edge_aware_drop(y, peak, max_edge_samples, noise)
            if drop < needed:
                continue
            offset = parabolic_offset(y[peak - 1], y[peak], y[peak + 1]) if refine else 0.0
            found.append(
                Extremum(
                    position=seg.start + peak + offset,
                    index=int(seg.start + peak),
                    strength=float(drop / spread),
                )
            )
    return sorted(found, key=lambda e: e.position)


def noise_sd(signal: FootSignal) -> float:
    """Robust estimate of the frame-to-frame jitter in ``signal.raw``, in pixels.

    Taken from second differences within each segment. For white jitter of
    standard deviation ``s`` their spread is ``sqrt(6) * s`` at any frame
    rate, while the movement itself adds almost nothing: a planted foot
    moves linearly relative to the pelvis, which has no second difference.
    The median absolute deviation is used, so the large values at the sharp
    turns of the signal (the kinks at HS and TO) do not count.
    """
    second = [np.diff(signal.raw[seg], 2) for seg in signal.segments if seg.stop - seg.start > 2]
    if not second:
        return 0.0
    d2 = np.concatenate(second)
    return float(1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6.0))


def _edge_aware_drop(y: np.ndarray, peak: int, max_edge_samples: int, noise: float = 0.0) -> float:
    """How far a peak stands out above its surroundings, allowing for segment edges.

    This is scipy's *prominence* with one change. On each side we look from
    the peak to the nearest higher sample (or the end of the data) and take
    the drop to the lowest point in between; the prominence is the smaller
    of the two drops. Next to the start or end of a segment that drop may be
    cut short: if the peak is at most ``max_edge_samples`` from the edge and
    the lowest point is the very first or last sample, the signal was still
    falling when the data ran out, so a real heel strike a few frames before
    the end of a clip would fail the test. Such a *censored* drop is divided
    by :data:`EDGE_PROMINENCE_FRACTION` before the comparison.

    Two guards keep jitter from passing this way. A censored drop ends on
    the outermost sample, which zero-phase filtering leaves almost
    unsmoothed (it keeps nearly all of that frame's ``noise``), so the drop
    is first shortened by :data:`EDGE_NOISE_MARGIN` times ``noise``: a fall
    that one noisy sample could explain is not relaxed. And only one side
    is relaxed; the other must pass in full, even in a segment so short that
    both sides run into an edge.
    """
    sides = []
    for side in (y[peak::-1], y[peak:]):  # each side starts at the peak and moves outwards
        higher = np.flatnonzero(side[1:] > side[0])
        # find_peaks never returns a peak with a higher neighbour, so the window is never empty.
        window = side[1 : higher[0] + 1] if higher.size else side[1:]
        drop = side[0] - window.min()
        censored = higher.size == 0 and len(window) <= max_edge_samples and np.argmin(window) == len(window) - 1
        relaxed = max(drop, (drop - EDGE_NOISE_MARGIN * noise) / EDGE_PROMINENCE_FRACTION) if censored else drop
        sides.append((drop, relaxed))
    (drop0, relaxed0), (drop1, relaxed1) = sides
    # Relax one side or the other, never both.
    return float(max(min(relaxed0, drop1), min(drop0, relaxed1)))


def parabolic_offset(left: float, centre: float, right: float) -> float:
    """Sub-frame offset of a peak from a parabola through three samples.

    Returns a value in [-0.5, 0.5] to add to the centre sample's position.
    """
    denom = left - 2.0 * centre + right
    if denom == 0 or not np.isfinite(denom):
        return 0.0
    return float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))


def positions_to_frames(kp: Keypoints, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert fractional array positions into source-video frames and times.

    Parameters
    ----------
    kp:
        Keypoints the positions refer to.
    positions:
        Fractional indices into ``kp``'s arrays (0 = first stored frame).

    Returns
    -------
    frames, times:
        ``frames`` are the nearest whole source-video frame indices;
        ``times`` are the presentation times of the fractional positions, so
        sub-frame refinement is kept in ``time_s`` even though ``frame`` is
        an integer.

    Notes
    -----
    Array position ``i`` is source frame ``kp.frame_idx[i]``, which differs
    from ``i`` whenever the keypoints start part-way into a video or skip
    frames. The fractional source frame is interpolated between neighbouring
    ``frame_idx`` values and its time looked up with
    :func:`gaitbench.formats.frames_to_times`, which handles variable frame rates.
    """
    positions = np.asarray(positions, dtype=float)
    fractional_frames = np.interp(positions, np.arange(kp.n_frames), kp.frame_idx.astype(float))
    times = frames_to_times(fractional_frames, kp)
    return np.round(fractional_frames).astype(np.int64), times


# ---------------------------------------------------------------------------
# Walking direction
# ---------------------------------------------------------------------------


def estimate_direction(kp: Keypoints, min_conf: float = 0.3) -> tuple[int, dict[str, Any]]:
    """Estimate whether the walker moves left-to-right (+1) or right-to-left (-1) in the image.

    Three cues are tried in order; the first one that is clear wins:

    1. ``"foot"``: toe x minus heel (or ankle) x. The foot points forward, so
       this works with a panning camera and even when the walker stays put
       in the image.
    2. ``"nose"``: nose x minus mid-shoulder x (or mid-hip x). The nose is
       on the front of the face, so it is ahead of the shoulders whenever the
       walker looks where they are going. Also pan-invariant.
    3. ``"displacement"``: how far the pelvis moved across the image during
       the clip. Wrong when the camera pans faster than the walker and useless
       when it pans at the same speed, so it is only a last resort.

    If nothing is clear, +1 is returned with ``info["method"] == "default"``
    and a warning.

    Parameters
    ----------
    kp:
        Keypoints of one clip.
    min_conf:
        Points below this confidence are ignored.

    Returns
    -------
    direction, info:
        ``direction`` is +1 or -1. ``info`` says which cue decided
        (``"method"``) and holds the numbers behind every cue tried, for
        checking by hand.
    """
    info: dict[str, Any] = {"method": "default"}

    diffs = []
    per_side = {}
    for side in SIDES:
        toe = choose_foot_point(kp, side, "toe", min_conf)
        back = choose_foot_point(kp, side, "heel", min_conf)
        if toe is None or back is None or toe == back:  # both fell back to the ankle: nothing to compare
            continue
        d = kp.masked_point(toe, min_conf)[:, 0] - kp.masked_point(back, min_conf)[:, 0]
        d = d[np.isfinite(d)]
        per_side[side] = {"points": (toe, back), "n_frames": int(d.size)}
        if d.size:
            per_side[side]["median_px"] = float(np.median(d))
        diffs.append(d)
    foot = _sign_cue(np.concatenate(diffs) if diffs else np.array([]))
    info["foot"] = {**foot, "per_side": per_side}
    if foot["sign"] != 0:
        return foot["sign"], {**info, "method": "foot"}

    if kp.has("NOSE"):
        trunk = _mid_x(kp, "LEFT_SHOULDER", "RIGHT_SHOULDER", min_conf)
        if trunk is None or np.isfinite(trunk).sum() < _MIN_DIRECTION_FRAMES:
            trunk = _mid_x(kp, "LEFT_HIP", "RIGHT_HIP", min_conf)
        if trunk is not None:
            d = kp.masked_point("NOSE", min_conf)[:, 0] - trunk
            nose = _sign_cue(d[np.isfinite(d)])
            info["nose"] = nose
            if nose["sign"] != 0:
                return nose["sign"], {**info, "method": "nose"}

    hips = _mid_x(kp, "LEFT_HIP", "RIGHT_HIP", min_conf)
    if hips is not None and np.isfinite(hips).sum() >= _MIN_DIRECTION_FRAMES:
        t, x = kp.time_s[np.isfinite(hips)], hips[np.isfinite(hips)]
        slope = float(np.polyfit(t - t[0], x, 1)[0])
        info["displacement"] = {"px_per_s": slope, "n_frames": int(len(t))}
        # A walker crossing the image moves hundreds of px/s; a few px/s is drift.
        if abs(slope) > 5.0:
            return (1 if slope > 0 else -1), {**info, "method": "displacement"}

    warnings.warn(f"{kp.clip_id}: walking direction unclear; assuming left-to-right (+1)", stacklevel=2)
    return 1, info


def resolve_direction(kp: Keypoints, direction: int | None, min_conf: float) -> tuple[int, dict[str, Any]]:
    """Use ``direction`` if given (+1/-1), otherwise :func:`estimate_direction`."""
    if direction is None:
        return estimate_direction(kp, min_conf)
    if direction not in (1, -1):
        raise ValueError(f"direction must be +1, -1 or None, got {direction!r}")
    return int(direction), {"method": "given"}


# ---------------------------------------------------------------------------
# The shared detector
# ---------------------------------------------------------------------------


def detect_relative_extrema(
    kp: Keypoints,
    *,
    hs_part: str,
    to_part: str,
    detector: str,
    min_conf: float = 0.3,
    max_gap_s: float = 0.1,
    cutoff_hz: float = 6.0,
    min_event_interval_s: float = 0.5,
    prominence_frac: float = 0.15,
    direction: int | None = None,
    refine: bool = True,
    enforce_alternation: bool = True,
    min_segment_s: float = 0.3,
) -> pd.DataFrame:
    """Heel strikes at maxima and toe offs at minima of foot-minus-pelvis signals.

    This is the engine behind :func:`gaitbench.detectors.zeni.detect_zeni`
    and :func:`gaitbench.detectors.stenum.detect_stenum`; see those for what
    the parameters mean in each method.

    Parameters
    ----------
    kp:
        Keypoints of one clip.
    hs_part, to_part:
        Foot points (see :func:`choose_foot_point`) whose signals give heel
        strikes (maxima) and toe offs (minima).
    detector:
        Name written to the ``detector`` column.
    min_conf, max_gap_s, cutoff_hz, min_segment_s:
        Passed to :func:`foot_signals`.
    min_event_interval_s, prominence_frac, refine:
        Passed to :func:`find_extrema`.
    direction:
        +1/-1 to skip :func:`estimate_direction`.
    enforce_alternation:
        Drop the weaker of two same-type events on one side that have no
        event of the other type between them (see :func:`alternate_events`).

    Returns
    -------
    pandas.DataFrame
        Validated events table (``formats.EVENT_COLUMNS``). ``confidence`` is
        the keypoint confidence of the foot point at the event's frame.

    Raises
    ------
    ValueError
        If the keypoints have no hip points, or ``direction`` is not +1/-1/None.

    Notes
    -----
    Known limits: an event can only be found if the pelvis and the foot
    point are both visible around it. Near either end of a stretch of data
    only part of a peak is visible, so events within about 50 ms of an end
    may be missed, and occasionally one up to about 0.12 s away (just past
    :data:`EDGE_RELAX_MAX_S`). That limit is in time, not frames: 50 ms is
    one frame at 25 fps, three at 60 fps and twelve at 240 fps. Noisy
    keypoints widen it, since a fall at the very edge must be clearly larger
    than the jitter to count. The walking direction is estimated once per
    clip, so a there-and-back walk must be split into single passes first.
    """
    defaults = {
        "clip_id": kp.clip_id,
        "tool": TOOL_NAME,
        "tool_version": __version__,
        "backend": str(kp.meta.get("backend", "")),
        "detector": detector,
    }
    if kp.n_frames < 3:
        return make_events([], defaults=defaults)

    walk_dir, _ = resolve_direction(kp, direction, min_conf)
    fps = kp.rate
    signal_kwargs = dict(
        direction=walk_dir,
        min_conf=min_conf,
        max_gap_s=max_gap_s,
        cutoff_hz=cutoff_hz,
        min_segment_s=min_segment_s,
    )
    hs_signals = foot_signals(kp, hs_part, **signal_kwargs)
    to_signals = hs_signals if to_part == hs_part else foot_signals(kp, to_part, **signal_kwargs)
    peak_kwargs = dict(min_event_interval_s=min_event_interval_s, prominence_frac=prominence_frac, refine=refine)

    records: list[dict[str, Any]] = []
    for side in SIDES:
        if side not in hs_signals or side not in to_signals:
            continue
        found = [("HS", e) for e in find_extrema(hs_signals[side], "max", fps, **peak_kwargs)]
        found += [("TO", e) for e in find_extrema(to_signals[side], "min", fps, **peak_kwargs)]
        found.sort(key=lambda item: item[1].position)
        if enforce_alternation:
            found = alternate_events(found, hs_signals[side], to_signals[side])
        if not found:
            continue
        frames, times = positions_to_frames(kp, np.array([e.position for _, e in found]))
        for (event, ext), frame, time in zip(found, frames, times):
            signal = hs_signals[side] if event == "HS" else to_signals[side]
            records.append(
                {
                    "side": side,
                    "event": event,
                    "frame": int(frame),
                    "time_s": float(time),
                    "confidence": float(signal.conf[ext.index]),
                }
            )
    return make_events(records, defaults=defaults)


def alternate_events(
    found: list[tuple[str, Extremum]],
    hs_signal: FootSignal,
    to_signal: FootSignal,
) -> list[tuple[str, Extremum]]:
    """Make one side's events alternate HS, TO, HS, ... by dropping weaker duplicates.

    Between two heel strikes of the same foot there must be a toe off, and
    vice versa. When two events of the same type follow each other, the one
    with the smaller prominence is dropped, but only if both signals were
    continuous between them. If the data had a break there, the missing event
    may simply have been invisible, and both events are kept.

    Parameters
    ----------
    found:
        ``(event_type, Extremum)`` pairs for one side, sorted by position.
    hs_signal, to_signal:
        The signals the HS and TO events came from.

    Returns
    -------
    list
        The kept pairs, still sorted by position.
    """
    kept: list[tuple[str, Extremum]] = []
    for event, ext in found:
        if kept and kept[-1][0] == event:
            prev = kept[-1][1]
            lo, hi = prev.index, ext.index + 1
            continuous = np.isfinite(hs_signal.smooth[lo:hi]).all() and np.isfinite(to_signal.smooth[lo:hi]).all()
            if continuous:
                if ext.strength > prev.strength:
                    kept[-1] = (event, ext)
                continue
        kept.append((event, ext))
    return kept


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``(start, stop)`` of each run of True values in a boolean array."""
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _mid_x(kp: Keypoints, left: str, right: str, min_conf: float) -> np.ndarray | None:
    """Mean x of a left/right pair (or either alone); None if neither is in the file."""
    xs = [kp.masked_point(n, min_conf)[:, 0] for n in (left, right) if kp.has(n)]
    if not xs:
        return None
    stacked = np.vstack(xs)
    n_valid = np.isfinite(stacked).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_valid > 0, np.nansum(stacked, axis=0) / n_valid, np.nan)


def _sign_cue(d: np.ndarray) -> dict[str, Any]:
    """Sign of the median of ``d`` if enough frames agree with it, else 0."""
    out: dict[str, Any] = {"sign": 0, "n_frames": int(d.size)}
    if d.size < _MIN_DIRECTION_FRAMES:
        return out
    median = float(np.median(d))
    out["median_px"] = median
    if median == 0:
        return out
    agreement = float(np.mean(np.sign(d) == np.sign(median)))
    out["agreement"] = agreement
    if agreement >= _MIN_DIRECTION_AGREEMENT:
        out["sign"] = 1 if median > 0 else -1
    return out

"""Zeni et al. (2008) coordinate-based gait events, adapted to 2-D keypoints.

Zeni, Richards and Higginson, "Two simple methods for determining gait events
during treadmill and overground walking using kinematic data", Gait & Posture
27(4), 2008. The original uses motion-capture markers on the heel, toe and
sacrum:

* heel strike = the moment the heel is furthest in front of the sacrum;
* toe off = the moment the toe is furthest behind the sacrum.

Here the sacrum is replaced by the mid-hip keypoint, distances are horizontal
pixels in the side-view image, and the toe is FOOT_INDEX (or BIG_TOE). Because
the distances are relative to the pelvis, the rule does not care whether the
camera is fixed or panning with the walker.
"""

from __future__ import annotations

import pandas as pd

from ..formats import Keypoints
from .base import detect_relative_extrema


def detect_zeni(
    kp: Keypoints,
    *,
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
    """Detect heel strikes and toe offs with the Zeni rule.

    HS are local maxima of ``direction * (heel_x - pelvis_x)``; TO are local
    minima of ``direction * (toe_x - pelvis_x)``. Heel falls back to ANKLE and
    toe to BIG_TOE then ANKLE when the backend lacks them.

    Parameters
    ----------
    kp:
        Side-view keypoints of one clip.
    min_conf:
        Keypoints below this confidence are treated as missing.
    max_gap_s:
        Missing stretches up to this long are linearly interpolated; longer
        ones split the signal into separate segments.
    cutoff_hz:
        Cutoff of the zero-phase 4th-order Butterworth low-pass filter.
        Clamped below Nyquist for low frame rates.
    min_event_interval_s:
        Minimum time between two events of the same type on the same side.
    prominence_frac:
        Minimum peak prominence as a fraction of the signal's 5-95 % range.
    direction:
        +1 (left-to-right in the image) or -1; None estimates it with
        :func:`gaitbench.detectors.base.estimate_direction`.
    refine:
        Place events between frames with a parabolic fit. ``frame`` is still
        a whole frame; ``time_s`` keeps the sub-frame time.
    enforce_alternation:
        Drop the weaker of two same-type events on one side that have no
        event of the other type between them.
    min_segment_s:
        Continuous stretches of data shorter than this are ignored.

    Returns
    -------
    pandas.DataFrame
        Validated events table with ``detector="zeni"``. ``confidence`` is
        the confidence of the heel/toe keypoint at the event frame.
    """
    return detect_relative_extrema(
        kp,
        hs_part="heel",
        to_part="toe",
        detector="zeni",
        min_conf=min_conf,
        max_gap_s=max_gap_s,
        cutoff_hz=cutoff_hz,
        min_event_interval_s=min_event_interval_s,
        prominence_frac=prominence_frac,
        direction=direction,
        refine=refine,
        enforce_alternation=enforce_alternation,
        min_segment_s=min_segment_s,
    )

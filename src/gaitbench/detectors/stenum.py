"""Stenum et al. (2021) gait events from video keypoints, re-implemented.

Stenum, Rossi and Roemmich, "Two-dimensional video-based analysis of human
gait using pose estimation", PLOS Computational Biology 17(4), 2021. They
ran OpenPose on side-view video and used the horizontal distance between
the ankle and the pelvis (mid-hip):

* heel strike at its maxima (ankle furthest in front of the pelvis);
* toe off at its minima (ankle furthest behind the pelvis).

This module is written from that description only; no code from the
authors' (GPL) MATLAB release was read or copied. It is the Zeni rule with
the ankle used for both events, so it shares all of its machinery with
:mod:`gaitbench.detectors.zeni`. The ankle is available from every pose
model, which makes this the baseline for backends without foot points.
"""

from __future__ import annotations

import pandas as pd

from ..formats import Keypoints
from .base import detect_relative_extrema


def detect_stenum(
    kp: Keypoints,
    *,
    min_conf: float = 0.3,
    max_gap_s: float = 0.1,
    cutoff_hz: float = 6.0,
    min_event_interval_s: float = 0.5,
    prominence_frac: float = 0.15,
    direction: int | None = None,
    refine: bool = True,
    enforce_alternation: bool = False,
    min_segment_s: float = 0.3,
) -> pd.DataFrame:
    """Detect heel strikes and toe offs from the ankle-minus-pelvis distance.

    Parameters are those of :func:`gaitbench.detectors.zeni.detect_zeni`.
    ``enforce_alternation`` defaults to False here because the published
    method has no such step; on a single continuous signal maxima and minima
    alternate anyway unless a peak is too weak to pass ``prominence_frac``.

    Returns
    -------
    pandas.DataFrame
        Validated events table with ``detector="stenum"``. ``confidence`` is
        the ankle keypoint confidence at the event frame.
    """
    return detect_relative_extrema(
        kp,
        hs_part="ankle",
        to_part="ankle",
        detector="stenum",
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

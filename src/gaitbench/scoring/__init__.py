"""Scoring: compare detected gait events and cycles with ground truth.

Typical use::

    from gaitbench.scoring import match_events, event_metrics

    matches = match_events(events, labels, tolerance_s=0.1)
    table = event_metrics(matches, fps=60, group_by=("detector", "side", "event"))

* :func:`match_events` pairs predicted HS/TO with labelled ones (optimal
  one-to-one assignment within a tolerance) and :func:`agreement` does the
  same between two detectors or raters.
* :func:`event_metrics` turns the matches into recall, precision, F1, timing
  bias, SD, MAE, limits of agreement and the share within 1 and 2 frames.
* :func:`temporal_parameters` splits events into cycles (stride, stance,
  swing, step, double support) and :func:`compare_parameters` scores them
  against the cycles from the labels; :func:`cadence_spm` gives cadence.
* :func:`bland_altman` and :func:`icc_2_1` are the agreement statistics.
"""

from .match import MATCH_COLUMNS, agreement, assign_within_tolerance, match_events
from .metrics import event_metrics
from .params import PARAMETERS, PARAM_COLUMNS, cadence_spm, compare_parameters, pair_cycles, temporal_parameters
from .stats import bland_altman, icc_2_1, icc_2_1_matrix

__all__ = [
    "MATCH_COLUMNS",
    "PARAMETERS",
    "PARAM_COLUMNS",
    "agreement",
    "assign_within_tolerance",
    "bland_altman",
    "cadence_spm",
    "compare_parameters",
    "event_metrics",
    "icc_2_1",
    "icc_2_1_matrix",
    "match_events",
    "pair_cycles",
    "temporal_parameters",
]

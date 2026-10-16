"""Detection and timing metrics from a table of matched events.

:func:`event_metrics` turns the TP/FP/FN rows of
:func:`gaitbench.scoring.match_events` into the numbers on the scorecard:

* **Detection**: recall (share of reference events found), precision (share
  of predictions that are real) and F1.
* **Timing**, over true positives only, with error = predicted - reference:
  signed bias, SD, mean and median absolute error and the Bland-Altman 95%
  limits of agreement, all in milliseconds.
* **Frame bars** (when ``fps`` is given): the share of reference events found
  within 1 and within 2 frame periods. Missed events count as not within, so
  this combines detection and timing in one number, matching the plan's
  "heel strike within 1 frame" bar.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .match import TIME_EPS_S
from .stats import LOA_Z

COUNT_COLUMNS: tuple[str, ...] = ("n_truth", "n_pred", "tp", "fp", "fn")
RATE_COLUMNS: tuple[str, ...] = ("recall", "precision", "f1")
TIMING_COLUMNS: tuple[str, ...] = ("bias_ms", "sd_ms", "mae_ms", "median_abs_ms", "loa_low_ms", "loa_high_ms")
FRAME_COLUMNS: tuple[str, ...] = ("mae_frames", "p_within_1_frame", "p_within_2_frames")


def event_metrics(
    matches: pd.DataFrame,
    fps: float | Mapping[str, float] | None = None,
    group_by: Sequence[str] | None = ("detector", "event"),
) -> pd.DataFrame:
    """Summarise matched events per group.

    Parameters
    ----------
    matches:
        Output of :func:`~gaitbench.scoring.match_events` or
        :func:`~gaitbench.scoring.agreement` (needs ``status`` and
        ``error_s``; ``clip_id`` too when ``fps`` is a mapping).
    fps:
        Nominal frame rate used for the frame-based columns: one number for
        every clip, or a ``{clip_id: fps}`` mapping when clips were recorded
        at different rates. None leaves those columns out.
    group_by:
        Columns of ``matches`` to group by, e.g. ``("detector", "event")``
        or ``("backend", "detector", "side", "event")``. Empty or None gives
        one overall row.

    Returns
    -------
    pandas.DataFrame
        One row per group with the ``group_by`` columns and then:

        * ``n_truth`` (TP + FN), ``n_pred`` (TP + FP), ``tp``, ``fp``, ``fn``;
        * ``recall`` = TP / n_truth, ``precision`` = TP / n_pred,
          ``f1`` = 2 TP / (2 TP + FP + FN) (NaN when the denominator is 0);
        * ``bias_ms`` (mean error), ``sd_ms`` (sample SD, ddof = 1),
          ``mae_ms``, ``median_abs_ms``, ``loa_low_ms`` / ``loa_high_ms``
          (bias -/+ 1.96 SD) over true positives; NaN with no TP (SD and
          limits need two);
        * with ``fps``: ``mae_frames``, ``p_within_1_frame`` and
          ``p_within_2_frames`` (TP with ``|error| <= k / fps``, divided by
          ``n_truth``).
    """
    group_by = list(group_by or [])
    missing = [c for c in ["status", "error_s", *group_by] if c not in matches.columns]
    if missing:
        raise ValueError(f"matches is missing columns {missing}")

    frame_s = _frame_periods(matches, fps)
    columns = group_by + list(COUNT_COLUMNS + RATE_COLUMNS + TIMING_COLUMNS) + (list(FRAME_COLUMNS) if fps is not None else [])
    if matches.empty:
        return _empty_result(matches, columns, group_by)

    status = matches["status"].to_numpy(dtype=object)
    error = matches["error_s"].to_numpy(dtype=float)
    if group_by:
        groups = matches.groupby(group_by, sort=True, dropna=False).indices
    else:
        groups = {(): np.arange(len(matches))}

    rows = []
    for key, pos in groups.items():
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_by, key))
        row.update(_group_metrics(status[pos], error[pos], None if frame_s is None else frame_s[pos]))
        rows.append(row)
    out = pd.DataFrame(rows, columns=columns)
    for col in COUNT_COLUMNS:
        out[col] = out[col].astype(np.int64)
    return out


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _frame_periods(matches: pd.DataFrame, fps: float | Mapping[str, float] | None) -> np.ndarray | None:
    """Frame period in seconds for every row, or None without ``fps``."""
    if fps is None:
        return None
    if isinstance(fps, Mapping):
        if "clip_id" not in matches.columns:
            raise ValueError("fps given per clip but matches has no clip_id column")
        rates = matches["clip_id"].map(lambda c: fps.get(c, np.nan)).to_numpy(dtype=float)
        unknown = sorted(set(matches["clip_id"][~np.isfinite(rates)]))
        if unknown:
            raise ValueError(f"No fps given for clip(s) {unknown}")
    else:
        rates = np.full(len(matches), float(fps))
    if np.any(rates <= 0):
        raise ValueError("fps must be positive")
    return 1.0 / rates


def _group_metrics(status: np.ndarray, error_s: np.ndarray, frame_s: np.ndarray | None) -> dict[str, float]:
    is_tp = status == "TP"
    tp = int(is_tp.sum())
    fp = int((status == "FP").sum())
    fn = int((status == "FN").sum())
    n_truth, n_pred = tp + fn, tp + fp
    nan = float("nan")

    err = error_s[is_tp]
    ok = np.isfinite(err)
    err_ms = 1000.0 * err[ok]
    bias = float(err_ms.mean()) if err_ms.size else nan
    sd = float(err_ms.std(ddof=1)) if err_ms.size > 1 else nan
    abs_ms = np.abs(err_ms)

    row: dict[str, float] = {
        "n_truth": n_truth,
        "n_pred": n_pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": tp / n_truth if n_truth else nan,
        "precision": tp / n_pred if n_pred else nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else nan,
        "bias_ms": bias,
        "sd_ms": sd,
        "mae_ms": float(abs_ms.mean()) if abs_ms.size else nan,
        "median_abs_ms": float(np.median(abs_ms)) if abs_ms.size else nan,
        "loa_low_ms": bias - LOA_Z * sd,
        "loa_high_ms": bias + LOA_Z * sd,
    }
    if frame_s is not None:
        period = frame_s[is_tp][ok]
        abs_s = np.abs(err[ok])
        row["mae_frames"] = float(np.mean(abs_s / period)) if abs_s.size else nan
        for k, col in ((1, "p_within_1_frame"), (2, "p_within_2_frames")):
            hits = int(np.sum(abs_s <= k * period + TIME_EPS_S))
            row[col] = hits / n_truth if n_truth else nan
    return row


def _empty_result(matches: pd.DataFrame, columns: list[str], group_by: list[str]) -> pd.DataFrame:
    out = pd.DataFrame({c: pd.Series(dtype=float) for c in columns})
    for col in group_by:
        out[col] = pd.Series(dtype=matches[col].dtype)
    for col in COUNT_COLUMNS:
        out[col] = out[col].astype(np.int64)
    return out

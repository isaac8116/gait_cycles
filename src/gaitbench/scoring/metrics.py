"""Detection and timing metrics from a table of matched events.

:func:`event_metrics` turns the TP/FP/FN rows of
:func:`gaitbench.scoring.match_events` into the numbers on the scorecard:

* **Detection**: recall (share of reference events found), precision (share
  of predictions that are real) and F1.
* **Timing**, over true positives only, with error = predicted - reference:
  signed bias, SD, mean and median absolute error and the Bland-Altman 95%
  limits of agreement, all in milliseconds.
* **Frame bars** (when ``fps`` is given): the share of reference events found
  within 1 and within 2 frames of the detection stream. Missed events count
  as not within, so this combines detection and timing in one number,
  matching the plan's "heel strike within 1 frame" bar.

By default there is one row per prediction source (tool, tool_version,
backend, detector) and event type, the same sources
:func:`~gaitbench.scoring.match_events` scores separately. A ``group_by``
without some of those columns pools the sources that differ in them.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..formats import sample_rate
from .match import SOURCE_COLUMNS, TIME_EPS_S
from .stats import LOA_Z

COUNT_COLUMNS: tuple[str, ...] = ("n_truth", "n_pred", "tp", "fp", "fn")
RATE_COLUMNS: tuple[str, ...] = ("recall", "precision", "f1")
TIMING_COLUMNS: tuple[str, ...] = ("bias_ms", "sd_ms", "mae_ms", "median_abs_ms", "loa_low_ms", "loa_high_ms")
FRAME_COLUMNS: tuple[str, ...] = ("mae_frames", "p_within_1_frame", "p_within_2_frames")
# One row per source and event type, so separately scored sources are never pooled.
DEFAULT_GROUP_BY: tuple[str, ...] = (*SOURCE_COLUMNS, "event")


def event_metrics(
    matches: pd.DataFrame,
    fps: float | Mapping[str, float] | None = None,
    group_by: Sequence[str] | None = DEFAULT_GROUP_BY,
    *,
    slack_s: float | Mapping[str, float] = 0.0,
    same_frame_grid: bool = False,
) -> pd.DataFrame:
    """Summarise matched events per group.

    Parameters
    ----------
    matches:
        Output of :func:`~gaitbench.scoring.match_events` or
        :func:`~gaitbench.scoring.agreement` (needs ``status`` and
        ``error_s``; ``clip_id`` too when ``fps`` or ``slack_s`` is a
        mapping; ``pred_frame`` and ``truth_frame`` with ``same_frame_grid``).
    fps:
        Frame rate of the detection stream (the keypoints the predictions
        were made on) for the frame-based columns: one number for every
        clip, or a ``{clip_id: fps}`` mapping when clips were recorded at
        different rates. Errors are counted in that stream's own frames, so
        a 60 fps copy scored against 240 fps labels is judged in 60 fps
        frames, as the plan's "within 1 frame" bars mean. Use the rate
        actually sampled (:attr:`gaitbench.formats.Keypoints.rate`). None
        leaves those columns out.
    group_by:
        Columns of ``matches`` to group by, e.g. ``("detector", "event")``
        or ``("backend", "detector", "side", "event")``. Empty or None gives
        one overall row. The default, :data:`DEFAULT_GROUP_BY`, is every
        source column present in ``matches`` plus ``event``; leaving source
        columns out pools those sources.
    slack_s:
        Timestamp quantisation of the detection stream in seconds, added to
        ``k / fps`` in the within-k-frames tests: one number or a
        ``{clip_id: s}`` mapping. Get it from the stream's frame times with
        :func:`timestamp_slack`. Without it an event exactly one frame late
        can fail "within 1 frame" when times are rounded to a coarse time
        base (240 fps in a 1/600 s base: a 5 ms interval is 1.2 periods).
        0 (the default) suits exact timestamps.
    same_frame_grid:
        True when ``pred_frame`` and ``truth_frame`` number the same frames,
        at rate ``fps``: one rater against another on the same video, or
        detections on full-rate keypoints of the labelled video against its
        labels. Errors are then counted exactly, as ``|pred_frame -
        truth_frame|`` frames, and ``slack_s`` is not used. Leave it False
        for decimated copies (their frames keep the source video's
        numbering) and for different videos of the same walk.

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
        * with ``fps``: ``mae_frames`` (mean ``|error| * fps``, or mean
          frame difference with ``same_frame_grid``), ``p_within_1_frame``
          and ``p_within_2_frames`` (TP with ``|error| <= k / fps +
          slack_s``, or ``|pred_frame - truth_frame| <= k``, divided by
          ``n_truth``).
    """
    if group_by is DEFAULT_GROUP_BY:
        # A hand-made table may lack tool / tool_version; there is nothing to pool then.
        group_by = [c for c in DEFAULT_GROUP_BY if c in matches.columns]
    group_by = list(group_by or [])
    missing = [c for c in ["status", "error_s", *group_by] if c not in matches.columns]
    if missing:
        raise ValueError(f"matches is missing columns {missing}")

    offsets = _frame_offsets(matches, fps, slack_s, same_frame_grid)
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
        part = None if offsets is None else (offsets[0][pos], offsets[1][pos])
        row.update(_group_metrics(status[pos], error[pos], part))
        rows.append(row)
    out = pd.DataFrame(rows, columns=columns)
    for col in COUNT_COLUMNS:
        out[col] = out[col].astype(np.int64)
    return out


def timestamp_slack(time_s: Sequence[float] | np.ndarray, fps: float | None = None) -> float:
    """Timestamp quantisation of a frame stream, for ``slack_s`` of :func:`event_metrics`.

    The largest ``|interval - 1 / fps|`` between consecutive frame times.
    Phone video stores times in a coarse time base: 240 fps in a 1/600 s
    base alternates 2- and 3-tick intervals (3.33 and 5 ms around the
    4.17 ms period), which gives 0.83 ms. Intervals more than half a period
    off are dropped or repeated frames, not rounding, and are ignored.

    Parameters
    ----------
    time_s:
        Presentation times of the stream's frames in seconds, e.g.
        :attr:`gaitbench.formats.Keypoints.time_s`.
    fps:
        The stream's frame rate, the same value passed to
        :func:`event_metrics`. By default it is measured with
        :func:`gaitbench.formats.sample_rate`.

    Returns
    -------
    float
        The slack in seconds; 0 with fewer than two frames.
    """
    t = np.asarray(time_s, dtype=float)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return 0.0
    period = 1.0 / (float(fps) if fps else sample_rate(t))
    dev = np.abs(np.diff(t) - period)
    dev = dev[dev < 0.5 * period]
    return float(dev.max()) if dev.size else 0.0


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _per_clip(matches: pd.DataFrame, value: float | Mapping[str, float], name: str) -> np.ndarray:
    """A number, or a ``{clip_id: number}`` mapping, as one value per row of ``matches``."""
    if isinstance(value, Mapping):
        if "clip_id" not in matches.columns:
            raise ValueError(f"{name} given per clip but matches has no clip_id column")
        out = matches["clip_id"].map(lambda c: value.get(c, np.nan)).to_numpy(dtype=float)
        unknown = sorted(set(matches["clip_id"][~np.isfinite(out)]))
        if unknown:
            raise ValueError(f"No {name} given for clip(s) {unknown}")
        return out
    return np.full(len(matches), float(value))


def _frame_offsets(
    matches: pd.DataFrame,
    fps: float | Mapping[str, float] | None,
    slack_s: float | Mapping[str, float],
    same_frame_grid: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per row, ``|error|`` in frames and the allowance added to k in the within-k tests.

    None without ``fps``.
    """
    if fps is None:
        return None
    rates = _per_clip(matches, fps, "fps")
    if np.any(~(rates > 0)):
        raise ValueError("fps must be positive")
    if same_frame_grid:
        missing = [c for c in ("pred_frame", "truth_frame") if c not in matches.columns]
        if missing:
            raise ValueError(f"same_frame_grid needs columns {missing} in matches")
        frames = np.abs(matches["pred_frame"].to_numpy(dtype=float) - matches["truth_frame"].to_numpy(dtype=float))
        return frames, np.zeros(len(matches))
    slack = _per_clip(matches, slack_s, "slack_s")
    if np.any(~(slack >= 0)):
        raise ValueError("slack_s must be >= 0")
    return np.abs(matches["error_s"].to_numpy(dtype=float)) * rates, (slack + TIME_EPS_S) * rates


def _group_metrics(
    status: np.ndarray, error_s: np.ndarray, offsets: tuple[np.ndarray, np.ndarray] | None
) -> dict[str, float]:
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
    if offsets is not None:
        frames, allowance = offsets
        use = is_tp & np.isfinite(frames)
        frames, allowance = frames[use], allowance[use]
        row["mae_frames"] = float(frames.mean()) if frames.size else nan
        for k, col in ((1, "p_within_1_frame"), (2, "p_within_2_frames")):
            hits = int(np.sum(frames <= k + allowance))
            row[col] = hits / n_truth if n_truth else nan
    return row


def _empty_result(matches: pd.DataFrame, columns: list[str], group_by: list[str]) -> pd.DataFrame:
    out = pd.DataFrame({c: pd.Series(dtype=float) for c in columns})
    for col in group_by:
        out[col] = pd.Series(dtype=matches[col].dtype)
    for col in COUNT_COLUMNS:
        out[col] = out[col].astype(np.int64)
    return out

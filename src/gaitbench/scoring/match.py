"""One-to-one matching of predicted gait events to reference events.

A predicted heel strike (HS) or toe off (TO) is a true positive (TP) when it
is paired with a reference event of the same clip, side and type no more than
``tolerance_s`` away. Unpaired predictions are false positives (FP) and
unpaired reference events false negatives (FN).

Pairing is done with :func:`scipy.optimize.linear_sum_assignment`, not
greedily. Within each (clip, side, event) group it finds the largest number of
pairs that respect the tolerance and, among those, the pairing with the
smallest total ``|time error|``. A greedy nearest-first match (what
``gaitkit.evaluation.match_events`` does) can give one prediction the closer of
two reference events and leave the other prediction with nothing in range,
which counts one TP as an FP plus an FN.

Matching uses ``time_s``, not ``frame``: frame indices of a phone video and a
240 fps reference are not comparable, presentation times are.

Each prediction *source* is matched separately against the same reference. A
source is one combination of ``tool``, ``tool_version``, ``backend`` and
``detector``, so several detectors, pose backends or tool versions can be
scored in one call. Every source is scored on every reference clip: a source
with no events on a labelled clip (for example a detector that crashed) gets
all of that clip's reference events as FN.

The reference must hold one source per clip (one rater, or the simulator).
To compare two raters, or two detectors with no ground truth, use
:func:`agreement`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ..formats import validate_events

# Columns that identify where a set of events came from.
SOURCE_COLUMNS: tuple[str, ...] = ("tool", "tool_version", "backend", "detector")
# Events are only ever paired within one of these groups.
MATCH_KEYS: tuple[str, ...] = ("clip_id", "side", "event")
# The labelled stretch used by ``restrict_to_truth_span`` is taken per one of these.
SPAN_KEYS: tuple[str, ...] = ("clip_id", "event")

MATCH_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "detector",
    "backend",
    "side",
    "event",
    "status",
    "truth_time_s",
    "pred_time_s",
    "truth_frame",
    "pred_frame",
    "error_s",
    "tool",
    "tool_version",
)

# Slack on the tolerance so that, e.g., a 6-frame error at 60 fps still counts
# as within 0.1 s after floating-point rounding of frame / fps.
TIME_EPS_S = 1e-9


def match_events(
    pred: pd.DataFrame,
    truth: pd.DataFrame,
    tolerance_s: float = 0.1,
    restrict_to_truth_span: bool = True,
) -> pd.DataFrame:
    """Pair predicted events with ground-truth events and label each TP, FP or FN.

    Parameters
    ----------
    pred:
        Events table (:data:`gaitbench.formats.EVENT_COLUMNS`) from one or more
        sources (detector / backend / tool version). ``time_s`` must be set.
    truth:
        Labels table in the same format, one source per clip.
    tolerance_s:
        Largest ``|pred - truth|`` time difference that can still be paired
        (inclusive). The evaluation plan uses 0.1 s.
    restrict_to_truth_span:
        If True, predictions outside ``[first truth time - tolerance, last
        truth time + tolerance]`` of their clip and event type (both sides
        together) are dropped instead of being counted as FP, because labels
        may cover only part of a clip: heel strikes may be labelled
        throughout and toe offs over part of it or not at all. Predictions of
        an event type with no truth on their clip, and on clips with no
        truth at all, are then dropped too. The span is not taken per side:
        the sides alternate, so one side's first label normally comes half a
        stride after the other's, and a spurious event in that stretch would
        go uncounted. If False, every prediction is scored.

    Returns
    -------
    pandas.DataFrame
        One row per TP, FP and FN with columns :data:`MATCH_COLUMNS`:
        ``clip_id, detector, backend, side, event, status`` (``'TP'``,
        ``'FP'`` or ``'FN'``), ``truth_time_s, pred_time_s, truth_frame,
        pred_frame`` (floats, NaN on the side that is absent), ``error_s``
        (``pred - truth``, positive = late, NaN unless TP), then the
        source's ``tool`` and ``tool_version``. Sorted by source, clip, side,
        event and time. If ``pred`` is empty the FN rows have empty source
        columns.
    """
    tolerance_s = _check_tolerance(tolerance_s)
    pred = _prepare_events(pred, "pred")
    truth = _prepare_events(truth, "truth")
    _check_single_source(truth, "truth")
    sources = _sources(pred)

    if restrict_to_truth_span:
        span = truth.groupby(list(SPAN_KEYS))["time_s"].agg(["min", "max"])
        key = pd.MultiIndex.from_frame(pred.loc[:, list(SPAN_KEYS)])
        lo = span["min"].reindex(key).to_numpy(float) - tolerance_s - TIME_EPS_S
        hi = span["max"].reindex(key).to_numpy(float) + tolerance_s + TIME_EPS_S
        # NaN bounds (no truth of that event type on the clip) make both comparisons False.
        t = pred["time_s"].to_numpy(float)
        pred = pred.loc[(t >= lo) & (t <= hi)].reset_index(drop=True)

    pairs = match_tables(pred, truth, sources, key_cols=MATCH_KEYS, time_col="time_s", tolerance_s=tolerance_s)
    return _matches_frame(pairs, pred, truth)


def agreement(
    events_a: pd.DataFrame,
    events_b: pd.DataFrame,
    tolerance_s: float = 0.1,
    *,
    restrict_to_reference_span: bool = False,
) -> pd.DataFrame:
    """Match ``events_a`` against ``events_b`` taken as the reference.

    Use it where there is no ground truth: detector against detector (e.g.
    myogait's Zeni against ours, the plan's decision rule), one tool version
    against the previous one (E6), or one rater against another. "TP" then
    means "agrees within tolerance", FP "only in a", FN "only in b", and
    ``error_s`` is ``a - b``.

    Parameters
    ----------
    events_a:
        Events from one or more sources, each scored separately.
    events_b:
        Reference events, one source per clip.
    tolerance_s:
        Largest time difference that still counts as agreement (inclusive).
    restrict_to_reference_span:
        Passed on as ``restrict_to_truth_span``. Off by default because both
        tables normally cover the whole clip, and a source that found nothing
        on a clip should show up as disagreement, not be dropped.

    Returns
    -------
    pandas.DataFrame
        The same columns as :func:`match_events` (``truth_*`` = reference),
        plus ``ref_detector`` and ``ref_backend`` naming the reference source.
    """
    ref = _prepare_events(events_b, "events_b")
    _check_single_source(ref, "events_b")
    out = match_events(events_a, ref, tolerance_s=tolerance_s, restrict_to_truth_span=restrict_to_reference_span)
    names = ref.drop_duplicates("clip_id").set_index("clip_id")
    out["ref_detector"] = out["clip_id"].map(names["detector"]).fillna("").astype(str)
    out["ref_backend"] = out["clip_id"].map(names["backend"]).fillna("").astype(str)
    return out


# ---------------------------------------------------------------------------
# shared engine (also used to pair gait cycles in params.py)
# ---------------------------------------------------------------------------


def assign_within_tolerance(pred_t: np.ndarray, truth_t: np.ndarray, tolerance_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Optimal one-to-one pairing of two sets of times within a tolerance.

    Pairs further apart than ``tolerance_s`` get a cost larger than the sum of
    every allowed pair, so the assignment first maximises the number of
    allowed pairs and then minimises their total ``|difference|``.

    Parameters
    ----------
    pred_t, truth_t:
        1-D arrays of times in seconds, any order.
    tolerance_s:
        Largest difference that can be paired (inclusive, with
        :data:`TIME_EPS_S` slack).

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Indices into ``pred_t`` and ``truth_t`` of the paired elements.
    """
    pred_t = np.asarray(pred_t, dtype=float)
    truth_t = np.asarray(truth_t, dtype=float)
    empty = np.zeros(0, dtype=np.int64)
    if pred_t.size == 0 or truth_t.size == 0:
        return empty, empty
    cost = np.abs(pred_t[:, None] - truth_t[None, :])
    allowed = cost <= tolerance_s + TIME_EPS_S
    if not allowed.any():
        return empty, empty
    # Any allowed matching costs at most n_pairs * tolerance, so one more
    # allowed pair always outweighs any saving in error.
    forbidden = (tolerance_s + 1.0) * (min(pred_t.size, truth_t.size) + 1)
    rows, cols = linear_sum_assignment(np.where(allowed, cost, forbidden))
    keep = allowed[rows, cols]
    return rows[keep].astype(np.int64), cols[keep].astype(np.int64)


def match_tables(
    pred: pd.DataFrame,
    truth: pd.DataFrame,
    sources: Sequence[tuple[str, ...]],
    *,
    key_cols: Sequence[str],
    time_col: str,
    tolerance_s: float,
) -> pd.DataFrame:
    """Pair rows of ``pred`` and ``truth`` per source and key group.

    Parameters
    ----------
    pred, truth:
        Tables with a default RangeIndex, the :data:`SOURCE_COLUMNS` (pred
        only), ``key_cols`` and ``time_col``.
    sources:
        Source tuples (values of :data:`SOURCE_COLUMNS`) to score. A source
        with no rows in ``pred`` still gets the FN rows.
    key_cols:
        Columns that must be equal for two rows to be paired.
    time_col:
        Column holding the times to pair on.
    tolerance_s:
        Passed to :func:`assign_within_tolerance`.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`SOURCE_COLUMNS`, ``key_cols``, ``status``, ``pred_row``
        and ``truth_row`` (row positions, -1 where absent).
    """
    key_cols = list(key_cols)
    truth_groups = _group_rows(truth, key_cols, time_col)
    by_source = _group_rows(pred, list(SOURCE_COLUMNS) + key_cols, time_col) if len(pred) else {}
    n_src = len(SOURCE_COLUMNS)

    src_idx: list[np.ndarray] = []
    keys_out: list[tuple[str, ...]] = []
    key_idx: list[np.ndarray] = []
    status: list[np.ndarray] = []
    pred_rows: list[np.ndarray] = []
    truth_rows: list[np.ndarray] = []
    no_rows = np.zeros(0, dtype=np.int64)
    pred_t = pred[time_col].to_numpy(float) if len(pred) else np.zeros(0)
    truth_t = truth[time_col].to_numpy(float) if len(truth) else np.zeros(0)

    for s, source in enumerate(sources):
        pred_groups = {k[n_src:]: v for k, v in by_source.items() if k[:n_src] == tuple(source)}
        for key in sorted(set(truth_groups) | set(pred_groups)):
            p_rows = pred_groups.get(key, no_rows)
            t_rows = truth_groups.get(key, no_rows)
            pi, ti = assign_within_tolerance(pred_t[p_rows], truth_t[t_rows], tolerance_s)
            fp = np.setdiff1d(np.arange(p_rows.size), pi)
            fn = np.setdiff1d(np.arange(t_rows.size), ti)
            n = pi.size + fp.size + fn.size
            if n == 0:
                continue
            keys_out.append(key)
            key_idx.append(np.full(n, len(keys_out) - 1))
            src_idx.append(np.full(n, s))
            status.append(np.repeat(np.array(["TP", "FP", "FN"]), [pi.size, fp.size, fn.size]))
            pred_rows.append(np.concatenate([p_rows[pi], p_rows[fp], np.full(fn.size, -1)]))
            truth_rows.append(np.concatenate([t_rows[ti], np.full(fp.size, -1), t_rows[fn]]))

    columns: dict[str, np.ndarray] = {}
    if not status:
        for col in list(SOURCE_COLUMNS) + key_cols + ["status"]:
            columns[col] = np.array([], dtype=object)
        columns["pred_row"] = no_rows
        columns["truth_row"] = no_rows
        return pd.DataFrame(columns).astype({c: str for c in list(SOURCE_COLUMNS) + key_cols + ["status"]})

    src_arr = np.array([tuple(s) for s in sources], dtype=object).reshape(len(sources), n_src)
    key_arr = np.array(keys_out, dtype=object).reshape(len(keys_out), len(key_cols))
    si, ki = np.concatenate(src_idx), np.concatenate(key_idx)
    for j, col in enumerate(SOURCE_COLUMNS):
        columns[col] = src_arr[si, j]
    for j, col in enumerate(key_cols):
        columns[col] = key_arr[ki, j]
    columns["status"] = np.concatenate(status)
    columns["pred_row"] = np.concatenate(pred_rows).astype(np.int64)
    columns["truth_row"] = np.concatenate(truth_rows).astype(np.int64)
    return pd.DataFrame(columns).astype({c: str for c in list(SOURCE_COLUMNS) + key_cols + ["status"]})


def take(values: pd.Series, rows: np.ndarray) -> np.ndarray:
    """``values[rows]`` as floats, NaN where ``rows`` is -1."""
    arr = values.to_numpy(dtype=float)
    out = np.full(rows.shape, np.nan)
    ok = rows >= 0
    out[ok] = arr[rows[ok]]
    return out


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _check_tolerance(tolerance_s: float) -> float:
    tolerance_s = float(tolerance_s)
    if not np.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError(f"tolerance_s must be a finite number >= 0, got {tolerance_s}")
    return tolerance_s


def _prepare_events(df: pd.DataFrame, what: str) -> pd.DataFrame:
    """Standardise an events table and insist on usable times."""
    out = validate_events(df)
    bad = ~np.isfinite(out["time_s"].to_numpy(float))
    if bad.any():
        raise ValueError(
            f"{what}: time_s is missing for {int(bad.sum())} event(s); matching is done on time. "
            "Fill it from the keypoints with gaitbench.formats.frames_to_times."
        )
    return out


def _sources(events: pd.DataFrame) -> list[tuple[str, ...]]:
    if events.empty:
        return [("",) * len(SOURCE_COLUMNS)]
    uniq = events.loc[:, list(SOURCE_COLUMNS)].drop_duplicates()
    return sorted(tuple(str(v) for v in row) for row in uniq.itertuples(index=False))


def _check_single_source(events: pd.DataFrame, what: str) -> None:
    n = events.loc[:, ["clip_id", *SOURCE_COLUMNS]].drop_duplicates().groupby("clip_id").size()
    multi = sorted(n.index[n > 1])
    if multi:
        raise ValueError(
            f"{what} has more than one source ({', '.join(SOURCE_COLUMNS)}) for clip(s) {multi}. "
            "Score against one rater at a time; compare raters with agreement()."
        )


def _group_rows(df: pd.DataFrame, cols: list[str], time_col: str) -> dict[tuple[str, ...], np.ndarray]:
    """Row positions of each group, in time order."""
    if df.empty:
        return {}
    order = np.argsort(df[time_col].to_numpy(float), kind="stable")
    ordered = df.iloc[order]
    groups = ordered.groupby(cols, sort=False, dropna=False).indices
    out: dict[tuple[str, ...], np.ndarray] = {}
    for key, pos in groups.items():
        key = key if isinstance(key, tuple) else (key,)
        out[tuple(str(k) for k in key)] = order[np.sort(pos)]
    return out


def _matches_frame(pairs: pd.DataFrame, pred: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    p_rows = pairs["pred_row"].to_numpy()
    t_rows = pairs["truth_row"].to_numpy()
    out = pd.DataFrame(
        {
            "clip_id": pairs["clip_id"],
            "detector": pairs["detector"],
            "backend": pairs["backend"],
            "side": pairs["side"],
            "event": pairs["event"],
            "status": pairs["status"],
            "truth_time_s": take(truth["time_s"], t_rows),
            "pred_time_s": take(pred["time_s"], p_rows),
            "truth_frame": take(truth["frame"], t_rows),
            "pred_frame": take(pred["frame"], p_rows),
            "tool": pairs["tool"],
            "tool_version": pairs["tool_version"],
        }
    )
    out["error_s"] = np.where(out["status"] == "TP", out["pred_time_s"] - out["truth_time_s"], np.nan)
    out["_t"] = out["truth_time_s"].fillna(out["pred_time_s"])
    out = out.sort_values(
        ["detector", "backend", "tool", "tool_version", "clip_id", "side", "event", "_t"], kind="stable"
    )
    return out.loc[:, list(MATCH_COLUMNS)].reset_index(drop=True)

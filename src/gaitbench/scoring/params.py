"""Temporal gait parameters from events, and their agreement with a reference.

A gait cycle (stride) of side S runs from one heel strike of S to the next.
In normal walking the events inside it come in a fixed order, shown here for
a left cycle (C is the contralateral side, R)::

    HS_L ── TO_R ── HS_R ── TO_L ── HS_L
    |initial DS|          |terminal DS|
    |------ stance of L --------|swing|
    |--- step ----|

* stride time = next HS_S - HS_S
* stance % = (TO_S - HS_S) / stride, swing % = 100 - stance %
* step time = first HS_C after HS_S - HS_S
* initial double support % = (TO_C - HS_S) / stride, terminal double support
  % = (TO_S - HS_C) / stride, double support % = their sum

A cycle is ``valid`` only if it holds exactly one TO_C, one HS_C and one TO_S
in that order. Anything else (a missed or doubled event, a swapped side)
makes it invalid, and ``reason`` says what was found. Stride time is always
reported; stance needs exactly one TO_S, step time a HS_C, and double support
a valid cycle. Downstream analyses should normally keep valid cycles only.

Events at exactly the cycle's start or end time are not counted as inside it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..formats import SIDES
from .match import SOURCE_COLUMNS, _check_single_source, _check_tolerance, _prepare_events, _sources, match_tables, take
from .stats import bland_altman, icc_2_1

PARAMETERS: tuple[str, ...] = (
    "stride_time_s",
    "stance_pct",
    "swing_pct",
    "step_time_s",
    "double_support_initial_pct",
    "double_support_terminal_pct",
    "double_support_pct",
)

PARAM_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "detector",
    "backend",
    "side",
    "cycle",
    "start_frame",
    "end_frame",
    "start_time_s",
    "end_time_s",
    "toe_off_time_s",
    *PARAMETERS,
    "valid",
    "reason",
    "tool",
    "tool_version",
)

CADENCE_COLUMNS: tuple[str, ...] = (
    "clip_id", "detector", "backend", "n_hs", "span_s", "cadence_spm", "tool", "tool_version",
)

_OTHER = {"L": "R", "R": "L"}
_EXPECTED_ORDER = ("TO_C", "HS_C", "TO_S")


def temporal_parameters(events: pd.DataFrame) -> pd.DataFrame:
    """Split events into gait cycles and compute temporal parameters per cycle.

    Parameters
    ----------
    events:
        Events or labels table (:data:`gaitbench.formats.EVENT_COLUMNS`) with
        ``time_s`` set, from any number of clips and sources. Each source
        (tool, tool_version, backend, detector) and clip is handled on its own.

    Returns
    -------
    pandas.DataFrame
        One row per cycle with columns :data:`PARAM_COLUMNS`: the source and
        clip, ``side``, ``cycle`` (0, 1, ... in time order per clip, source and
        side), ``start_frame`` / ``end_frame`` and ``start_time_s`` /
        ``end_time_s`` (the two heel strikes), ``toe_off_time_s`` (NaN unless
        exactly one TO of that side lies inside), the parameters in
        :data:`PARAMETERS` (times in s, the rest in % of stride time, NaN
        where not computable, see module docstring), ``valid`` (bool) and
        ``reason`` (empty when valid).
    """
    ev = _prepare_events(events, "events")
    rows: list[dict] = []
    if not ev.empty:
        for key, grp in ev.groupby(["clip_id", *SOURCE_COLUMNS], sort=True, dropna=False):
            info = dict(zip(["clip_id", *SOURCE_COLUMNS], key))
            times = {
                (side, kind): np.sort(grp.loc[(grp["side"] == side) & (grp["event"] == kind), "time_s"].to_numpy(float))
                for side in SIDES
                for kind in ("HS", "TO")
            }
            for side in SIDES:
                hs = grp.loc[(grp["side"] == side) & (grp["event"] == "HS")].sort_values("time_s", kind="stable")
                hs_t, hs_f = hs["time_s"].to_numpy(float), hs["frame"].to_numpy()
                for c in range(len(hs_t) - 1):
                    row = {**info, "side": side, "cycle": c, "start_frame": int(hs_f[c]), "end_frame": int(hs_f[c + 1])}
                    row.update(_cycle(hs_t[c], hs_t[c + 1], side, times))
                    rows.append(row)
    out = pd.DataFrame(rows, columns=list(PARAM_COLUMNS))
    if out.empty:
        out = out.astype({c: float for c in ("start_time_s", "end_time_s", "toe_off_time_s", *PARAMETERS)})
        out = out.astype({c: str for c in ("clip_id", "detector", "backend", "side", "reason", "tool", "tool_version")})
    out = out.astype({"cycle": np.int64, "start_frame": np.int64, "end_frame": np.int64, "valid": bool})
    return out.sort_values(["detector", "backend", "tool", "tool_version", "clip_id", "side", "cycle"], kind="stable").reset_index(drop=True)


def cadence_spm(events: pd.DataFrame) -> pd.DataFrame:
    """Cadence in steps per minute from the heel strikes of both sides.

    ``cadence = 60 * (n_hs - 1) / (last HS time - first HS time)``, i.e. 60
    over the mean interval between consecutive heel strikes of either side
    (the same definition as ``gaitkit.evaluation.compute_cadence_error``).
    Gaps from missed heel strikes and turns inside the clip lower it.

    Parameters
    ----------
    events:
        Events or labels table with ``time_s`` set.

    Returns
    -------
    pandas.DataFrame
        One row per clip and source with columns :data:`CADENCE_COLUMNS`;
        ``cadence_spm`` is NaN with fewer than two heel strikes.
    """
    ev = _prepare_events(events, "events")
    rows = []
    hs = ev.loc[ev["event"] == "HS"]
    for key, grp in hs.groupby(["clip_id", *SOURCE_COLUMNS], sort=True, dropna=False):
        t = np.sort(grp["time_s"].to_numpy(float))
        span = float(t[-1] - t[0])
        cadence = 60.0 * (t.size - 1) / span if t.size > 1 and span > 0 else float("nan")
        rows.append({**dict(zip(["clip_id", *SOURCE_COLUMNS], key)), "n_hs": int(t.size), "span_s": span, "cadence_spm": cadence})
    out = pd.DataFrame(rows, columns=list(CADENCE_COLUMNS))
    return out.astype({"n_hs": np.int64, "span_s": float, "cadence_spm": float})


def pair_cycles(pred_params: pd.DataFrame, truth_params: pd.DataFrame, tolerance_s: float = 0.1) -> pd.DataFrame:
    """Pair predicted cycles with reference cycles by their start (heel strike) time.

    Cycles are paired one to one within each clip and side, per prediction
    source, with the same optimal assignment as
    :func:`~gaitbench.scoring.match_events`.

    Parameters
    ----------
    pred_params:
        Output of :func:`temporal_parameters` for one or more sources.
    truth_params:
        Output of :func:`temporal_parameters` on the labels, one source per clip.
    tolerance_s:
        Largest start-time difference that can be paired.

    Returns
    -------
    pandas.DataFrame
        One row per paired (``TP``), prediction-only (``FP``) or
        reference-only (``FN``) cycle: the source columns, ``clip_id``,
        ``side``, ``status``, ``pred_cycle`` / ``truth_cycle`` (-1 if
        absent), ``start_error_s``, ``valid_pred`` / ``valid_truth`` and, for
        each parameter ``p`` in :data:`PARAMETERS`, ``p_pred`` and ``p_truth``.
    """
    tolerance_s = _check_tolerance(tolerance_s)
    pred = _prepare_params(pred_params, "pred_params")
    truth = _prepare_params(truth_params, "truth_params")
    _check_single_source(truth, "truth_params")
    sources = _sources(pred)
    pairs = match_tables(pred, truth, sources, key_cols=("clip_id", "side"), time_col="start_time_s", tolerance_s=tolerance_s)
    p_rows, t_rows = pairs["pred_row"].to_numpy(), pairs["truth_row"].to_numpy()

    out = pairs.loc[:, ["clip_id", "detector", "backend", "side", "status", "tool", "tool_version"]].copy()
    out["pred_cycle"] = _take_int(pred["cycle"], p_rows)
    out["truth_cycle"] = _take_int(truth["cycle"], t_rows)
    out["start_error_s"] = take(pred["start_time_s"], p_rows) - take(truth["start_time_s"], t_rows)
    out["valid_pred"] = _take_bool(pred["valid"], p_rows)
    out["valid_truth"] = _take_bool(truth["valid"], t_rows)
    for p in PARAMETERS:
        out[p + "_pred"] = take(pred[p], p_rows)
        out[p + "_truth"] = take(truth[p], t_rows)
    out["_t"] = take(truth["start_time_s"], t_rows)
    out["_t"] = out["_t"].fillna(pd.Series(take(pred["start_time_s"], p_rows), index=out.index))
    out = out.sort_values(["detector", "backend", "tool", "tool_version", "clip_id", "side", "_t"], kind="stable")
    return out.drop(columns="_t").reset_index(drop=True)


def compare_parameters(
    pred_params: pd.DataFrame,
    truth_params: pd.DataFrame,
    tolerance_s: float = 0.1,
    *,
    group_by: Sequence[str] | None = SOURCE_COLUMNS,
    parameters: Sequence[str] = PARAMETERS,
    valid_only: bool = True,
) -> pd.DataFrame:
    """Agreement of predicted temporal parameters with reference values.

    Cycles are paired by :func:`pair_cycles`. For each group and parameter,
    the statistics use the pairs in which both values are finite (and, with
    ``valid_only``, both cycles are valid). Differences are ``pred - truth``.

    Parameters
    ----------
    pred_params, truth_params, tolerance_s:
        As for :func:`pair_cycles`.
    group_by:
        Columns of the pairing table to group by: any of ``detector``,
        ``backend``, ``tool``, ``tool_version``, ``clip_id``, ``side``. The
        default, all four source columns, keeps apart the sources that
        :func:`pair_cycles` pairs separately; leaving some out pools those
        sources. Empty or None gives one row per parameter.
    parameters:
        Which of :data:`PARAMETERS` to report.
    valid_only:
        Use only pairs where both cycles are valid. Invalid cycles usually
        hide a missed or extra event, which detection metrics already count.

    Returns
    -------
    pandas.DataFrame
        One row per group and parameter: the ``group_by`` columns,
        ``parameter``, ``n_pred_cycles`` (TP + FP), ``n_truth_cycles``
        (TP + FN), ``n_pairs`` (pairs used), ``mean_pred``, ``mean_truth``,
        ``mean_diff`` (Bland-Altman bias), ``sd_diff``, ``mae``,
        ``loa_low``, ``loa_high`` and ``icc_2_1``, in the parameter's units
        (s or % of stride).
    """
    unknown = [p for p in parameters if p not in PARAMETERS]
    if unknown:
        raise ValueError(f"Unknown parameters {unknown}; choose from {PARAMETERS}")
    group_by = list(group_by or [])
    pairs = pair_cycles(pred_params, truth_params, tolerance_s=tolerance_s)
    bad = [c for c in group_by if c not in pairs.columns]
    if bad:
        raise ValueError(f"Cannot group by {bad}")

    groups = pairs.groupby(group_by, sort=True, dropna=False).indices if group_by else {(): np.arange(len(pairs))}
    if pairs.empty:
        groups = {}
    nan = float("nan")
    rows = []
    for key, pos in groups.items():
        key = key if isinstance(key, tuple) else (key,)
        g = pairs.iloc[pos]
        tp = g["status"].to_numpy() == "TP"
        if valid_only:
            tp &= g["valid_pred"].to_numpy(bool) & g["valid_truth"].to_numpy(bool)
        for p in parameters:
            a, b = g[p + "_pred"].to_numpy(float), g[p + "_truth"].to_numpy(float)
            use = tp & np.isfinite(a) & np.isfinite(b)
            a, b = a[use], b[use]
            ba = bland_altman(a, b)
            rows.append(
                {
                    **dict(zip(group_by, key)),
                    "parameter": p,
                    "n_pred_cycles": int((g["status"] != "FN").sum()),
                    "n_truth_cycles": int((g["status"] != "FP").sum()),
                    "n_pairs": int(use.sum()),
                    "mean_pred": float(a.mean()) if a.size else nan,
                    "mean_truth": float(b.mean()) if b.size else nan,
                    "mean_diff": ba["bias"],
                    "sd_diff": ba["sd"],
                    "mae": float(np.abs(a - b).mean()) if a.size else nan,
                    "loa_low": ba["loa_low"],
                    "loa_high": ba["loa_high"],
                    "icc_2_1": icc_2_1(a, b),
                }
            )
    columns = group_by + [
        "parameter", "n_pred_cycles", "n_truth_cycles", "n_pairs", "mean_pred", "mean_truth",
        "mean_diff", "sd_diff", "mae", "loa_low", "loa_high", "icc_2_1",
    ]
    out = pd.DataFrame(rows, columns=columns)
    return out.astype({"n_pred_cycles": np.int64, "n_truth_cycles": np.int64, "n_pairs": np.int64})


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _cycle(t0: float, t1: float, side: str, times: dict[tuple[str, str], np.ndarray]) -> dict:
    """Parameters of the cycle of ``side`` from heel strike ``t0`` to ``t1``."""
    other = _OTHER[side]
    nan = float("nan")
    stride = t1 - t0

    def inside(s: str, kind: str) -> np.ndarray:
        t = times[(s, kind)]
        return t[(t > t0) & (t < t1)]

    to_s, to_c, hs_c = inside(side, "TO"), inside(other, "TO"), inside(other, "HS")
    row = {
        "start_time_s": t0,
        "end_time_s": t1,
        "stride_time_s": stride,
        "toe_off_time_s": nan,
        **{p: nan for p in PARAMETERS[1:]},
    }

    problems = []
    if stride <= 0:
        problems.append("zero-length cycle (duplicate heel strike)")
    labels = {"TO_S": f"TO_{side}", "TO_C": f"TO_{other}", "HS_C": f"HS_{other}"}
    for name, found in (("TO_C", to_c), ("HS_C", hs_c), ("TO_S", to_s)):
        if found.size != 1:
            problems.append(f"{found.size} {labels[name]} (expected 1)")
    if not problems:
        order = sorted(((to_c[0], "TO_C"), (hs_c[0], "HS_C"), (to_s[0], "TO_S")))
        if tuple(name for _, name in order) != _EXPECTED_ORDER:
            seq = " -> ".join([f"HS_{side}", *(labels[name] for _, name in order), f"HS_{side}"])
            problems.append(f"out of order: {seq}")

    if stride > 0:
        if to_s.size == 1:
            row["toe_off_time_s"] = float(to_s[0])
            row["stance_pct"] = 100.0 * (to_s[0] - t0) / stride
            row["swing_pct"] = 100.0 - row["stance_pct"]
        if hs_c.size:
            row["step_time_s"] = float(hs_c[0] - t0)
        if not problems:
            row["double_support_initial_pct"] = 100.0 * (to_c[0] - t0) / stride
            row["double_support_terminal_pct"] = 100.0 * (to_s[0] - hs_c[0]) / stride
            row["double_support_pct"] = row["double_support_initial_pct"] + row["double_support_terminal_pct"]
    row["valid"] = not problems
    row["reason"] = "; ".join(problems)
    return row


def _prepare_params(df: pd.DataFrame, what: str) -> pd.DataFrame:
    missing = [c for c in ("clip_id", "side", "cycle", "start_time_s", "valid", *SOURCE_COLUMNS, *PARAMETERS) if c not in df.columns]
    if missing:
        raise ValueError(f"{what} is missing columns {missing}; build it with temporal_parameters()")
    out = df.reset_index(drop=True).copy()
    for col in ("clip_id", "side", *SOURCE_COLUMNS):
        out[col] = out[col].fillna("").astype(str)
    if not np.all(np.isfinite(out["start_time_s"].to_numpy(float))):
        raise ValueError(f"{what}: start_time_s must be set for every cycle")
    return out


def _take_int(values: pd.Series, rows: np.ndarray) -> np.ndarray:
    arr = values.to_numpy(dtype=np.int64)
    out = np.full(rows.shape, -1, dtype=np.int64)
    out[rows >= 0] = arr[rows[rows >= 0]]
    return out


def _take_bool(values: pd.Series, rows: np.ndarray) -> np.ndarray:
    arr = values.to_numpy(dtype=bool)
    out = np.zeros(rows.shape, dtype=bool)
    out[rows >= 0] = arr[rows[rows >= 0]]
    return out

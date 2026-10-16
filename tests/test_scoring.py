"""Tests for gaitbench.scoring: event matching, metrics, temporal parameters and statistics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gaitbench.formats import make_events
from gaitbench.scoring import (
    MATCH_COLUMNS,
    PARAM_COLUMNS,
    agreement,
    assign_within_tolerance,
    bland_altman,
    cadence_spm,
    compare_parameters,
    event_metrics,
    icc_2_1,
    icc_2_1_matrix,
    match_events,
    pair_cycles,
    temporal_parameters,
)
from gaitbench.synthetic import WalkerParams, simulate_walk


def events(rows, clip="c1", detector="det", backend="bk", tool="t", tool_version="1", fps=None):
    """Events from (side, event, time_s) or (side, event, time_s, frame) tuples."""
    recs = []
    for r in rows:
        side, ev, t = r[:3]
        frame = r[3] if len(r) > 3 else int(round(t * (fps or 100)))
        recs.append({"side": side, "event": ev, "time_s": t, "frame": frame})
    return make_events(
        recs,
        defaults={"clip_id": clip, "detector": detector, "backend": backend, "tool": tool, "tool_version": tool_version},
    )


def truth_events(rows, clip="c1"):
    return events(rows, clip=clip, detector="", backend="", tool="human", tool_version="raterA")


def statuses(m):
    return list(m["status"])


# ---------------------------------------------------------------------------
# match_events
# ---------------------------------------------------------------------------


def test_output_columns_and_exact_match():
    truth = truth_events([("L", "HS", 1.0), ("L", "TO", 1.6)])
    pred = events([("L", "HS", 1.02), ("L", "TO", 1.55)])
    m = match_events(pred, truth)
    assert tuple(m.columns) == MATCH_COLUMNS
    assert statuses(m) == ["TP", "TP"]
    assert m["error_s"].tolist() == pytest.approx([0.02, -0.05])
    assert m["detector"].tolist() == ["det", "det"]
    assert m["backend"].tolist() == ["bk", "bk"]


def test_two_predictions_near_one_truth():
    truth = truth_events([("L", "HS", 1.0)])
    pred = events([("L", "HS", 0.97), ("L", "HS", 1.05)])
    m = match_events(pred, truth)
    tp, fp = m[m.status == "TP"], m[m.status == "FP"]
    assert len(tp) == 1 and len(fp) == 1 and (m.status == "FN").sum() == 0
    assert tp["pred_time_s"].iloc[0] == pytest.approx(0.97)
    assert tp["error_s"].iloc[0] == pytest.approx(-0.03)
    assert fp["pred_time_s"].iloc[0] == pytest.approx(1.05)
    assert math.isnan(fp["truth_time_s"].iloc[0]) and math.isnan(fp["error_s"].iloc[0])


def test_prediction_beyond_tolerance_is_fp_and_fn():
    truth = truth_events([("R", "TO", 2.0)])
    pred = events([("R", "TO", 2.15)])
    m = match_events(pred, truth, tolerance_s=0.1, restrict_to_truth_span=False)
    assert sorted(statuses(m)) == ["FN", "FP"]
    assert m["error_s"].isna().all()
    # With the default span restriction the far prediction is outside the labelled span.
    assert statuses(match_events(pred, truth, tolerance_s=0.1)) == ["FN"]
    # The same pair is a TP with a wider tolerance.
    m2 = match_events(pred, truth, tolerance_s=0.2)
    assert statuses(m2) == ["TP"]


def test_tolerance_is_inclusive_at_frame_resolution():
    # 6 frames at 60 fps is exactly 0.1 s; frame / fps rounding must not reject it.
    truth = truth_events([("L", "HS", 60 / 60, 60)])
    pred = events([("L", "HS", 66 / 60, 66)])
    assert statuses(match_events(pred, truth, tolerance_s=0.1)) == ["TP"]
    pred7 = events([("L", "HS", 67 / 60, 67)])
    m7 = match_events(pred7, truth, tolerance_s=0.1, restrict_to_truth_span=False)
    assert sorted(statuses(m7)) == ["FN", "FP"]


def test_optimal_matching_beats_greedy():
    # Greedy nearest-first gives the prediction at frame 16 the truth at 20
    # (4 frames) and leaves 24 with nothing within 6 frames. The optimal
    # assignment pairs 16-11 and 24-20: two TPs.
    truth = truth_events([("L", "HS", 11 / 60, 11), ("L", "HS", 20 / 60, 20)])
    pred = events([("L", "HS", 16 / 60, 16), ("L", "HS", 24 / 60, 24)])
    m = match_events(pred, truth, tolerance_s=0.1)
    assert statuses(m) == ["TP", "TP"]
    assert m["truth_frame"].tolist() == [11, 20]
    assert m["pred_frame"].tolist() == [16, 24]
    assert m["error_s"].tolist() == pytest.approx([5 / 60, 4 / 60])


def test_assignment_prefers_more_pairs_then_less_error():
    p, t = assign_within_tolerance(np.array([1.0, 1.08]), np.array([1.05, 1.15]), 0.1)
    assert sorted(zip(p.tolist(), t.tolist())) == [(0, 0), (1, 1)]
    p, t = assign_within_tolerance(np.array([]), np.array([1.0]), 0.1)
    assert p.size == 0 and t.size == 0


def test_sides_and_event_types_never_cross():
    truth = truth_events([("L", "HS", 1.0), ("R", "TO", 1.5)])
    pred = events([("R", "HS", 1.0), ("L", "TO", 1.5)])
    m = match_events(pred, truth)
    assert (m.status == "TP").sum() == 0
    assert (m.status == "FP").sum() == 2 and (m.status == "FN").sum() == 2


def test_events_outside_truth_span_are_ignored():
    truth = truth_events([("L", "HS", 1.0), ("R", "TO", 1.5), ("L", "HS", 2.0)])
    pred = events([("L", "HS", 0.5), ("L", "HS", 0.92), ("R", "TO", 1.5), ("L", "HS", 2.08), ("L", "HS", 2.5)])
    m = match_events(pred, truth, tolerance_s=0.1, restrict_to_truth_span=True)
    assert statuses(m) == ["TP", "TP", "TP"]
    assert m["error_s"].tolist() == pytest.approx([-0.08, 0.08, 0.0])
    # Inside the span (any side/event sets it), an unmatched prediction still counts.
    pred_in = pd.concat([pred, events([("R", "HS", 1.2)])])
    m_in = match_events(pred_in, truth)
    assert (m_in.status == "FP").sum() == 1
    # Without the restriction the predictions at 0.5 s and 2.5 s are false positives.
    m_all = match_events(pred, truth, restrict_to_truth_span=False)
    fp = m_all[m_all.status == "FP"]
    assert fp["pred_time_s"].tolist() == pytest.approx([0.5, 2.5])


def test_clip_without_truth_is_skipped_only_when_restricted():
    truth = truth_events([("L", "HS", 1.0)], clip="c1")
    pred = pd.concat([events([("L", "HS", 1.0)], clip="c1"), events([("L", "HS", 1.0)], clip="c2")])
    m = match_events(pred, truth)
    assert m["clip_id"].tolist() == ["c1"] and statuses(m) == ["TP"]
    m_all = match_events(pred, truth, restrict_to_truth_span=False)
    assert m_all.loc[m_all.clip_id == "c2", "status"].tolist() == ["FP"]


def test_empty_pred_gives_all_fn():
    truth = truth_events([("L", "HS", 1.0), ("R", "HS", 1.5), ("L", "TO", 1.6)])
    m = match_events(make_events([]), truth)
    assert statuses(m) == ["FN", "FN", "FN"]
    assert m["pred_time_s"].isna().all() and m["error_s"].isna().all()
    assert set(m["detector"]) == {""}
    assert m["truth_frame"].tolist() == [100, 160, 150]


def test_empty_truth():
    pred = events([("L", "HS", 1.0), ("R", "TO", 1.4)])
    assert match_events(pred, make_events([])).empty
    m = match_events(pred, make_events([]), restrict_to_truth_span=False)
    assert statuses(m) == ["FP", "FP"]
    both = match_events(make_events([]), make_events([]))
    assert both.empty and tuple(both.columns) == MATCH_COLUMNS


def test_multiple_detectors_are_matched_separately():
    truth = pd.concat(
        [
            truth_events([("L", "HS", 1.0), ("L", "HS", 2.0), ("R", "HS", 1.5)], clip="c1"),
            truth_events([("L", "HS", 1.0)], clip="c2"),
        ]
    )
    good = pd.concat(
        [
            events([("L", "HS", 1.0), ("L", "HS", 2.01), ("R", "HS", 1.49)], clip="c1", detector="good"),
            events([("L", "HS", 1.0)], clip="c2", detector="good"),
        ]
    )
    # 'bad' misses the right heel strike, adds a spurious one and was never run on c2.
    bad = events([("L", "HS", 1.03), ("L", "HS", 1.6), ("L", "HS", 2.0)], clip="c1", detector="bad")
    m = match_events(pd.concat([good, bad]), truth)
    g, b = m[m.detector == "good"], m[m.detector == "bad"]
    assert statuses(g) == ["TP"] * 4
    assert (b.status == "TP").sum() == 2
    assert (b.status == "FP").sum() == 1
    assert b.loc[b.status == "FN", "clip_id"].tolist() == ["c1", "c2"]


def test_same_detector_different_versions_or_backends_scored_apart():
    truth = truth_events([("L", "HS", 1.0)])
    v1 = events([("L", "HS", 1.01)], tool_version="0.9.0")
    v2 = events([("L", "HS", 1.02)], tool_version="0.9.1")
    other_backend = events([("L", "HS", 0.99)], backend="rtmw")
    m = match_events(pd.concat([v1, v2, other_backend]), truth)
    assert statuses(m) == ["TP", "TP", "TP"]
    assert sorted(m["error_s"].round(6).tolist()) == [-0.01, 0.01, 0.02]


def test_frames_are_passed_through_as_source_video_indices():
    truth = truth_events([("L", "HS", 10.0, 1200)])
    pred = events([("L", "HS", 10.02, 1201)])
    m = match_events(pred, truth)
    assert m["truth_frame"].iloc[0] == 1200 and m["pred_frame"].iloc[0] == 1201


def test_truth_with_two_sources_is_rejected():
    a = truth_events([("L", "HS", 1.0)])
    b = events([("L", "HS", 1.0)], detector="", backend="", tool="human", tool_version="raterB")
    with pytest.raises(ValueError, match="more than one source"):
        match_events(events([("L", "HS", 1.0)]), pd.concat([a, b]))


def test_missing_time_is_rejected():
    pred = make_events([{"side": "L", "event": "HS", "frame": 10}], defaults={"clip_id": "c1"})
    with pytest.raises(ValueError, match="time_s"):
        match_events(pred, truth_events([("L", "HS", 0.1)]))
    with pytest.raises(ValueError):
        match_events(pred, truth_events([("L", "HS", 0.1)]), tolerance_s=-1)


def test_agreement_between_detectors():
    a = events([("L", "HS", 1.0), ("L", "HS", 2.0), ("R", "TO", 1.6)], detector="myogait:zeni")
    b = events([("L", "HS", 1.01), ("R", "TO", 1.65), ("R", "TO", 2.6)], detector="zeni")
    m = agreement(a, b, tolerance_s=0.1)
    assert set(MATCH_COLUMNS) <= set(m.columns)
    assert set(m["ref_detector"]) == {"zeni"}
    assert sorted(statuses(m)) == ["FN", "FP", "TP", "TP"]
    assert m.loc[m.status == "TP", "error_s"].tolist() == pytest.approx([-0.01, -0.05])
    # A clip only in a counts against it rather than being dropped.
    m2 = agreement(pd.concat([a, events([("L", "HS", 1.0)], clip="c9", detector="myogait:zeni")]), b)
    assert m2.loc[m2.clip_id == "c9", "status"].tolist() == ["FP"]


# ---------------------------------------------------------------------------
# event_metrics
# ---------------------------------------------------------------------------


def known_case():
    """Five heel strikes; errors +10, -20, +30, +40 ms; one miss; one extra."""
    truth = truth_events([("L", "HS", t) for t in (1.0, 2.0, 3.0, 4.0, 5.0)])
    pred = events([("L", "HS", t) for t in (1.01, 1.98, 3.03, 4.04, 4.5)])
    return match_events(pred, truth, tolerance_s=0.1)


@pytest.mark.filterwarnings("error")
def test_metrics_on_known_case():
    out = event_metrics(known_case(), fps=50)
    assert len(out) == 1
    r = out.iloc[0]
    assert (r.n_truth, r.n_pred, r.tp, r.fp, r.fn) == (5, 5, 4, 1, 1)
    assert r.recall == pytest.approx(0.8) and r.precision == pytest.approx(0.8) and r.f1 == pytest.approx(0.8)
    assert r.bias_ms == pytest.approx(15.0)
    sd = math.sqrt(2100 / 3)  # deviations -5, -35, 15, 25 ms
    assert r.sd_ms == pytest.approx(sd)
    assert r.mae_ms == pytest.approx(25.0)
    assert r.median_abs_ms == pytest.approx(25.0)
    assert r.loa_low_ms == pytest.approx(15 - 1.96 * sd)
    assert r.loa_high_ms == pytest.approx(15 + 1.96 * sd)
    # At 50 fps a frame is 20 ms: 10 and 20 ms are within one frame, all four within two; the miss is not.
    assert r.mae_frames == pytest.approx(1.25)
    assert r.p_within_1_frame == pytest.approx(2 / 5)
    assert r.p_within_2_frames == pytest.approx(4 / 5)


def test_metrics_without_fps_and_overall_row():
    out = event_metrics(known_case())
    assert "p_within_1_frame" not in out.columns
    overall = event_metrics(known_case(), group_by=None)
    assert len(overall) == 1 and overall["tp"].iloc[0] == 4


def test_metrics_fps_per_clip():
    truth = pd.concat([truth_events([("L", "HS", 1.0)], clip="a"), truth_events([("L", "HS", 1.0)], clip="b")])
    pred = pd.concat([events([("L", "HS", 1.01)], clip="a"), events([("L", "HS", 1.01)], clip="b")])
    m = match_events(pred, truth)
    # 10 ms is within one frame at 60 fps (16.7 ms) but not at 240 fps (4.2 ms).
    out = event_metrics(m, fps={"a": 60.0, "b": 240.0}, group_by=("clip_id",))
    assert out.set_index("clip_id")["p_within_1_frame"].to_dict() == {"a": 1.0, "b": 0.0}
    with pytest.raises(ValueError, match="No fps"):
        event_metrics(m, fps={"a": 60.0})


@pytest.mark.filterwarnings("error")
def test_metrics_degenerate_groups_without_warnings():
    truth = truth_events([("L", "HS", 1.0), ("L", "TO", 1.6), ("R", "HS", 1.5)])
    pred = events([("L", "HS", 1.02), ("R", "TO", 1.7)])
    m = match_events(pred, truth)
    out = event_metrics(m, fps=60, group_by=("side", "event")).set_index(["side", "event"])
    single = out.loc[("L", "HS")]  # one TP
    assert single.tp == 1 and single.bias_ms == pytest.approx(20.0)
    assert math.isnan(single.sd_ms) and math.isnan(single.loa_low_ms) and math.isnan(single.loa_high_ms)
    only_fn = out.loc[("L", "TO")]
    assert only_fn.recall == 0 and math.isnan(only_fn.precision) and only_fn.f1 == 0
    assert math.isnan(only_fn.bias_ms) and math.isnan(only_fn.mae_ms) and only_fn.p_within_1_frame == 0
    only_fp = out.loc[("R", "TO")]
    assert math.isnan(only_fp.recall) and only_fp.precision == 0 and math.isnan(only_fp.p_within_2_frames)
    empty = event_metrics(match_events(make_events([]), make_events([])), fps=60)
    assert empty.empty and "f1" in empty.columns


def test_metrics_cross_check_with_gaitkit():
    gk = pytest.importorskip("gaitkit.evaluation")
    fps = 60.0
    truth_f = [12, 75, 140, 203, 268, 331]
    # Detections where greedy and optimal matching agree: a few frames off,
    # one miss (203), one extra far from any truth (300), one out of range (155 vs 140 = 15 frames).
    det_f = [14, 73, 155, 270, 300, 333]
    ref = gk.compute_event_metrics(det_f, truth_f, tolerance_ms=100, fps=fps)
    truth = truth_events([("L", "HS", f / fps, f) for f in truth_f])
    pred = events([("L", "HS", f / fps, f) for f in det_f])
    ours = event_metrics(match_events(pred, truth, restrict_to_truth_span=False), fps=fps).iloc[0]
    assert (ours.tp, ours.fp, ours.fn) == (ref["n_tp"], ref["n_fp"], ref["n_fn"])
    assert ours.recall == pytest.approx(ref["recall"])
    assert ours.precision == pytest.approx(ref["precision"])
    assert ours.f1 == pytest.approx(ref["f1"])
    assert ours.mae_ms == pytest.approx(ref["mae_ms"])
    assert ours.mae_frames == pytest.approx(ref["mae_frames"])

    # Where greedy matching loses a pair (see test_optimal_matching_beats_greedy).
    ref2 = gk.compute_event_metrics([16, 24], [11, 20], tolerance_ms=100, fps=fps)
    assert ref2["n_tp"] == 1 and ref2["recall"] == 0.5
    m2 = match_events(
        events([("L", "HS", f / fps, f) for f in (16, 24)]), truth_events([("L", "HS", f / fps, f) for f in (11, 20)])
    )
    assert event_metrics(m2).iloc[0].recall == 1.0


# ---------------------------------------------------------------------------
# temporal_parameters and cadence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"stance_frac": 0.65, "stride_time_s": 1.2, "fps": 100.0, "duration_s": 6.0},
        {"direction": -1, "pan_speed_mps": 1.3, "stance_frac": 0.58},
    ],
)
def test_temporal_parameters_on_synthetic_truth(overrides):
    walk = simulate_walk(**overrides)
    p: WalkerParams = walk.params
    out = temporal_parameters(walk.truth)
    assert tuple(out.columns) == PARAM_COLUMNS
    assert len(out) >= 6 and out["valid"].all() and (out["reason"] == "").all()
    assert set(out["side"]) == {"L", "R"}
    T, s = p.stride_time_s, p.stance_frac
    assert out["stride_time_s"].to_numpy() == pytest.approx(T)
    assert out["stance_pct"].to_numpy() == pytest.approx(100 * s)
    assert out["swing_pct"].to_numpy() == pytest.approx(100 * (1 - s))
    # Left and right are half a cycle apart.
    assert out["step_time_s"].to_numpy() == pytest.approx(T / 2)
    assert out["double_support_initial_pct"].to_numpy() == pytest.approx(100 * (s - 0.5))
    assert out["double_support_terminal_pct"].to_numpy() == pytest.approx(100 * (s - 0.5))
    assert out["double_support_pct"].to_numpy() == pytest.approx(2 * (s - 0.5) * 100)
    for side, grp in out.groupby("side"):
        assert grp["cycle"].tolist() == list(range(len(grp)))
        assert np.allclose(grp["end_time_s"].to_numpy()[:-1], grp["start_time_s"].to_numpy()[1:])


def test_default_synthetic_walk_values():
    out = temporal_parameters(simulate_walk().truth)
    assert len(out) == 8
    first = out[(out.side == "L") & (out.cycle == 0)].iloc[0]
    assert first.start_time_s == pytest.approx(0.2625) and first.end_time_s == pytest.approx(1.3125)
    assert first.toe_off_time_s == pytest.approx(0.8925)
    assert (first.start_frame, first.end_frame) == (16, 79)
    assert first.stride_time_s == pytest.approx(1.05) and first.stance_pct == pytest.approx(60.0)
    assert first.double_support_pct == pytest.approx(20.0)


def test_missing_toe_off_makes_cycles_invalid():
    truth = simulate_walk().truth
    # Drop the right toe off at 1.4175 s: inside left cycle 1 (1.3125-2.3625) and right cycle 0 (0.7875-1.8375).
    drop = (truth.side == "R") & (truth.event == "TO") & np.isclose(truth.time_s, 1.4175)
    assert drop.sum() == 1
    out = temporal_parameters(truth[~drop]).set_index(["side", "cycle"])
    left = out.loc[("L", 1)]
    assert not left.valid and left.reason == "0 TO_R (expected 1)"
    assert left.stance_pct == pytest.approx(60.0) and left.step_time_s == pytest.approx(0.525)
    assert math.isnan(left.double_support_pct)
    right = out.loc[("R", 0)]
    assert not right.valid and right.reason == "0 TO_R (expected 1)"
    assert math.isnan(right.stance_pct) and math.isnan(right.toe_off_time_s)
    assert right.stride_time_s == pytest.approx(1.05)
    assert out["valid"].sum() == len(out) - 2


def test_out_of_order_and_missed_heel_strike():
    ev = events(
        [
            ("L", "HS", 0.0), ("R", "HS", 0.1), ("R", "TO", 0.2), ("L", "TO", 0.6), ("L", "HS", 1.0),
        ]
    )
    row = temporal_parameters(ev).query("side == 'L'").iloc[0]
    assert not row.valid
    assert row.reason == "out of order: HS_L -> HS_R -> TO_R -> TO_L -> HS_L"
    assert row.stance_pct == pytest.approx(60.0) and row.step_time_s == pytest.approx(0.1)
    assert math.isnan(row.double_support_initial_pct)

    # A missed left heel strike makes one two-stride cycle.
    truth = simulate_walk().truth
    drop = (truth.side == "L") & (truth.event == "HS") & np.isclose(truth.time_s, 1.3125)
    out = temporal_parameters(truth[~drop])
    long = out[(out.side == "L") & (out.cycle == 0)].iloc[0]
    assert long.stride_time_s == pytest.approx(2.1) and not long.valid
    assert "2 TO_L" in long.reason and "2 TO_R" in long.reason and "2 HS_R" in long.reason
    assert math.isnan(long.stance_pct)


def test_temporal_parameters_per_source_and_empty():
    truth = simulate_walk().truth
    a = truth.assign(detector="a")
    b = truth.assign(detector="b")
    out = temporal_parameters(pd.concat([a, b]))
    assert out.groupby("detector").size().to_dict() == {"a": 8, "b": 8}
    assert out["valid"].all()
    empty = temporal_parameters(make_events([]))
    assert empty.empty and tuple(empty.columns) == PARAM_COLUMNS
    one_side = temporal_parameters(truth[truth.side == "L"])
    assert len(one_side) == 4 and not one_side["valid"].any()
    assert one_side["stance_pct"].to_numpy() == pytest.approx(60.0)
    assert one_side["step_time_s"].isna().all()


def test_cadence():
    walk = simulate_walk()
    out = cadence_spm(walk.truth)
    assert len(out) == 1
    # A heel strike every half stride.
    assert out["cadence_spm"].iloc[0] == pytest.approx(60.0 / (walk.params.stride_time_s / 2))
    assert out["n_hs"].iloc[0] == 10
    single = cadence_spm(events([("L", "HS", 1.0), ("L", "TO", 1.5)]))
    assert single["n_hs"].iloc[0] == 1 and math.isnan(single["cadence_spm"].iloc[0])


# ---------------------------------------------------------------------------
# compare_parameters
# ---------------------------------------------------------------------------


def three_walkers():
    walks = [simulate_walk(stride_time_s=T, clip_id=f"w{i}") for i, T in enumerate((0.95, 1.05, 1.15))]
    return pd.concat([w.truth for w in walks], ignore_index=True)


def test_compare_parameters_with_late_heel_strikes():
    truth = three_walkers()
    shift = 0.01
    late = truth.assign(detector="late")
    late.loc[late.event == "HS", "time_s"] += shift
    truth_p = temporal_parameters(truth)
    pred_p = temporal_parameters(late)
    assert pred_p["valid"].all()

    out = compare_parameters(pred_p, truth_p, tolerance_s=0.1).set_index("parameter")
    assert (out["n_pairs"] == len(truth_p)).all()
    assert (out["n_truth_cycles"] == len(truth_p)).all() and (out["n_pred_cycles"] == len(pred_p)).all()

    stride = out.loc["stride_time_s"]
    assert stride.mean_diff == pytest.approx(0.0, abs=1e-12) and stride.mae == pytest.approx(0.0, abs=1e-12)
    assert stride.icc_2_1 == pytest.approx(1.0)
    assert out.loc["step_time_s"].mean_diff == pytest.approx(0.0, abs=1e-12)
    # Stance and both double-support phases shrink by shift / stride time.
    expected = -(100 * shift / truth_p["stride_time_s"]).mean()
    assert out.loc["stance_pct"].mean_diff == pytest.approx(expected)
    assert out.loc["stance_pct"].mae == pytest.approx(-expected)
    assert out.loc["swing_pct"].mean_diff == pytest.approx(-expected)
    assert out.loc["double_support_initial_pct"].mean_diff == pytest.approx(expected)
    assert out.loc["double_support_pct"].mean_diff == pytest.approx(2 * expected)
    ba = bland_altman(pred_p["stance_pct"], truth_p["stance_pct"])
    assert out.loc["stance_pct"].loa_low == pytest.approx(ba["loa_low"])
    assert out.loc["stance_pct"].sd_diff == pytest.approx(ba["sd"])


def test_compare_parameters_unpaired_and_invalid_cycles():
    truth = three_walkers()
    truth_p = temporal_parameters(truth)
    # The detector misses clip w2 entirely and one right toe off in w0.
    pred = truth[truth.clip_id != "w2"].assign(detector="d")
    drop = (pred.clip_id == "w0") & (pred.side == "R") & (pred.event == "TO")
    pred = pred.drop(index=pred.index[drop][1])
    pred_p = temporal_parameters(pred)
    n_invalid = int((~pred_p["valid"]).sum())
    assert n_invalid == 2

    pairs = pair_cycles(pred_p, truth_p)
    assert (pairs.status == "FN").sum() == (truth_p.clip_id == "w2").sum()
    assert (pairs.status == "FP").sum() == 0
    assert pairs.loc[pairs.status == "TP", "start_error_s"].abs().max() < 1e-12

    out = compare_parameters(pred_p, truth_p).set_index("parameter")
    n_tp = int((pairs.status == "TP").sum())
    assert out.loc["stride_time_s"].n_pairs == n_tp - n_invalid
    assert out.loc["stride_time_s"].n_truth_cycles == len(truth_p)
    assert out.loc["stride_time_s"].n_pred_cycles == len(pred_p)
    loose = compare_parameters(pred_p, truth_p, valid_only=False).set_index("parameter")
    assert loose.loc["stride_time_s"].n_pairs == n_tp
    # Stance is undefined in the right cycle with no toe off.
    assert loose.loc["stance_pct"].n_pairs == n_tp - 1
    by_side = compare_parameters(pred_p, truth_p, group_by=("detector", "side"), parameters=["stride_time_s"])
    assert sorted(by_side["side"]) == ["L", "R"]
    with pytest.raises(ValueError):
        compare_parameters(pred_p, truth_p, parameters=["cadence"])


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

# Shrout & Fleiss (1979), Table 2: 6 targets rated by 4 judges.
SHROUT_FLEISS = np.array(
    [
        [9, 2, 5, 8],
        [6, 1, 3, 2],
        [8, 4, 6, 8],
        [7, 1, 2, 6],
        [10, 5, 6, 9],
        [6, 2, 4, 7],
    ],
    dtype=float,
)


def test_icc_shrout_fleiss_table():
    # Published ICC(2,1) for all four judges: 0.29.
    assert round(icc_2_1_matrix(SHROUT_FLEISS), 2) == 0.29


def test_icc_two_judges_by_hand():
    # Judges 1 and 2 of the same table, worked by hand (n = 6, k = 2):
    #   grand mean 61/12; SS_total = 417 - 61^2/12 = 1283/12
    #   SS_rows = 2 * sum((row mean - grand)^2) = 281/12  -> MS_R = 281/60
    #   SS_cols = 6 * 2 * (31/12)^2             = 961/12  -> MS_C = 961/12
    #   SS_err  = 1283/12 - 281/12 - 961/12     = 41/12   -> MS_E = 41/60
    #   ICC = (MS_R - MS_E) / (MS_R + MS_E + 2 (MS_C - MS_E) / 6) = 24/191
    assert icc_2_1(SHROUT_FLEISS[:, 0], SHROUT_FLEISS[:, 1]) == pytest.approx(24 / 191)


def test_icc_identical_offset_and_degenerate():
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert icc_2_1(b, b) == pytest.approx(1.0)
    # A constant offset c lowers absolute agreement: 2 var(b) / (2 var(b) + c^2) = 5 / 6.
    assert icc_2_1(b + 1.0, b) == pytest.approx(5 / 6)
    assert math.isnan(icc_2_1([1.0], [1.0]))
    assert math.isnan(icc_2_1([2.0, 2.0, 2.0], [2.0, 2.0, 2.0]))
    # Rounding noise on a constant (stride times from event differences) is not variance.
    assert math.isnan(icc_2_1([1.05, 1.05 + 2e-16, 1.05], [1.05, 1.05, 1.05 - 2e-16]))
    assert icc_2_1([1.0, 2.0, np.nan, 3.0], [1.0, 2.0, 5.0, 3.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        icc_2_1([1.0, 2.0], [1.0])


def test_bland_altman():
    ba = bland_altman([1.0, 2.0, 3.0, 4.0], [0.5, 2.5, 2.0, 4.0])
    # Differences 0.5, -0.5, 1, 0: mean 0.25, sample variance 1.25 / 3.
    sd = math.sqrt(1.25 / 3)
    assert ba["n"] == 4
    assert ba["bias"] == pytest.approx(0.25)
    assert ba["sd"] == pytest.approx(sd)
    assert ba["loa_low"] == pytest.approx(0.25 - 1.96 * sd)
    assert ba["loa_high"] == pytest.approx(0.25 + 1.96 * sd)
    one = bland_altman([1.0, np.nan], [0.5, 1.0])
    assert one["n"] == 1 and one["bias"] == pytest.approx(0.5) and math.isnan(one["sd"])
    none = bland_altman([], [])
    assert none["n"] == 0 and math.isnan(none["bias"])


# ---------------------------------------------------------------------------
# end to end with the reference detector, when it is available
# ---------------------------------------------------------------------------


def test_zeni_on_clean_synthetic_panning_walk():
    detectors = pytest.importorskip("gaitbench.detectors")
    walk = simulate_walk(pan_speed_mps=1.3)
    pred = detectors.run_detector("zeni", walk.keypoints)
    m = match_events(pred, walk.truth)
    out = event_metrics(m, fps=walk.params.fps)
    assert (out["precision"] == 1.0).all()
    # An event on the first or last frame cannot be a peak, so allow edge misses only.
    missed = m.loc[m.status == "FN", "truth_frame"]
    n = walk.keypoints.n_frames
    assert missed.isin([0, 1, n - 2, n - 1]).all()
    tp_err = m.loc[m.status == "TP", "error_s"].abs()
    assert (tp_err <= 1.0 / walk.params.fps + 1e-9).all()

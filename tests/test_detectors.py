"""Tests for gaitbench.detectors (Zeni and Stenum rules, helpers and registry).

Accuracy is checked against the synthetic walker, which has exact event
times. The matcher here is deliberately local and simple so these tests do
not depend on the scoring module.
"""

from __future__ import annotations

import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest

from gaitbench import __version__
from gaitbench.detectors import (
    detect_stenum,
    detect_zeni,
    estimate_direction,
    get_detector,
    list_detectors,
    run_detector,
)
from gaitbench.detectors import base
from gaitbench.formats import EVENT_COLUMNS, Keypoints, make_events, validate_events
from gaitbench.synthetic import simulate_walk

DETECTORS = ("zeni", "stenum")
SPEED = 1.3


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def visible_truth(walk) -> pd.DataFrame:
    """Truth events that happen strictly inside the sampled clip.

    An event in the first or last frame interval (or after the last sample,
    which the simulator can produce: frame 299 at t = 4.9875 s in a 300-frame
    60 fps clip) has no data on one side, so no peak-based rule can see it.
    """
    kp = walk.keypoints
    dt = 1.0 / kp.fps
    keep = (walk.truth["time_s"] >= kp.time_s[0] + dt) & (walk.truth["time_s"] <= kp.time_s[-1] - dt)
    return walk.truth[keep].reset_index(drop=True)


def match(pred: pd.DataFrame, truth: pd.DataFrame, fps: float, tol_frames: float = 2.0) -> dict:
    """Greedy one-to-one matching per side and event type, closest pairs first.

    Returns the signed errors (pred - truth, in frames) of matched pairs and
    the unmatched truth and predicted rows.
    """
    errors, missed, extra = [], [], []
    for (side, event), t_grp in truth.groupby(["side", "event"]):
        p_grp = pred[(pred["side"] == side) & (pred["event"] == event)]
        t_times, p_times = t_grp["time_s"].to_numpy(), p_grp["time_s"].to_numpy()
        pairs = sorted(
            (abs(pt - tt) * fps, i, j) for i, tt in enumerate(t_times) for j, pt in enumerate(p_times)
        )
        used_t, used_p = set(), set()
        for dist, i, j in pairs:
            if dist <= tol_frames and i not in used_t and j not in used_p:
                used_t.add(i)
                used_p.add(j)
                errors.append((p_times[j] - t_times[i]) * fps)
        missed += [(side, event, t_times[i]) for i in range(len(t_times)) if i not in used_t]
        extra += [(side, event, p_times[j]) for j in range(len(p_times)) if j not in used_p]
    matched_sides = set(zip(truth["side"], truth["event"]))
    extra += [
        (r.side, r.event, r.time_s) for r in pred.itertuples() if (r.side, r.event) not in matched_sides
    ]
    return {"errors": np.array(errors), "missed": missed, "extra": extra, "n_truth": len(truth)}


def side_swaps(pred: pd.DataFrame, truth: pd.DataFrame) -> list:
    """Predicted events whose nearest truth event of the same type is on the other side."""
    swaps = []
    for row in pred.itertuples():
        same_type = truth[truth["event"] == row.event]
        nearest = same_type.iloc[int(np.argmin(np.abs(same_type["time_s"].to_numpy() - row.time_s)))]
        if nearest["side"] != row.side:
            swaps.append((row.side, row.event, row.frame))
    return swaps


def with_offset(kp: Keypoints, frames: int, seconds: float) -> Keypoints:
    return Keypoints(
        clip_id=kp.clip_id,
        names=kp.names,
        xy=kp.xy,
        conf=kp.conf,
        time_s=kp.time_s + seconds,
        frame_idx=kp.frame_idx + frames,
        meta=kp.meta,
    )


def without(kp: Keypoints, parts: tuple[str, ...]) -> Keypoints:
    """Keypoints with every LEFT_/RIGHT_ version of ``parts`` (and bare names) removed."""
    drop = {p for part in parts for p in (part, "LEFT_" + part, "RIGHT_" + part)}
    return kp.subset([n for n in kp.names if n not in drop])


# ---------------------------------------------------------------------------
# accuracy on the synthetic walker
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize("pan", [0.0, SPEED], ids=["fixed", "panning"])
@pytest.mark.parametrize("direction", [1, -1], ids=["ltr", "rtl"])
@pytest.mark.parametrize("fps", [60, 30])
def test_clean_walk_every_event_found(detector, pan, direction, fps):
    walk = simulate_walk(fps=fps, direction=direction, speed_mps=SPEED, pan_speed_mps=pan)
    kp = walk.keypoints
    events = run_detector(detector, kp)
    truth = visible_truth(walk)

    result = match(events, truth, fps, tol_frames=2.0)
    assert result["missed"] == []
    assert result["extra"] == []
    assert np.abs(result["errors"]).max() <= 2.0
    assert side_swaps(events, walk.truth) == []
    assert events["frame"].between(kp.frame_idx[0], kp.frame_idx[-1]).all()
    assert events["time_s"].between(kp.time_s[0], kp.time_s[-1]).all()


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize("fps", [60, 30])
def test_events_near_clip_edges_are_found(detector, fps):
    # Shift the gait phase so events land in the first and last few frames.
    for phase in np.linspace(0.0, 1.0, 21):
        walk = simulate_walk(fps=fps, right_phase0=float(phase))
        result = match(run_detector(detector, walk.keypoints), visible_truth(walk), fps)
        assert result["missed"] == [], f"phase {phase:.2f}"
        assert result["extra"] == [], f"phase {phase:.2f}"


@pytest.mark.parametrize("fps", [60, 30])
@pytest.mark.parametrize("direction", [1, -1])
def test_zeni_is_pan_invariant(fps, direction):
    fixed = run_detector("zeni", simulate_walk(fps=fps, direction=direction, pan_speed_mps=0.0).keypoints)
    panning = run_detector("zeni", simulate_walk(fps=fps, direction=direction, pan_speed_mps=SPEED).keypoints)
    assert len(fixed) == len(panning)
    assert (fixed[["side", "event"]].to_numpy() == panning[["side", "event"]].to_numpy()).all()
    assert np.abs(fixed["frame"].to_numpy() - panning["frame"].to_numpy()).max() <= 1
    assert np.abs(fixed["time_s"].to_numpy() - panning["time_s"].to_numpy()).max() * fps <= 1.0


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize("fps", [60, 30])
def test_noise_and_dropout(detector, fps):
    n_ok = n_truth = n_extra = n_pred = 0
    for seed in range(5):
        for direction in (1, -1):
            for pan in (0.0, SPEED):
                walk = simulate_walk(
                    fps=fps, direction=direction, pan_speed_mps=pan, noise_px=2.0, dropout_prob=0.03, seed=seed
                )
                events = run_detector(detector, walk.keypoints)
                result = match(events, visible_truth(walk), fps, tol_frames=2.0)
                n_ok += len(result["errors"])
                n_truth += result["n_truth"]
                n_extra += len(result["extra"])
                n_pred += len(events)
    assert n_ok / n_truth >= 0.95
    assert n_extra / n_pred <= 0.05


def test_low_frame_rate_and_high_cutoff_are_clamped():
    walk = simulate_walk(fps=25)
    events = run_detector("zeni", walk.keypoints, cutoff_hz=20.0)  # above the 12.5 Hz Nyquist
    result = match(events, visible_truth(walk), 25)
    assert result["missed"] == []
    signals = base.foot_signals(walk.keypoints, "heel", direction=1, cutoff_hz=20.0)
    assert signals["L"].cutoff_hz == pytest.approx(0.9 * 12.5)


def test_refine_gives_sub_frame_times():
    walk = simulate_walk(fps=30)
    truth = visible_truth(walk)
    coarse = match(run_detector("zeni", walk.keypoints, refine=False), truth, 30)
    fine = match(run_detector("zeni", walk.keypoints, refine=True), truth, 30)
    assert np.abs(fine["errors"]).mean() < np.abs(coarse["errors"]).mean()
    # Unrefined events sit exactly on stored frames.
    raw = run_detector("zeni", walk.keypoints, refine=False)
    assert np.isin(raw["time_s"], walk.keypoints.time_s).all()


# ---------------------------------------------------------------------------
# frames, times and missing points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("detector", DETECTORS)
def test_frame_idx_offset_shifts_frames_and_times(detector):
    kp = simulate_walk(fps=60, pan_speed_mps=SPEED).keypoints
    plain = run_detector(detector, kp)
    shifted = run_detector(detector, with_offset(kp, frames=1000, seconds=10.0))
    assert len(plain) == len(shifted) > 0
    assert (shifted["frame"].to_numpy() - plain["frame"].to_numpy() == 1000).all()
    np.testing.assert_allclose(shifted["time_s"].to_numpy() - plain["time_s"].to_numpy(), 10.0, atol=1e-9)


def test_keypoints_from_every_second_frame_report_source_frames():
    # 120 fps video, pose run on every second frame: arrays are sampled at 60 Hz
    # but meta['fps'] is still the video's 120 fps.
    walk = simulate_walk(fps=120)
    kp = walk.keypoints
    half = Keypoints(
        clip_id=kp.clip_id,
        names=kp.names,
        xy=kp.xy[::2],
        conf=kp.conf[::2],
        time_s=kp.time_s[::2],
        frame_idx=kp.frame_idx[::2],
        meta=kp.meta,
    )
    assert base.sample_rate(half) == pytest.approx(60.0)
    events = run_detector("zeni", half)
    truth = visible_truth(walk)
    result = match(events, truth, fps=120, tol_frames=4.0)  # 4 source frames = 2 sampled frames
    assert result["missed"] == [] and result["extra"] == []
    # Frames are in source-video (120 fps) units, so they match the truth frames.
    merged = pd.merge_asof(
        events.sort_values("time_s"), truth.sort_values("time_s"), on="time_s", by=["side", "event"],
        direction="nearest", suffixes=("", "_truth"),
    )
    assert np.abs(merged["frame"] - merged["frame_truth"]).max() <= 2


def test_missing_heel_falls_back_to_ankle():
    walk = simulate_walk(fps=60, pan_speed_mps=SPEED)
    kp = without(walk.keypoints, ("HEEL",))
    assert base.choose_foot_point(kp, "L", "heel") == "LEFT_ANKLE"
    signals = base.foot_signals(kp, "heel", direction=1)
    assert {s.point for s in signals.values()} == {"LEFT_ANKLE", "RIGHT_ANKLE"}
    result = match(run_detector("zeni", kp), visible_truth(walk), 60)
    assert result["missed"] == [] and result["extra"] == []


def test_missing_toes_fall_back_to_ankle():
    walk = simulate_walk(fps=60)
    kp = without(walk.keypoints, ("FOOT_INDEX",))
    assert base.choose_foot_point(kp, "R", "toe") == "RIGHT_BIG_TOE"
    kp = without(kp, ("BIG_TOE",))
    assert base.choose_foot_point(kp, "R", "toe") == "RIGHT_ANKLE"
    result = match(run_detector("zeni", kp), visible_truth(walk), 60)
    assert result["missed"] == []


def test_zero_confidence_heel_is_skipped():
    # Some backends export a heel slot that is always empty.
    kp = simulate_walk().keypoints
    conf = kp.conf.copy()
    conf[:, kp.index("LEFT_HEEL")] = 0.0
    kp = Keypoints(kp.clip_id, kp.names, kp.xy, conf, kp.time_s, kp.frame_idx, kp.meta)
    assert base.choose_foot_point(kp, "L", "heel") == "LEFT_ANKLE"
    assert base.choose_foot_point(kp, "R", "heel") == "RIGHT_HEEL"


def test_one_hip_is_enough_for_the_pelvis():
    kp = simulate_walk().keypoints
    both = base.pelvis_x(kp)
    left_only = base.pelvis_x(without(kp, ("RIGHT_HIP",)))
    np.testing.assert_allclose(left_only, both)
    with pytest.raises(ValueError, match="HIP"):
        base.pelvis_x(without(kp, ("HIP",)))


def test_long_gap_splits_signal_and_short_gap_is_filled():
    walk = simulate_walk(fps=60, pan_speed_mps=SPEED)
    kp = walk.keypoints
    xy = kp.xy.copy()
    conf = kp.conf.copy()
    j = kp.index("LEFT_HEEL")
    xy[100:103, j] = np.nan  # 3 frames: filled
    conf[100:103, j] = 0.0
    xy[200:230, j] = np.nan  # 0.5 s: a break
    conf[200:230, j] = 0.0
    gappy = Keypoints(kp.clip_id, kp.names, xy, conf, kp.time_s, kp.frame_idx, kp.meta)
    sig = base.foot_signals(gappy, "heel", direction=1)["L"]
    assert np.isfinite(sig.raw[100:103]).all()
    assert np.isnan(sig.raw[200:230]).all()
    assert [(s.start, s.stop) for s in sig.segments] == [(0, 200), (230, kp.n_frames)]
    events = run_detector("zeni", gappy)
    left_hs = events[(events["side"] == "L") & (events["event"] == "HS")]
    assert not left_hs["frame"].between(200, 229).any()


# ---------------------------------------------------------------------------
# walking direction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pan", [0.0, SPEED, 2 * SPEED], ids=["fixed", "panning", "overtaking"])
@pytest.mark.parametrize("direction", [1, -1])
def test_estimate_direction(direction, pan):
    kp = simulate_walk(direction=direction, pan_speed_mps=pan).keypoints
    est, info = estimate_direction(kp)
    assert est == direction
    assert info["method"] == "foot"


@pytest.mark.parametrize("direction", [1, -1])
def test_estimate_direction_fallbacks(direction):
    kp = simulate_walk(direction=direction, pan_speed_mps=SPEED).keypoints
    no_feet = without(kp, ("FOOT_INDEX", "BIG_TOE", "SMALL_TOE"))
    # With toes gone the ankle-heel offset still points forward.
    est, info = estimate_direction(no_feet)
    assert (est, info["method"]) == (direction, "foot")

    no_feet = without(no_feet, ("HEEL",))
    est, info = estimate_direction(no_feet)
    assert (est, info["method"]) == (direction, "nose")

    # Displacement only works when the walker actually crosses the image.
    fixed = simulate_walk(direction=direction).keypoints
    fixed = without(fixed, ("FOOT_INDEX", "BIG_TOE", "SMALL_TOE", "HEEL", "NOSE"))
    est, info = estimate_direction(fixed)
    assert (est, info["method"]) == (direction, "displacement")


def test_estimate_direction_default_warns():
    kp = simulate_walk(pan_speed_mps=SPEED).keypoints
    hips_only = kp.subset(["LEFT_HIP", "RIGHT_HIP"])
    with pytest.warns(UserWarning, match="direction unclear"):
        est, info = estimate_direction(hips_only)
    assert (est, info["method"]) == (1, "default")


@pytest.mark.parametrize("detector", DETECTORS)
def test_direction_override(detector):
    kp = simulate_walk(direction=-1).keypoints
    auto = run_detector(detector, kp)
    given = run_detector(detector, kp, direction=-1)
    pd.testing.assert_frame_equal(auto, given)
    with pytest.raises(ValueError, match="direction"):
        run_detector(detector, kp, direction=0)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def test_fill_gaps_rules():
    t = np.arange(10) / 10.0
    x = np.array([np.nan, 1, 2, np.nan, np.nan, 5, np.nan, np.nan, np.nan, 9.0])
    out = base.fill_gaps(x, t, max_gap_frames=2)
    assert np.isnan(out[0])  # leading gap is not extrapolated
    np.testing.assert_allclose(out[3:5], [3, 4])  # 2-frame gap filled
    assert np.isnan(out[6:9]).all()  # 3-frame gap left as a break
    assert np.isnan(x[3])  # input untouched


def test_fill_gaps_uses_timestamps():
    t = np.array([0.0, 0.01, 0.03, 0.04])  # uneven spacing, as in variable-frame-rate video
    out = base.fill_gaps(np.array([0.0, np.nan, np.nan, 4.0]), t, max_gap_frames=2)
    np.testing.assert_allclose(out, [0.0, 1.0, 3.0, 4.0])


def test_finite_segments_and_short_segments_dropped():
    x = np.r_[np.ones(10), np.nan, np.ones(3), np.nan, np.ones(6)]
    assert [(s.start, s.stop) for s in base.finite_segments(x, 5)] == [(0, 10), (15, 21)]


def test_lowpass_short_segment_and_zero_phase():
    fps = 30.0
    t = np.arange(12) / fps
    y = np.cos(2 * np.pi * 1.0 * (t - t[6]))  # peak at sample 6
    out = base.lowpass(y, fps, 6.0)
    assert out.shape == y.shape
    assert int(np.argmax(out)) == 6


def test_parabolic_offset_is_exact_for_a_parabola():
    vertex = 0.3
    y = [-(x - vertex) ** 2 for x in (-1.0, 0.0, 1.0)]
    assert base.parabolic_offset(*y) == pytest.approx(vertex)
    assert base.parabolic_offset(1.0, 1.0, 1.0) == 0.0


def test_positions_to_frames_maps_through_frame_idx():
    kp = simulate_walk(duration_s=1.0).keypoints
    kp = with_offset(kp, frames=50, seconds=2.0)
    frames, times = base.positions_to_frames(kp, np.array([0.0, 10.4, 20.6]))
    assert frames.tolist() == [50, 60, 71]
    np.testing.assert_allclose(times, 2.0 + np.array([0.0, 10.4, 20.6]) / 60.0)


def test_edge_aware_drop_matches_prominence_inside_and_relaxes_at_edges():
    from scipy.signal import find_peaks, peak_prominences

    rng = np.random.default_rng(3)
    y = np.cumsum(rng.normal(size=300))
    peaks, _ = find_peaks(y)
    prominences = peak_prominences(y, peaks)[0]
    drops = np.array([base._edge_aware_drop(y, p) for p in peaks])
    assert (drops >= prominences - 1e-12).all()
    # A peak 2 samples from the end: the edge side counts 1/EDGE_PROMINENCE_FRACTION times.
    y = np.array([0.0, 5.0, 10.0, 9.0, 8.0])
    assert base._edge_aware_drop(y, 2) == pytest.approx(2.0 / base.EDGE_PROMINENCE_FRACTION)


def test_alternation_drops_weaker_duplicate_only_when_data_is_continuous():
    n = 100
    sig = base.FootSignal("L", "LEFT_HEEL", np.zeros(n), np.zeros(n), np.ones(n), [slice(0, n)])
    hs1, hs2 = base.Extremum(10.0, 10, 0.3), base.Extremum(30.0, 30, 0.8)
    to = base.Extremum(50.0, 50, 0.5)
    kept = base.alternate_events([("HS", hs1), ("HS", hs2), ("TO", to)], sig, sig)
    assert kept == [("HS", hs2), ("TO", to)]

    gappy = base.FootSignal("L", "LEFT_TOE", np.zeros(n), np.zeros(n), np.ones(n), [slice(0, 15), slice(25, n)])
    gappy.smooth[15:25] = np.nan
    kept = base.alternate_events([("HS", hs1), ("HS", hs2), ("TO", to)], sig, gappy)
    assert kept == [("HS", hs1), ("HS", hs2), ("TO", to)]


def test_short_clip_gives_empty_valid_table():
    kp = simulate_walk(duration_s=2 / 60).keypoints  # 2 frames
    events = run_detector("zeni", kp)
    assert list(events.columns) == list(EVENT_COLUMNS)
    assert events.empty


def test_confidence_is_keypoint_confidence():
    walk = simulate_walk(near_side="L", near_side_conf=0.95, far_side_conf=0.85)
    events = run_detector("zeni", walk.keypoints)
    assert np.allclose(events.loc[events["side"] == "L", "confidence"], 0.95)
    assert np.allclose(events.loc[events["side"] == "R", "confidence"], 0.85)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_list_detectors_has_builtins():
    names = list_detectors()
    assert names[:2] == ["zeni", "stenum"]


def test_unknown_detector_raises_keyerror_listing_names():
    with pytest.raises(KeyError, match="zeni"):
        get_detector("no-such-detector")


def test_unknown_parameter_is_rejected_early():
    with pytest.raises(TypeError, match="cutoff"):
        get_detector("zeni", cutof_hz=5.0)


@pytest.mark.parametrize("detector", DETECTORS)
def test_run_detector_output_is_valid_and_stamped(detector):
    kp = simulate_walk(clip_id="walk01").keypoints
    events = run_detector(detector, kp)
    pd.testing.assert_frame_equal(events, validate_events(events))
    assert (events["detector"] == detector).all()
    assert (events["clip_id"] == "walk01").all()
    assert (events["backend"] == "synthetic").all()
    assert (events["tool"] == "gaitbench").all()
    assert (events["tool_version"] == __version__).all()
    pd.testing.assert_frame_equal(get_detector(detector)(kp), events)


def test_direct_calls_match_registry():
    kp = simulate_walk().keypoints
    pd.testing.assert_frame_equal(detect_zeni(kp), run_detector("zeni", kp))
    pd.testing.assert_frame_equal(detect_stenum(kp), run_detector("stenum", kp))


def test_missing_backend_meta_gives_empty_backend():
    kp = simulate_walk().keypoints
    meta = {k: v for k, v in kp.meta.items() if k != "backend"}
    kp = Keypoints(kp.clip_id, kp.names, kp.xy, kp.conf, kp.time_s, kp.frame_idx, meta)
    assert (run_detector("zeni", kp)["backend"] == "").all()


@pytest.fixture
def fake_bridge(monkeypatch):
    """A stand-in gaitbench.myogait_bridge, so delegation is tested without myogait."""
    calls = []

    def detect_events_myogait(kp, method, **params):
        calls.append((method, params))
        return make_events(
            [{"side": "L", "event": "HS", "frame": int(kp.frame_idx[5]), "time_s": float(kp.time_s[5])}],
            defaults={"tool": "myogait", "tool_version": "9.9.9", "confidence": 1.0},
        )

    module = types.ModuleType("gaitbench.myogait_bridge")
    module.list_methods = lambda: ["zeni", "consensus"]
    module.detect_events_myogait = detect_events_myogait
    monkeypatch.setitem(sys.modules, "gaitbench.myogait_bridge", module)
    import gaitbench.detectors as registry

    monkeypatch.setattr(registry, "_myogait_available", lambda: True)
    return calls


def test_myogait_names_are_delegated_to_the_bridge(fake_bridge):
    assert "myogait:zeni" in list_detectors() and "myogait:consensus" in list_detectors()
    kp = simulate_walk(clip_id="walk02").keypoints
    events = run_detector("myogait:consensus", kp, butterworth_cutoff=5.0)
    assert fake_bridge == [("consensus", {"butterworth_cutoff": 5.0})]
    assert events.loc[0, "detector"] == "myogait:consensus"
    assert events.loc[0, "clip_id"] == "walk02"
    assert events.loc[0, "backend"] == "synthetic"
    assert (events.loc[0, "tool"], events.loc[0, "tool_version"]) == ("myogait", "9.9.9")
    with pytest.raises(KeyError, match="myogait:zeni"):
        get_detector("myogait:nope")


def test_myogait_names_absent_without_myogait(monkeypatch):
    import gaitbench.detectors as registry

    monkeypatch.setattr(registry, "_myogait_available", lambda: False)
    assert list_detectors() == ["zeni", "stenum"]
    with pytest.raises(ImportError, match="myogait"):
        get_detector("myogait:zeni")


def test_real_bridge_methods_are_listed():
    bridge = pytest.importorskip("gaitbench.myogait_bridge")
    pytest.importorskip("myogait")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        names = list_detectors()
    assert [n for n in names if n.startswith("myogait:")] == ["myogait:" + m for m in bridge.list_methods()]

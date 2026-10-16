"""Tests for gaitbench.clean (leg-swap correction).

Swaps are injected into the synthetic walker, whose labels are exact, so the
correction can be checked frame by frame and our Zeni detector can be scored
against the walker's truth. The real GPJATK OpenPose sample (skipped if it
has not been fetched) has known swaps near t = 1.4, 2.5, 2.7 and 3.1 s.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gaitbench import __version__
from gaitbench.clean import (
    LEG_PARTS,
    REPORT_COLUMNS,
    clean_keypoints,
    clean_keypoints_file,
    cleaned_backend_name,
    fix_leg_swaps,
)
from gaitbench.cli import decimate_keypoints
from gaitbench.detectors import run_detector
from gaitbench.detectors.base import estimate_direction, foot_signals
from gaitbench.formats import Keypoints
from gaitbench.synthetic import simulate_walk

CHAIN = ("KNEE", "ANKLE", "HEEL", "FOOT_INDEX", "BIG_TOE", "SMALL_TOE")
GPJATK = Path(__file__).resolve().parents[1] / "data" / "gpjatk_sample" / "keypoints"
GPJATK_LEFT = GPJATK / "gpjatk_p01s01_sag_left__openpose.npz"
GPJATK_RIGHT = GPJATK / "gpjatk_p01s01_sag_right__openpose.npz"
# Known swap or doubled-leg moments in the left-camera clip (from its signal plot).
GPJATK_SWAP_TIMES = (1.4, 2.5, 2.7, 3.1)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def with_arrays(kp: Keypoints, **changes) -> Keypoints:
    fields = dict(clip_id=kp.clip_id, names=kp.names, xy=kp.xy, conf=kp.conf, time_s=kp.time_s,
                  frame_idx=kp.frame_idx, meta=kp.meta)
    fields.update(changes)
    return Keypoints(**fields)


def swap_labels(kp: Keypoints, frames, parts=LEG_PARTS) -> Keypoints:
    """What a pose model does: LEFT_ and RIGHT_ leg points (xy and conf) exchanged on ``frames``."""
    rows = np.asarray(sorted(set(int(f) for f in frames)), dtype=int)
    xy, conf = kp.xy.copy(), kp.conf.copy()
    for part in parts:
        if kp.has("LEFT_" + part) and kp.has("RIGHT_" + part):
            li, ri = kp.index("LEFT_" + part), kp.index("RIGHT_" + part)
            xy[rows, li], xy[rows, ri] = kp.xy[rows, ri], kp.xy[rows, li]
            conf[rows, li], conf[rows, ri] = kp.conf[rows, ri], kp.conf[rows, li]
    return with_arrays(kp, xy=xy, conf=conf)


def leg_columns(kp: Keypoints) -> list[int]:
    return [kp.index(s + p) for p in LEG_PARTS for s in ("LEFT_", "RIGHT_") if kp.has(s + p)]


def frames_matching(out: Keypoints, original: Keypoints) -> np.ndarray:
    """Per frame: are all leg points (xy and conf) back to the original labelling?"""
    cols = leg_columns(original)
    xy_same = [np.array_equal(out.xy[i, cols], original.xy[i, cols], equal_nan=True) for i in range(out.n_frames)]
    conf_same = np.all(out.conf[:, cols] == original.conf[:, cols], axis=1)
    return np.asarray(xy_same) & conf_same


def chain_separation(kp: Keypoints) -> np.ndarray:
    """Median distance between matching left and right chain points, per frame."""
    parts = [p for p in CHAIN if kp.has("LEFT_" + p)]
    d = np.stack([np.linalg.norm(kp.point("LEFT_" + p) - kp.point("RIGHT_" + p), axis=1) for p in parts], axis=1)
    return np.nanmedian(d, axis=1)


def crossings(kp: Keypoints) -> list[int]:
    """Frames where the legs pass each other (local minima of the chain separation)."""
    sep = chain_separation(kp)
    return [i for i in range(1, kp.n_frames - 1) if sep[i] <= sep[i - 1] and sep[i] < sep[i + 1] and sep[i] < 40]


def choose_swaps(n: int, rng: np.random.Generator, n_single: int = 6, run: int = 4) -> list[int]:
    """Frame 0, one ``run``-frame stretch mid-clip and ``n_single`` isolated frames."""
    frames = {0}
    start = int(rng.integers(n // 4, 3 * n // 4))
    frames |= set(range(start, start + run))
    while len(frames) < 1 + run + n_single:
        f = int(rng.integers(1, n - 1))
        if all(abs(f - g) > 3 for g in frames):
            frames.add(f)
    return sorted(frames)


def tolerance_frames(fps: float) -> float:
    # 2 frames, except at 240 fps where 2 frames (8 ms) is below Zeni's own
    # scatter on 2 px noise even without any swap; there 20 ms is used.
    return max(2.0, 0.02 * fps)


def match(pred: pd.DataFrame, truth: pd.DataFrame, fps: float, tol_frames: float) -> tuple[list, list]:
    """Greedy one-to-one matching per side and event type; returns (missed truth, extra predictions)."""
    missed, extra = [], []
    for side in ("L", "R"):
        for event in ("HS", "TO"):
            t_times = truth[(truth.side == side) & (truth.event == event)]["time_s"].to_numpy()
            p_times = pred[(pred.side == side) & (pred.event == event)]["time_s"].to_numpy()
            pairs = sorted((abs(p - t) * fps, i, j) for i, t in enumerate(t_times) for j, p in enumerate(p_times))
            used_t, used_p = set(), set()
            for dist, i, j in pairs:
                if dist <= tol_frames and i not in used_t and j not in used_p:
                    used_t.add(i)
                    used_p.add(j)
            missed += [(side, event, t_times[i]) for i in range(len(t_times)) if i not in used_t]
            extra += [(side, event, p_times[j]) for j in range(len(p_times)) if j not in used_p]
    return missed, extra


def inner_truth(walk, margin_s: float = 0.1) -> pd.DataFrame:
    """Truth events with at least ``margin_s`` of data on each side (a peak needs both flanks)."""
    t = walk.keypoints.time_s
    keep = (walk.truth.time_s >= t[0] + margin_s) & (walk.truth.time_s <= t[-1] - margin_s)
    return walk.truth[keep].reset_index(drop=True)


def zeni_errors(kp: Keypoints, walk) -> tuple[list, list]:
    """(missed inner truth events, predictions matching no truth event) for our Zeni on ``kp``."""
    fps = walk.params.fps
    pred = run_detector("zeni", kp)
    missed, _ = match(pred, inner_truth(walk), fps, tolerance_frames(fps))
    _, extra = match(pred, walk.truth, fps, tolerance_frames(fps))
    return missed, extra


def assert_corrected(out: Keypoints, original: Keypoints, swapped, *, min_restored=0.95, max_changed=0.01):
    same = frames_matching(out, original)
    sw = np.zeros(original.n_frames, dtype=bool)
    sw[list(swapped)] = True
    restored = same[sw].mean()
    changed = (~same[~sw]).mean()
    assert restored >= min_restored, f"restored {restored:.1%}; not restored: {np.flatnonzero(sw & ~same).tolist()}"
    assert changed <= max_changed, f"changed {changed:.1%} of clean frames: {np.flatnonzero(~sw & ~same).tolist()}"


# ---------------------------------------------------------------------------
# synthetic walker: injected swaps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("noise", [0.0, 2.0], ids=["exact", "noise2px"])
@pytest.mark.parametrize("pan", [0.0, 1.3], ids=["fixed", "panning"])
@pytest.mark.parametrize("fps", [25, 60, 240])
def test_swaps_are_restored_and_zeni_recovers(fps, pan, noise):
    walk = simulate_walk(fps=fps, pan_speed_mps=pan, noise_px=noise, seed=fps)
    kp = walk.keypoints
    frames = choose_swaps(kp.n_frames, np.random.default_rng(fps + int(10 * pan) + int(noise)))
    bad = swap_labels(kp, frames)

    out, report = fix_leg_swaps(bad)

    assert_corrected(out, kp, frames)
    assert set(report.loc[report.action == "swap", "frame"]) == set(frames)
    assert (report.dropped == "").all()
    # The swaps matter to the detector, and after cleaning it is as good as on the clean walk.
    raw_missed, raw_extra = zeni_errors(bad, walk)
    assert raw_missed or raw_extra
    missed, extra = zeni_errors(out, walk)
    assert missed == [] and extra == []


@pytest.mark.parametrize("fps", [25, 60, 240])
def test_zeni_events_identical_to_unswapped_clip(fps):
    walk = simulate_walk(fps=fps, pan_speed_mps=1.3, noise_px=2.0, direction=-1, seed=3)
    kp = walk.keypoints
    frames = choose_swaps(kp.n_frames, np.random.default_rng(99))
    out, _ = fix_leg_swaps(swap_labels(kp, frames))
    cols = ["side", "event", "frame", "time_s"]
    pd.testing.assert_frame_equal(run_detector("zeni", out)[cols], run_detector("zeni", kp)[cols])


@pytest.mark.parametrize("fps", [25, 60, 240])
@pytest.mark.parametrize("where", ["start", "end"])
def test_swapped_run_at_clip_edge(fps, where):
    walk = simulate_walk(fps=fps, noise_px=2.0, seed=1)
    kp = walk.keypoints
    k = max(4, int(0.05 * fps))
    frames = range(k) if where == "start" else range(kp.n_frames - k, kp.n_frames)
    out, report = fix_leg_swaps(swap_labels(kp, frames))
    assert_corrected(out, kp, frames, min_restored=1.0, max_changed=0.0)


@pytest.mark.parametrize("noise", [0.0, 2.0], ids=["exact", "noise2px"])
@pytest.mark.parametrize("fps", [25, 60, 240])
def test_swap_from_one_crossing_to_the_next(fps, noise):
    """Half a stride with the labels exchanged, starting and ending where the legs cross."""
    walk = simulate_walk(fps=fps, noise_px=noise, pan_speed_mps=1.3, seed=2)
    kp = walk.keypoints
    cross = crossings(simulate_walk(fps=fps, pan_speed_mps=1.3).keypoints)
    for start, stop in ((cross[1], cross[2]), (cross[1] + 1, cross[2] + 1)):
        out, _ = fix_leg_swaps(swap_labels(kp, range(start, stop)))
        assert_corrected(out, kp, range(start, stop))


@pytest.mark.parametrize("fps", [25, 60, 240])
def test_longer_swaps_mid_clip(fps):
    walk = simulate_walk(fps=fps, noise_px=2.0, seed=4)
    kp = walk.keypoints
    n = kp.n_frames
    frames = list(range(n // 3, n // 3 + int(0.1 * fps))) + list(range(n // 2, n // 2 + int(0.3 * fps)))
    out, _ = fix_leg_swaps(swap_labels(kp, frames))
    assert_corrected(out, kp, frames)
    assert zeni_errors(out, walk) == ([], [])


@pytest.mark.parametrize("names", ["all", "mediapipe", "openpose"])
@pytest.mark.parametrize("noise", [0.0, 2.0, 4.0])
@pytest.mark.parametrize("fps", [25, 60, 240])
def test_clean_walk_is_left_alone(fps, noise, names):
    """Genuine crossings (and noise) never trigger a swap or a removal."""
    kp = simulate_walk(fps=fps, noise_px=noise, pan_speed_mps=1.3 if noise else 0.0, seed=7).keypoints
    if names == "mediapipe":
        kp = kp.subset([n for n in kp.names if not n.endswith(("BIG_TOE", "SMALL_TOE"))])
    elif names == "openpose":
        kp = kp.subset([n for n in kp.names if not n.endswith("FOOT_INDEX")])
    out, report = fix_leg_swaps(kp)
    assert np.array_equal(out.xy, kp.xy, equal_nan=True)
    assert np.array_equal(out.conf, kp.conf)
    assert (report.action == "keep").all() and (report.dropped == "").all()
    assert out.meta["cleaned"]["leg_swaps"]["n_swapped"] == 0


@pytest.mark.parametrize("direction", [1, -1], ids=["ltr", "rtl"])
def test_walker_entering_and_leaving_the_image(direction):
    """A fixed camera: legs are partly outside the image at the start and end."""
    walk = simulate_walk(fps=25, duration_s=8.0, direction=direction, seed=5)
    kp = walk.keypoints
    full = np.flatnonzero(np.isfinite(kp.xy[:, leg_columns(kp)]).all(axis=(1, 2)))
    frames = sorted(np.random.default_rng(5).choice(full, 8, replace=False).tolist())
    out, _ = fix_leg_swaps(swap_labels(kp, frames))
    assert_corrected(out, kp, frames, max_changed=0.0)
    clean_out, _ = fix_leg_swaps(kp)
    assert np.array_equal(clean_out.xy, kp.xy, equal_nan=True)


@pytest.mark.parametrize("fps", [25, 60, 240])
def test_doubled_leg_is_removed(fps):
    """Both labels on one leg: the copy is removed, the true leg and the other frames are kept."""
    walk = simulate_walk(fps=fps, noise_px=2.0, seed=6)
    kp = walk.keypoints
    rng = np.random.default_rng(fps)
    apart = np.flatnonzero(chain_separation(kp) > 80)
    apart = apart[(apart > 10) & (apart < kp.n_frames - 20)]
    starts = sorted(rng.choice(apart, 4, replace=False))
    run = max(1, int(0.02 * fps))
    xy = kp.xy.copy()
    copies = {}  # frame -> side that holds the copy
    for j, start in enumerate(starts):
        src, dst = ("RIGHT_", "LEFT_") if j % 2 else ("LEFT_", "RIGHT_")
        rows = list(range(start, start + (run if j % 2 else 1)))
        for part in CHAIN:
            xy[rows, kp.index(dst + part)] = kp.xy[rows, kp.index(src + part)] + rng.normal(0, 1, (len(rows), 2))
        copies.update({r: dst[0] for r in rows})
    doubled = with_arrays(kp, xy=xy)

    out, report = fix_leg_swaps(doubled)

    dropped = report.loc[report.dropped != "", ["frame", "dropped"]]
    assert dict(zip(dropped.frame, dropped.dropped)) == copies
    for frame, side in copies.items():
        other = "RIGHT_" if side == "L" else "LEFT_"
        prefix = "LEFT_" if side == "L" else "RIGHT_"
        assert np.isnan(out.point(prefix + "HEEL")[frame]).all() and out.confidence(prefix + "HEEL")[frame] == 0
        assert np.array_equal(out.point(other + "HEEL")[frame], kp.point(other + "HEEL")[frame])
        # Hips are never removed.
        assert np.array_equal(out.point(prefix + "HIP")[frame], kp.point(prefix + "HIP")[frame])
    others = np.setdiff1d(np.arange(kp.n_frames), list(copies))
    assert frames_matching(out, kp)[others].all()
    assert zeni_errors(out, walk) == ([], [])


@pytest.mark.parametrize("fps", [25, 60, 240])
def test_doubled_leg_next_to_a_swap(fps):
    """As on the GPJATK sample: one doubled frame, then a swapped frame."""
    walk = simulate_walk(fps=fps, noise_px=2.0, seed=8)
    kp = walk.keypoints
    sep = chain_separation(kp)
    frame = int(np.flatnonzero(sep > 80)[len(sep) // 4])
    xy = kp.xy.copy()
    for part in CHAIN:
        xy[frame, kp.index("RIGHT_" + part)] = kp.xy[frame, kp.index("LEFT_" + part)]
    bad = swap_labels(with_arrays(kp, xy=xy), [frame + 1])
    out, report = fix_leg_swaps(bad)
    assert report.dropped.tolist().count("R") == 1 and report.loc[frame, "dropped"] == "R"
    same = frames_matching(out, kp)
    assert same[frame + 1] and same[np.setdiff1d(np.arange(kp.n_frames), [frame])].all()


@pytest.mark.parametrize("fps", [25, 60, 240])
def test_missing_points_and_missing_frames(fps):
    walk = simulate_walk(fps=fps, noise_px=2.0, dropout_prob=0.1, pan_speed_mps=1.3, seed=9)
    kp = walk.keypoints
    n = kp.n_frames
    xy, conf = kp.xy.copy(), kp.conf.copy()
    gaps = [range(n // 5, n // 5 + 2), range(n // 3, n // 3 + 3), range(2 * n // 3, 2 * n // 3 + 4)]
    for gap in gaps:  # whole frames where the person was not detected
        xy[list(gap)] = np.nan
        conf[list(gap)] = 0.0
    holes = with_arrays(kp, xy=xy, conf=conf)
    in_gap = {f for gap in gaps for f in gap}
    frames = [f for f in choose_swaps(n, np.random.default_rng(fps)) if f not in in_gap]
    # Plus a swap that starts inside the middle gap and goes on for 0.1 s after it.
    frames += list(range(gaps[1][-1] + 1, gaps[1][-1] + 1 + int(0.1 * fps)))
    out, _ = fix_leg_swaps(swap_labels(holes, sorted(set(frames) | set(gaps[1]))))
    assert_corrected(out, holes, frames)


@pytest.mark.parametrize("fps", [30, 60, 240])
def test_variable_frame_rate(fps):
    kp = simulate_walk(fps=fps, noise_px=2.0, seed=10).keypoints
    rng = np.random.default_rng(fps)
    steps = (1.0 / fps) * (1.0 + rng.uniform(-0.2, 0.2, kp.n_frames - 1))
    vfr = with_arrays(kp, time_s=np.concatenate([[0.0], np.cumsum(steps)]))
    frames = choose_swaps(kp.n_frames, rng)
    out, _ = fix_leg_swaps(swap_labels(vfr, frames))
    assert_corrected(out, vfr, frames)


@pytest.mark.parametrize("step", [4, 8])
def test_decimated_copies_of_a_240fps_clip(step):
    """The 60 and 30 fps copies made with ``gaitbench decimate`` are cleaned as well."""
    kp = simulate_walk(fps=240, noise_px=2.0, pan_speed_mps=1.3, seed=11).keypoints
    frames = choose_swaps(kp.n_frames // step, np.random.default_rng(step))
    copy = decimate_keypoints(kp, step)
    out, report = fix_leg_swaps(swap_labels(copy, frames))
    assert_corrected(out, copy, frames)
    # The report speaks in source-video frames.
    assert set(report.loc[report.action == "swap", "frame"]) == {int(copy.frame_idx[f]) for f in frames}


def test_heavy_noise_degrades_gracefully():
    """5 px noise with the far leg at confidence 0.3 throughout: fewer swaps are fixed.

    A clean clip is still left exactly as it is. With swaps, the misses are
    a swapped first frame (one frame triple of evidence) and swaps where the
    legs are close. Two swaps a few frames apart around a crossing may be
    merged into one run, which relabels the crossing frames between them
    (legs within ~40 px there).
    """
    for fps in (25, 60, 240):
        walk = simulate_walk(fps=fps, noise_px=5.0, far_side_conf=0.3, seed=12)
        kp = walk.keypoints
        out, _ = fix_leg_swaps(kp)
        assert np.array_equal(out.xy, kp.xy, equal_nan=True)
        frames = choose_swaps(kp.n_frames, np.random.default_rng(fps))
        out, _ = fix_leg_swaps(swap_labels(kp, frames))
        assert_corrected(out, kp, frames, min_restored=0.7, max_changed=0.02)


# ---------------------------------------------------------------------------
# API details
# ---------------------------------------------------------------------------


def test_report_and_metadata():
    kp = simulate_walk(fps=30, noise_px=1.0, seed=13).keypoints
    kp = with_arrays(kp, frame_idx=kp.frame_idx + 1000, time_s=kp.time_s + 5.0)
    bad = swap_labels(kp, [10, 11, 50])
    before = bad.xy.copy()

    out, report = fix_leg_swaps(bad)

    assert np.array_equal(bad.xy, before, equal_nan=True), "input must not be modified"
    assert out is not bad and out.xy is not bad.xy
    assert tuple(report.columns) == REPORT_COLUMNS
    assert len(report) == kp.n_frames
    assert report.frame.tolist() == kp.frame_idx.tolist()
    np.testing.assert_allclose(report.time_s, kp.time_s)
    assert set(report.action) <= {"keep", "swap"}
    swapped = report[report.action == "swap"]
    assert swapped.frame.tolist() == [1010, 1011, 1050]
    assert (swapped.cost_keep > swapped.cost_swap).all()
    summary = out.meta["cleaned"]["leg_swaps"]
    assert summary["swapped_frames"] == [[1010, 1011], [1050, 1050]]
    assert summary["n_swapped"] == 3 and summary["n_dropped"] == 0
    assert summary["params"]["cost_parts"] == list(CHAIN) and "HIP" in summary["params"]["parts"]
    json.dumps(out.meta)
    # Hips travel with their leg.
    np.testing.assert_array_equal(out.point("LEFT_HIP")[10], kp.point("LEFT_HIP")[10])


def test_only_requested_parts_are_swapped():
    kp = simulate_walk(fps=30, seed=14).keypoints
    bad = swap_labels(kp, [40])
    out, _ = fix_leg_swaps(bad, parts=("KNEE", "ANKLE", "HEEL", "FOOT_INDEX", "BIG_TOE", "SMALL_TOE"))
    np.testing.assert_array_equal(out.point("LEFT_ANKLE")[40], kp.point("LEFT_ANKLE")[40])
    # HIP was not in `parts`, so it stays as the (swapped) input had it.
    np.testing.assert_array_equal(out.point("LEFT_HIP")[40], bad.point("LEFT_HIP")[40])


def test_bad_parameters_and_missing_parts():
    kp = simulate_walk(fps=30, duration_s=2.0).keypoints
    with pytest.raises(ValueError):
        fix_leg_swaps(kp, switch_penalty=-1)
    with pytest.raises(ValueError):
        fix_leg_swaps(kp, max_span_s=0)
    upper = kp.subset(["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"])
    with pytest.warns(UserWarning, match="leg swaps not checked"):
        out, report = fix_leg_swaps(upper)
    assert np.array_equal(out.xy, upper.xy) and (report.action == "keep").all()
    tiny = with_arrays(kp, xy=kp.xy[:2], conf=kp.conf[:2], time_s=kp.time_s[:2], frame_idx=kp.frame_idx[:2])
    out, report = fix_leg_swaps(tiny)
    assert len(report) == 2 and np.array_equal(out.xy, tiny.xy)


def test_cleaned_backend_name():
    assert cleaned_backend_name("openpose") == "openpose-clean"
    assert cleaned_backend_name("mediapipe-60fps") == "mediapipe-60fps-clean"
    assert cleaned_backend_name("openpose-clean") == "openpose-clean"
    assert cleaned_backend_name("my pose+model") == "my-pose-model-clean"
    assert cleaned_backend_name("") == "unknown-clean"


def test_clean_keypoints_in_memory():
    kp = simulate_walk(fps=30, seed=15).keypoints
    out, report = clean_keypoints(swap_labels(kp, [20]), step_params={"leg_swaps": {"switch_penalty": 0.4}})
    assert out.meta["backend"] == "synthetic-clean"
    assert out.meta["cleaned"]["steps"] == ["leg_swaps"]
    assert out.meta["cleaned"]["source_backend"] == "synthetic"
    assert out.meta["cleaned"]["tool_version"] == __version__
    assert out.meta["cleaned"]["leg_swaps"]["params"]["switch_penalty"] == 0.4
    assert list(report.columns[:2]) == ["clip_id", "step"] and (report.step == "leg_swaps").all()
    assert frames_matching(out, kp).all()
    with pytest.raises(ValueError, match="Unknown cleaning step"):
        clean_keypoints(kp, ["smooth"])
    with pytest.raises(TypeError, match="does not take"):
        clean_keypoints(kp, step_params={"leg_swaps": {"min_confidence": 0.2}})


def test_clean_keypoints_file(tmp_path):
    kp = simulate_walk(fps=30, seed=16).keypoints
    src = swap_labels(kp, [30, 31]).save(tmp_path / "keypoints" / "walk__synthetic.npz")

    report = clean_keypoints_file(src, report_path=tmp_path / "reports" / "walk.csv")

    dest = tmp_path / "keypoints" / "walk__synthetic-clean.npz"
    cleaned = Keypoints.load(dest)
    assert cleaned.meta["backend"] == "synthetic-clean"
    assert cleaned.meta["cleaned"]["source_path"] == str(src)
    assert cleaned.meta["cleaned"]["leg_swaps"]["swapped_frames"] == [[30, 31]]
    assert frames_matching(cleaned, kp).all()
    assert report.loc[report.action == "swap", "frame"].tolist() == [30, 31]
    assert pd.read_csv(tmp_path / "reports" / "walk.csv").shape == report.shape
    # The original is untouched, and cleaning cannot overwrite it.
    assert np.array_equal(Keypoints.load(src).xy, swap_labels(kp, [30, 31]).xy, equal_nan=True)
    with pytest.raises(ValueError, match="overwrite"):
        clean_keypoints_file(src, src)
    # Cleaning a cleaned file keeps one suffix and records both runs.
    again = tmp_path / "again.npz"
    clean_keypoints_file(dest, again)
    twice = Keypoints.load(again)
    assert twice.meta["backend"] == "synthetic-clean"
    assert twice.meta["cleaned"]["steps"] == ["leg_swaps", "leg_swaps"]
    assert twice.meta["cleaned"]["source_backend"] == "synthetic"


# ---------------------------------------------------------------------------
# real data: GPJATK sample (OpenPose, 25 fps)
# ---------------------------------------------------------------------------


def _heel_jumps(kp: Keypoints) -> dict[str, np.ndarray]:
    """Frame-to-frame change of heel-minus-pelvis (the signal Zeni uses), after its gap filling."""
    direction, _ = estimate_direction(kp)
    return {side: np.abs(np.diff(sig.raw)) for side, sig in foot_signals(kp, "heel", direction=direction).items()}


def _near_swaps(kp: Keypoints, window_s: float = 0.12) -> np.ndarray:
    t = kp.time_s[1:]
    return np.any([np.abs(t - ts) <= window_s for ts in GPJATK_SWAP_TIMES], axis=0)


def _regular_intervals(events: pd.DataFrame) -> None:
    for (side, event), group in events.groupby(["side", "event"]):
        gaps = np.diff(np.sort(group.time_s.to_numpy()))
        if len(gaps):
            assert gaps.min() >= 0.5, f"{side} {event} closer than 0.5 s: {gaps}"
            # Every stride within 25 % of the clip's median stride.
            assert np.all(np.abs(gaps / np.median(gaps) - 1.0) <= 0.25), f"{side} {event} irregular: {gaps}"


@pytest.mark.skipif(not GPJATK_LEFT.exists(), reason="GPJATK sample not fetched (scripts/fetch_sample_data.py)")
def test_gpjatk_left_swaps_are_fixed():
    kp = Keypoints.load(GPJATK_LEFT)
    out, report = fix_leg_swaps(kp)

    near = _near_swaps(kp)
    raw, clean = _heel_jumps(kp), _heel_jumps(out)
    # The uncleaned signal jumps at the known times; the cleaned one does not.
    assert max(np.nanmax(raw[s][near]) for s in "LR") > 80
    for side in "LR":
        assert np.nanmax(clean[side][near]) <= 40, f"{side}: {np.round(clean[side][near], 1)}"
        assert np.nanmax(clean[side]) <= 40

    # Every change is at one of the known moments, and each moment got one.
    changed = report[(report.action == "swap") | (report.dropped != "")]
    assert len(changed) > 0
    for t in changed.time_s:
        assert min(abs(t - ts) for ts in GPJATK_SWAP_TIMES) <= 0.12, t
    for ts in GPJATK_SWAP_TIMES:
        assert (np.abs(changed.time_s - ts) <= 0.12).any(), ts

    raw_events = run_detector("zeni", kp)
    with pytest.raises(AssertionError):
        _regular_intervals(raw_events)
    events = run_detector("zeni", out)
    _regular_intervals(events)
    hs = events[events.event == "HS"].groupby("side").size()
    assert set(hs.index) == {"L", "R"} and abs(hs["L"] - hs["R"]) <= 1


@pytest.mark.skipif(not GPJATK_RIGHT.exists(), reason="GPJATK sample not fetched (scripts/fetch_sample_data.py)")
def test_gpjatk_right_is_left_alone():
    kp = Keypoints.load(GPJATK_RIGHT)
    out, report = fix_leg_swaps(kp)
    assert (report.action == "keep").all() and (report.dropped == "").all()
    assert np.array_equal(out.xy, kp.xy, equal_nan=True)
    _regular_intervals(run_detector("zeni", out))


@pytest.mark.skipif(not GPJATK_LEFT.exists(), reason="GPJATK sample not fetched (scripts/fetch_sample_data.py)")
def test_gpjatk_clean_file_round_trip(tmp_path):
    src = tmp_path / "keypoints" / GPJATK_LEFT.name
    src.parent.mkdir()
    src.write_bytes(GPJATK_LEFT.read_bytes())
    report = clean_keypoints_file(src)
    cleaned = Keypoints.load(tmp_path / "keypoints" / "gpjatk_p01s01_sag_left__openpose-clean.npz")
    assert cleaned.meta["backend"] == "openpose-clean"
    assert cleaned.meta["importer"] == "gaitbench.io.openpose"  # the source's metadata is kept
    assert report.clip_id.unique().tolist() == ["gpjatk_p01s01_sag_left"]

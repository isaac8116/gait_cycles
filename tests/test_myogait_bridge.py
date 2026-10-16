"""Tests for gaitbench.myogait_bridge (needs myogait; some tests also need gaitkit).

Run the slow ones (the method survey and MediaPipe extraction on the sample
video) with ``-m slow -s`` to see their printed tables and timings.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

myogait = pytest.importorskip("myogait")

from gaitbench import myogait_bridge as mb  # noqa: E402
from gaitbench.formats import EVENT_COLUMNS, Keypoints, validate_events  # noqa: E402
from gaitbench.synthetic import simulate_walk  # noqa: E402

pytestmark = pytest.mark.myogait

SAMPLE_VIDEO = Path(
    os.environ.get(
        "GAITBENCH_SAMPLE_VIDEO",
        "/tmp/claude-0/-home-user-gait-cycles/c7355d3c-ee2e-55f4-9abf-b2748ece7a5c/scratchpad/sample/"
        "C_Sag_Left/p01s01_c1_pose.avi",
    )
)


def _has(package: str) -> bool:
    try:
        __import__(package)
    except ImportError:
        return False
    return True


needs_gaitkit = pytest.mark.skipif(not _has("gaitkit"), reason="gaitkit not installed")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def match_within(pred: pd.DataFrame, truth: pd.DataFrame, tol_frames: int = 2) -> pd.DataFrame:
    """Greedy one-to-one matching per (side, event), closest pairs first.

    Deliberately independent of gaitbench.scoring so these tests do not
    depend on another module.
    """
    rows = []
    for (side, event), t in truth.groupby(["side", "event"]):
        p = pred[(pred["side"] == side) & (pred["event"] == event)]
        tf, pf = t["frame"].to_numpy(), p["frame"].to_numpy()
        pairs = sorted(
            (abs(int(a) - int(b)), i, j) for i, a in enumerate(tf) for j, b in enumerate(pf) if abs(int(a) - int(b)) <= tol_frames
        )
        used_t, used_p = set(), set()
        for d, i, j in pairs:
            if i in used_t or j in used_p:
                continue
            used_t.add(i)
            used_p.add(j)
            rows.append({"side": side, "event": event, "truth": int(tf[i]), "pred": int(pf[j]), "err": int(pf[j]) - int(tf[i])})
    return pd.DataFrame(rows, columns=["side", "event", "truth", "pred", "err"])


def shift_frames(kp: Keypoints, scale: int = 1, offset: int = 0) -> Keypoints:
    """Same keypoints, renumbered as source frames ``scale * i + offset`` (times unchanged)."""
    return Keypoints(
        clip_id=kp.clip_id, names=kp.names, xy=kp.xy, conf=kp.conf,
        time_s=kp.time_s, frame_idx=kp.frame_idx * scale + offset, meta=kp.meta,
    )


@pytest.fixture(scope="module")
def clean_walk():
    return simulate_walk(fps=60.0, duration_s=6.0, clip_id="clean60")


# ---------------------------------------------------------------------------
# availability and method list
# ---------------------------------------------------------------------------


def test_list_methods_has_myogait_methods_and_consensus():
    methods = mb.list_methods()
    assert methods[: len(myogait.list_event_methods())] == list(myogait.list_event_methods())
    assert methods[-1] == "consensus"
    assert "zeni" in methods and len(set(methods)) == len(methods)
    assert mb.is_available()
    assert mb.myogait_version() == myogait.__version__


def test_missing_dependency_error_is_runtime_and_import_error():
    err = mb.MissingDependencyError("x")
    assert isinstance(err, RuntimeError) and isinstance(err, ImportError)


def test_without_myogait_functions_raise_clear_error(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "myogait", None)  # makes `import myogait` raise ImportError
    assert mb.is_available() is False
    with pytest.raises(mb.MissingDependencyError, match="pip install"):
        mb.list_methods()
    with pytest.raises(mb.MissingDependencyError):
        mb.detect_events_myogait(simulate_walk(duration_s=0.5).keypoints, "zeni")


# ---------------------------------------------------------------------------
# keypoints <-> pivot
# ---------------------------------------------------------------------------


def test_round_trip_preserves_everything():
    walk = simulate_walk(fps=60.0, duration_s=3.0, noise_px=1.5, dropout_prob=0.05, seed=3, clip_id="rt")
    kp = walk.keypoints
    # One frame with nothing detected, as extract() writes it.
    xy, conf = kp.xy.copy(), kp.conf.copy()
    xy[10], conf[10] = np.nan, 0.0
    kp = Keypoints(kp.clip_id, kp.names, xy, conf, kp.time_s + 0.0123, kp.frame_idx + 40, kp.meta)

    pivot = mb.keypoints_to_pivot(kp)
    back = mb.pivot_to_keypoints(pivot, kp.clip_id, backend="synthetic")

    assert back.names == kp.names
    np.testing.assert_array_equal(np.isnan(back.xy), np.isnan(kp.xy))
    np.testing.assert_allclose(back.xy, kp.xy, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(back.conf, kp.conf, atol=1e-12)
    np.testing.assert_array_equal(back.frame_idx, kp.frame_idx)
    np.testing.assert_allclose(back.time_s, kp.time_s, atol=1e-12)
    assert back.meta["backend"] == "synthetic" and back.meta["tool"] == "myogait"
    assert back.meta["width"] == 1920 and back.meta["height"] == 1080 and back.meta["fps"] == 60.0


def test_pivot_layout_matches_extract():
    walk = simulate_walk(fps=60.0, duration_s=1.0, dropout_prob=0.1, seed=1)
    kp = walk.keypoints
    xy, conf = kp.xy.copy(), kp.conf.copy()
    xy[5], conf[5] = np.nan, 0.0
    kp = Keypoints(kp.clip_id, kp.names, xy, conf, kp.time_s, kp.frame_idx, kp.meta)
    pivot = mb.keypoints_to_pivot(kp)

    for key in ("myogait_version", "meta", "subject", "extraction", "frames", "normalization", "angles", "events"):
        assert key in pivot
    assert pivot["meta"]["fps"] == 60.0 and pivot["meta"]["n_frames"] == kp.n_frames
    assert (pivot["meta"]["width"], pivot["meta"]["height"]) == (1920, 1080)
    assert pivot["extraction"]["model"] == "synthetic"
    assert len(pivot["frames"]) == kp.n_frames
    assert pivot["frames"][5]["landmarks"] == {} and pivot["frames"][5]["confidence"] == 0.0

    f0 = pivot["frames"][0]
    assert set(f0) >= {"frame_idx", "time_s", "landmarks", "confidence"}
    lm = f0["landmarks"]
    # All 33 MediaPipe names plus the extended toes the synthetic walker has.
    assert set(myogait.constants.MP_LANDMARK_NAMES) <= set(lm)
    assert {"LEFT_BIG_TOE", "RIGHT_SMALL_TOE"} <= set(lm)
    assert np.isnan(lm["LEFT_EYE"]["x"]) and lm["LEFT_EYE"]["visibility"] == 0.0
    j = kp.index("LEFT_ANKLE")
    assert lm["LEFT_ANKLE"]["x"] == pytest.approx(kp.xy[0, j, 0] / 1920)
    assert lm["LEFT_ANKLE"]["y"] == pytest.approx(kp.xy[0, j, 1] / 1080)
    # A point dropped in a frame that has other points is NaN with visibility 0.
    rows, cols = np.where(np.isnan(kp.xy[:, :, 0]))
    for r, c in zip(rows, cols):
        if r == 5:
            continue
        entry = pivot["frames"][r]["landmarks"][kp.names[c]]
        assert np.isnan(entry["x"]) and np.isnan(entry["y"]) and entry["visibility"] == 0.0
        break
    else:
        pytest.fail("fixture should contain a dropped point")
    # The pivot is plain JSON data.
    json.dumps(pivot)


def test_pivot_is_independent_of_keypoints():
    kp = simulate_walk(duration_s=0.5).keypoints
    pivot = mb.keypoints_to_pivot(kp)
    pivot["frames"][0]["landmarks"]["LEFT_HIP"]["x"] = -5.0
    pivot["extraction"]["gaitbench"]["meta"]["fps"] = 1.0
    assert kp.meta["fps"] == 60.0 and np.all(np.isfinite(kp.xy[0, kp.index("LEFT_HIP")]))
    xy0 = kp.xy.copy()
    again = mb.keypoints_to_pivot(kp)
    kp.xy[:] = 0.0
    back = mb.pivot_to_keypoints(again, kp.clip_id, "synthetic")
    np.testing.assert_allclose(back.xy, xy0, atol=1e-6, equal_nan=True)


def test_foot_index_derived_from_toes_and_dropped_again():
    kp = simulate_walk(duration_s=0.5).keypoints
    keep = [n for n in kp.names if not n.endswith("FOOT_INDEX")]
    kp = kp.subset(keep)  # like an OpenPose BODY_25 file: toes but no FOOT_INDEX
    pivot = mb.keypoints_to_pivot(kp)
    lm = pivot["frames"][0]["landmarks"]
    big, small = kp.point("LEFT_BIG_TOE")[0], kp.point("LEFT_SMALL_TOE")[0]
    assert lm["LEFT_FOOT_INDEX"]["x"] == pytest.approx((big[0] + small[0]) / 2 / 1920)
    back = mb.pivot_to_keypoints(pivot, kp.clip_id, "openpose")
    assert back.names == kp.names
    no_derive = mb.keypoints_to_pivot(kp, derive_foot_index=False)
    assert np.isnan(no_derive["frames"][0]["landmarks"]["LEFT_FOOT_INDEX"]["x"])


def _extract_like_pivot(kp: Keypoints, *, flipped: bool) -> dict:
    """A pivot shaped as myogait.extract writes it (MediaPipe, optional flip)."""
    from myogait.constants import MP_LANDMARK_NAMES
    from myogait.schema import create_empty

    w, h = kp.meta["width"], kp.meta["height"]
    pivot = create_empty("clip.mp4", kp.fps, w, h, kp.n_frames)
    frames = []
    for i in range(kp.n_frames):
        lm = {}
        for name in MP_LANDMARK_NAMES:
            src = name
            if flipped:
                src = name.replace("LEFT_", "RIGHT_") if name.startswith("LEFT_") else name.replace("RIGHT_", "LEFT_")
            if kp.has(src):
                x, y = kp.point(src)[i] / (w, h)
                lm[name] = {"x": float(1 - x if flipped else x), "y": float(y), "visibility": float(kp.confidence(src)[i])}
            else:
                lm[name] = {"x": 0.5, "y": 0.5, "visibility": 0.9}
        frames.append({"frame_idx": int(kp.frame_idx[i]), "time_s": round(float(kp.time_s[i]), 4), "landmarks": lm, "confidence": 0.9})
    # Frame 3: heel replaced by extract's geometric estimate.
    frames[3]["landmarks"]["LEFT_HEEL"] = {"x": 0.4, "y": 0.9, "visibility": 0.5}
    frames[3]["foot_landmarks_source"] = "estimated"
    pivot["frames"] = frames
    pivot["extraction"] = {"model": "mediapipe", "model_detail": "mediapipe", "landmark_names": list(MP_LANDMARK_NAMES), "was_flipped": flipped}
    return pivot


@pytest.mark.parametrize("flipped", [False, True])
def test_pivot_from_extract_unflips_and_drops_estimated_feet(flipped):
    kp = simulate_walk(duration_s=0.5, direction=-1).keypoints
    pivot = _extract_like_pivot(kp, flipped=flipped)
    back = mb.pivot_to_keypoints(pivot, "c", backend="mediapipe")
    # Only canonical names that MediaPipe has (no extended toes in this pivot).
    expected = tuple(n for n in kp.names if not n.endswith(("BIG_TOE", "SMALL_TOE")))
    assert back.names == expected
    for name in ("LEFT_ANKLE", "RIGHT_KNEE", "NOSE"):
        np.testing.assert_allclose(back.point(name), kp.point(name), atol=1e-3)  # 4-dp times, float coords
    heel_name = "RIGHT_HEEL" if flipped else "LEFT_HEEL"  # the estimated entry, in walker terms
    assert np.isnan(back.point(heel_name)[3]).all() and back.confidence(heel_name)[3] == 0.0
    assert back.meta["myogait_estimated_foot_points_dropped"] == 1
    kept = mb.pivot_to_keypoints(pivot, "c", backend="mediapipe", keep_estimated_feet=True)
    assert np.isfinite(kept.point(heel_name)[3]).all()
    assert back.meta["model"] == "mediapipe" and back.meta["source"] == "clip.mp4"


def test_pivot_from_subsampled_extract_scales_frame_idx():
    kp = simulate_walk(duration_s=0.5).keypoints
    pivot = _extract_like_pivot(kp, flipped=False)
    pivot["extraction"]["experimental"] = {"frame_stride": 4, "effective_fps": 15.0}
    back = mb.pivot_to_keypoints(pivot, "c", backend="mediapipe")
    np.testing.assert_array_equal(back.frame_idx, kp.frame_idx * 4)


def test_pivot_without_frame_size_is_rejected():
    kp = simulate_walk(duration_s=0.5).keypoints
    meta = {k: v for k, v in kp.meta.items() if k != "width"}
    with pytest.raises(ValueError, match="width"):
        mb.keypoints_to_pivot(Keypoints(kp.clip_id, kp.names, kp.xy, kp.conf, kp.time_s, kp.frame_idx, meta))


# ---------------------------------------------------------------------------
# event detection
# ---------------------------------------------------------------------------


def _check_zeni(kp: Keypoints, truth: pd.DataFrame) -> pd.DataFrame:
    ev = mb.detect_events_myogait(kp, "zeni")
    pd.testing.assert_frame_equal(ev, validate_events(ev))
    assert list(ev.columns) == list(EVENT_COLUMNS)
    assert set(ev["detector"]) == {"myogait:zeni"} and set(ev["tool"]) == {"myogait"}
    assert set(ev["tool_version"]) == {myogait.__version__}
    assert set(ev["backend"]) == {kp.meta["backend"]} and set(ev["clip_id"]) == {kp.clip_id}
    assert set(ev["frame"]) <= set(kp.frame_idx.tolist())
    m = match_within(ev, truth, tol_frames=2)
    assert len(m) >= 0.9 * len(truth), f"only {len(m)}/{len(truth)} truth events matched within 2 frames"
    # time_s comes from the keypoints' own timestamps.
    lookup = dict(zip(kp.frame_idx.tolist(), kp.time_s.tolist()))
    np.testing.assert_allclose(ev["time_s"], [lookup[f] for f in ev["frame"]])
    return ev


def test_zeni_finds_synthetic_events(clean_walk):
    _check_zeni(clean_walk.keypoints, clean_walk.truth)


@pytest.mark.parametrize("scale,offset", [(1, 1000), (1, 7), (2, 5)])
def test_zeni_frames_are_source_indices(clean_walk, scale, offset):
    base = mb.detect_events_myogait(clean_walk.keypoints, "zeni")
    kp = shift_frames(clean_walk.keypoints, scale, offset)
    truth = clean_walk.truth.assign(frame=clean_walk.truth["frame"] * scale + offset)
    ev = _check_zeni(kp, truth)
    np.testing.assert_array_equal(ev["frame"], base["frame"] * scale + offset)


@needs_gaitkit
@pytest.mark.parametrize("scale,offset", [(1, 0), (1, 1000), (2, 5)])
def test_consensus_frames_are_source_indices(clean_walk, scale, offset):
    # myogait 0.9.1's event_consensus returns array positions; the bridge must not.
    kp = shift_frames(clean_walk.keypoints, scale, offset)
    truth = clean_walk.truth.assign(frame=clean_walk.truth["frame"] * scale + offset)
    ev = mb.detect_events_myogait(kp, "consensus")
    assert set(ev["detector"]) == {"myogait:consensus"}
    assert set(ev["frame"]) <= set(kp.frame_idx.tolist())
    assert len(match_within(ev, truth, 2)) >= 0.9 * len(truth)
    assert ev["confidence"].between(0, 1).all()


def test_consensus_members_can_be_chosen(clean_walk):
    ev = mb.detect_events_myogait(clean_walk.keypoints, "consensus", consensus_methods=["zeni", "oconnor", "crossing"])
    assert len(match_within(ev, clean_walk.truth, 2)) >= 0.9 * len(clean_walk.truth)
    with pytest.raises(ValueError):
        mb.detect_events_myogait(clean_walk.keypoints, "consensus", consensus_methods=["zeni", "nope"])


def test_run_myogait_returns_pipeline_result_in_source_frames(clean_walk):
    kp = shift_frames(clean_walk.keypoints, 1, 500)
    for method in ("zeni", "consensus") if _has("gaitkit") else ("zeni",):
        res = mb.run_myogait(kp, method)
        assert set(res) >= {"data", "cycles", "stats", "quality", "source_type"}
        cycles = res["cycles"]["cycles"]
        assert cycles, f"{method}: no cycles"
        hs = {e["frame"] for e in res["data"]["events"]["left_hs"] + res["data"]["events"]["right_hs"]}
        for c in cycles:
            assert c["start_frame"] in hs and 500 <= c["start_frame"] < 500 + kp.n_frames
        assert res["quality"]["n_frames"] == kp.n_frames


def test_unknown_method_raises_value_error(clean_walk):
    with pytest.raises(ValueError, match="Unknown myogait event method"):
        mb.detect_events_myogait(clean_walk.keypoints, "not_a_method")


@needs_gaitkit
@pytest.mark.skipif(_has("onnxruntime"), reason="onnxruntime is installed")
def test_missing_onnxruntime_is_named(clean_walk):
    with pytest.raises(mb.MissingDependencyError, match="onnxruntime"):
        mb.detect_events_myogait(clean_walk.keypoints, "gk_intellevent")


@needs_gaitkit
@pytest.mark.skipif(_has("keras") and _has("tensorflow"), reason="tensorflow is installed")
def test_missing_tensorflow_is_named(clean_walk):
    with pytest.raises(RuntimeError, match="tensorflow|keras"):
        mb.detect_events_myogait(clean_walk.keypoints, "gk_deepevent")


@needs_gaitkit
def test_gaitkit_auto_install_disabled_during_call_and_restored(clean_walk, monkeypatch):
    monkeypatch.delenv("GAITKIT_AUTO_INSTALL_DEPS", raising=False)
    seen = []
    import gaitkit._core as core

    real = core._auto_install_enabled

    def spy(flag):
        seen.append(os.environ.get("GAITKIT_AUTO_INSTALL_DEPS"))
        return real(flag)

    monkeypatch.setattr(core, "_auto_install_enabled", spy)
    mb.detect_events_myogait(clean_walk.keypoints, "gk_zeni")
    assert seen and set(seen) == {"0"}
    assert "GAITKIT_AUTO_INSTALL_DEPS" not in os.environ


def test_run_is_quiet_and_logs_through_bridge(clean_walk, capsys, caplog):
    caplog.set_level(logging.DEBUG, logger="gaitbench.myogait_bridge")
    ev = mb.detect_events_myogait(clean_walk.keypoints, "zeni")
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
    bridge = [r for r in caplog.records if r.name == "gaitbench.myogait_bridge"]
    assert any("[myogait." in r.getMessage() for r in bridge)
    assert not any(r.name.startswith("myogait") for r in caplog.records)
    info = ev.attrs["myogait"]
    assert info["method"] == "zeni" and "warnings" in info["quality"] and info["log"]
    assert logging.getLogger("myogait").propagate is True


def test_input_keypoints_are_not_modified(clean_walk):
    kp = clean_walk.keypoints
    before = (kp.xy.copy(), kp.conf.copy(), json.dumps(kp.meta))
    mb.detect_events_myogait(kp, "zeni")
    np.testing.assert_array_equal(kp.xy, before[0])
    np.testing.assert_array_equal(kp.conf, before[1])
    assert json.dumps(kp.meta) == before[2]


# ---------------------------------------------------------------------------
# diagnostic: every method, fixed vs panning camera (prints, does not assert)
# ---------------------------------------------------------------------------


SURVEY_CONDITIONS = {
    "fixed": {},
    "pan1.3": {"pan_speed_mps": 1.3},
}


def survey_methods(fps: float = 60.0, duration_s: float = 6.0, tol_frames: int = 2) -> pd.DataFrame:
    """Recall/precision within ``tol_frames`` for every bridge method on synthetic walks."""
    rows = []
    for cond, overrides in SURVEY_CONDITIONS.items():
        walk = simulate_walk(fps=fps, duration_s=duration_s, clip_id=f"survey_{cond}", **overrides)
        truth = walk.truth
        for method in mb.list_methods():
            row = {"method": method, "condition": cond, "n_truth": len(truth)}
            t0 = time.perf_counter()
            try:
                ev = mb.detect_events_myogait(walk.keypoints, method)
            except mb.MissingDependencyError as exc:
                row["status"] = "missing dep: " + str(exc).split(" needs ", 1)[-1].split(",")[0]
                rows.append(row)
                continue
            except Exception as exc:  # report, do not fail: this is a survey
                row["status"] = f"error: {type(exc).__name__}"
                rows.append(row)
                continue
            m = match_within(ev, truth, tol_frames)
            row.update(
                status="ok",
                n_pred=len(ev),
                recall=len(m) / len(truth),
                recall_HS=(m["event"] == "HS").sum() / max((truth["event"] == "HS").sum(), 1),
                recall_TO=(m["event"] == "TO").sum() / max((truth["event"] == "TO").sum(), 1),
                precision=len(m) / len(ev) if len(ev) else np.nan,
                mean_err=m["err"].mean() if len(m) else np.nan,
                seconds=time.perf_counter() - t0,
            )
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.slow
def test_survey_all_methods_fixed_vs_panning():
    table = survey_methods()
    assert set(table["method"]) == set(mb.list_methods())
    wide = table.pivot_table(index="method", columns="condition", values="recall", aggfunc="first")
    with pd.option_context("display.width", 200, "display.max_rows", 100, "display.precision", 2):
        print("\nRecall within 2 frames (60 fps, 6 s synthetic walk):")
        print(wide.reindex(mb.list_methods()))
        print(table.drop(columns=["seconds"], errors="ignore").to_string(index=False, float_format="%.2f"))


# ---------------------------------------------------------------------------
# pose stage on the sample video (slow, downloads the MediaPipe model once)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_extract_keypoints_mediapipe_on_sample_video():
    pytest.importorskip("cv2")
    pytest.importorskip("mediapipe")
    if not SAMPLE_VIDEO.exists():
        pytest.skip(f"sample video not found: {SAMPLE_VIDEO}")
    problem = mb._mediapipe_native_problem()
    if problem:
        pytest.skip(problem)
    try:
        from myogait.models.mediapipe import _ensure_model

        _ensure_model()
    except Exception as exc:  # offline, proxy, ...
        pytest.skip(f"MediaPipe model could not be downloaded: {exc}")

    t0 = time.perf_counter()
    kp = mb.extract_keypoints(SAMPLE_VIDEO, model="mediapipe")
    elapsed = time.perf_counter() - t0

    assert kp.clip_id == SAMPLE_VIDEO.stem
    assert kp.meta["backend"] == "mediapipe" and kp.meta["tool"] == "myogait"
    assert (kp.meta["width"], kp.meta["height"]) == (960, 540) and kp.meta["fps"] == pytest.approx(25.0)
    assert kp.meta["model_package"] == "mediapipe" and kp.meta["model_package_version"]
    assert kp.meta["myogait_extraction"]["was_flipped"] is False
    assert {"LEFT_ANKLE", "RIGHT_HEEL", "LEFT_FOOT_INDEX"} <= set(kp.names)
    assert 0 < kp.n_frames <= 135 and 0 <= kp.frame_idx[0] and kp.frame_idx[-1] < 135
    np.testing.assert_allclose(kp.time_s, kp.frame_idx / 25.0, atol=1e-3)
    detected = np.isfinite(kp.xy[:, kp.index("LEFT_HIP"), 0]).mean()
    n_dec = kp.meta["n_frames_decoded"]
    print(
        f"\nMediaPipe (heavy) via myogait on CPU: {n_dec} frames in {elapsed:.1f} s "
        f"= {1000 * elapsed / n_dec:.0f} ms/frame (pose stage alone {1000 * kp.meta['pose_s_per_frame']:.0f} ms/frame); "
        f"{kp.n_frames} frames kept after myogait's trim, hip detected in {100 * detected:.0f}%"
    )
    ev = mb.detect_events_myogait(kp, "zeni")
    validate_events(ev)
    print(f"myogait:zeni found {len(ev)} events on the sample (not checked: the video has a skeleton overlay)")

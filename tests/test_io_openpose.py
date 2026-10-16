"""Tests for gaitbench.io.openpose: the GPJATK sample if present, plus small fabricated JSON directories."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from gaitbench.formats import FormatError, Keypoints
from gaitbench.io import BODY_25, COCO_18, load_openpose_dir, read_openpose_frame

SAMPLE_DIR = Path(
    os.environ.get(
        "GAITBENCH_GPJATK_SAMPLE",
        "/tmp/claude-0/-home-user-gait-cycles/c7355d3c-ee2e-55f4-9abf-b2748ece7a5c/scratchpad/sample/C_Sag_Left",
    )
)
SAMPLE_JSON = SAMPLE_DIR / "JSON Files"


# ---------------------------------------------------------------------------
# fabricated data
# ---------------------------------------------------------------------------


def person(x: float, conf: float, model: tuple[str, ...] = BODY_25, missing: tuple[str, ...] = ()) -> dict:
    """A standing figure with its hips at (x, 300); every point at confidence ``conf``."""
    offsets = {
        "Nose": (0, -250), "Neck": (0, -200), "RShoulder": (-5, -200), "LShoulder": (5, -200),
        "RElbow": (-5, -130), "LElbow": (5, -130), "RWrist": (-5, -70), "LWrist": (5, -70),
        "MidHip": (0, 0), "RHip": (-5, 0), "LHip": (5, 0), "RKnee": (-5, 120), "LKnee": (5, 120),
        "RAnkle": (-5, 240), "LAnkle": (5, 240), "REye": (-3, -255), "LEye": (3, -255),
        "REar": (-6, -250), "LEar": (6, -250), "LBigToe": (35, 255), "LSmallToe": (25, 255),
        "LHeel": (0, 255), "RBigToe": (25, 255), "RSmallToe": (15, 255), "RHeel": (-10, 255),
    }
    flat: list[float] = []
    for name in model:
        if name in missing:
            flat += [0.0, 0.0, 0.0]
        else:
            dx, dy = offsets[name]
            flat += [x + dx, 300.0 + dy, conf]
    return {"person_id": [-1], "pose_keypoints_2d": flat, "face_keypoints_2d": []}


def write_dir(root: Path, frames: list[list[dict]], prefix: str = "clip") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for i, people in enumerate(frames):
        (root / f"{prefix}_{i:012d}_keypoints.json").write_text(json.dumps({"version": 1.3, "people": people}))
    return root


# ---------------------------------------------------------------------------
# real sample
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE_JSON.is_dir(), reason="GPJATK sample not downloaded (scripts/fetch_sample_data.py)")
class TestGpjatkSample:
    def test_loads_all_frames(self) -> None:
        kp = load_openpose_dir(SAMPLE_JSON, "gpjatk_p01s01_c1", 25, 960, 540)
        assert kp.n_frames == 135
        np.testing.assert_array_equal(kp.frame_idx, np.arange(135))
        np.testing.assert_allclose(kp.time_s, np.arange(135) / 25)
        assert kp.meta["openpose_model"] == "BODY_25"
        assert kp.meta["backend"] == "openpose"
        assert kp.meta["time_source"] == "nominal_fps"
        assert kp.meta["max_people_per_frame"] >= 1
        for name in ("LEFT_HEEL", "RIGHT_HEEL", "LEFT_BIG_TOE", "RIGHT_SMALL_TOE", "LEFT_HIP", "NOSE"):
            assert kp.has(name)
        assert not kp.has("LEFT_FOOT_INDEX")
        assert np.all((kp.conf >= 0) & (kp.conf <= 1))
        # The near-side leg and hips are tracked in almost every frame.
        assert np.isfinite(kp.point("LEFT_HIP")).all()
        assert np.isfinite(kp.point("LEFT_HEEL")).mean() > 0.95
        assert np.all(kp.xy[..., 0][np.isfinite(kp.xy[..., 0])] < 960)

    def test_round_trip_and_detector(self, tmp_path: Path) -> None:
        kp = load_openpose_dir(SAMPLE_JSON, "gpjatk", 25, 960, 540)
        back = Keypoints.load(kp.save(tmp_path / "kp.npz"))
        np.testing.assert_array_equal(back.xy, kp.xy)
        detectors = pytest.importorskip("gaitbench.detectors")
        events = detectors.run_detector("zeni", kp)
        # 5.4 s of walking at about 1 stride/s: several heel strikes per side.
        for side in ("L", "R"):
            assert ((events["side"] == side) & (events["event"] == "HS")).sum() >= 3

    def test_times_from_video(self) -> None:
        video = SAMPLE_DIR / "p01s01_c1_pose.avi"
        if not video.exists():
            pytest.skip("sample video not present")
        try:
            kp = load_openpose_dir(SAMPLE_JSON, "gpjatk", video_path=video, foot_index="mean_toes")
        except RuntimeError as exc:  # neither ffprobe nor OpenCV available
            pytest.skip(str(exc))
        assert (kp.meta["fps"], kp.meta["width"], kp.meta["height"]) == (25, 960, 540)
        assert kp.meta["time_source"] == "timestamps"
        np.testing.assert_allclose(kp.time_s, np.arange(135) / 25, atol=1e-4)
        assert kp.has("LEFT_FOOT_INDEX") and kp.meta["foot_index_source"] == "mean_toes"


# ---------------------------------------------------------------------------
# fabricated data
# ---------------------------------------------------------------------------


def test_missing_points_become_nan_and_conf_is_clipped(tmp_path: Path) -> None:
    frames = [[person(100, 0.9, missing=("LHeel", "RKnee"))], [person(110, 1.04)]]
    kp = load_openpose_dir(write_dir(tmp_path / "j", frames), "c", 30, 640, 480)
    assert np.isnan(kp.point("LEFT_HEEL")[0]).all() and kp.confidence("LEFT_HEEL")[0] == 0
    assert np.isnan(kp.point("RIGHT_KNEE")[0]).all()
    assert kp.confidence("LEFT_HIP")[1] == 1.0
    np.testing.assert_allclose(kp.point("LEFT_HIP")[0], [105, 300])
    np.testing.assert_allclose(kp.point("LEFT_HEEL")[1], [110, 555])


def test_mapping_of_body25_points(tmp_path: Path) -> None:
    kp = load_openpose_dir(write_dir(tmp_path / "j", [[person(200, 0.8)]]), "c", 30, 640, 480)
    expected = {"RIGHT_HEEL": (190, 555), "LEFT_BIG_TOE": (235, 555), "RIGHT_SMALL_TOE": (215, 555),
                "LEFT_ANKLE": (205, 540), "NOSE": (200, 50), "RIGHT_WRIST": (195, 230)}
    for name, xy in expected.items():
        np.testing.assert_allclose(kp.point(name)[0], xy)
    assert "NECK" not in " ".join(kp.names)


def test_best_person_prefers_continuity_when_close(tmp_path: Path) -> None:
    frames = []
    for i in range(10):
        walker = person(100 + 10 * i, 0.85)
        # The bystander scores a little higher in frames 4-6, within the 0.15 margin.
        bystander = person(500, 0.95 if 4 <= i <= 6 else 0.5)
        frames.append([bystander, walker] if i % 2 else [walker, bystander])
    kp = load_openpose_dir(write_dir(tmp_path / "j", frames), "c", 30, 640, 480)
    hips = kp.point("LEFT_HIP")[:, 0]
    np.testing.assert_allclose(hips, 105 + 10 * np.arange(10))
    assert kp.meta["n_frames_continuity_pick"] == 3
    assert kp.meta["n_frames_multi_person"] == 10
    assert kp.meta["n_hip_jumps"] == 0


def test_best_person_switches_when_clearly_better(tmp_path: Path) -> None:
    frames = [[person(100, 0.9), person(500, 0.2)], [person(110, 0.2), person(500, 0.95)]]
    kp = load_openpose_dir(write_dir(tmp_path / "j", frames), "c", 30, 640, 480)
    np.testing.assert_allclose(kp.point("LEFT_HIP")[:, 0], [105, 505])
    assert kp.meta["n_hip_jumps"] == 1


def test_empty_frames_and_index_selection(tmp_path: Path) -> None:
    frames = [[person(100, 0.9), person(400, 0.8)], [], [person(120, 0.9)]]
    d = write_dir(tmp_path / "j", frames)
    kp = load_openpose_dir(d, "c", 30, 640, 480)
    assert kp.n_frames == 3
    assert np.isnan(kp.xy[1]).all() and (kp.conf[1] == 0).all()
    assert kp.meta["n_frames_no_person"] == 1
    second = load_openpose_dir(d, "c", 30, 640, 480, person=1)
    np.testing.assert_allclose(second.point("LEFT_HIP")[0], [405, 300])
    assert np.isnan(second.xy[2]).all()  # frame 2 has only one person
    assert second.meta["n_frames_no_person"] == 2


def test_frame_numbers_from_file_names(tmp_path: Path) -> None:
    d = tmp_path / "j"
    d.mkdir()
    for i in (3, 10, 7):
        (d / f"video_{i:012d}_keypoints.json").write_text(json.dumps({"version": 1.3, "people": [person(i, 0.9)]}))
    kp = load_openpose_dir(d, "c", 50, 640, 480)
    np.testing.assert_array_equal(kp.frame_idx, [3, 7, 10])
    np.testing.assert_allclose(kp.time_s, [0.06, 0.14, 0.2])
    np.testing.assert_allclose(kp.point("LEFT_HIP")[:, 0], [8, 12, 15])


def test_foot_index_options(tmp_path: Path) -> None:
    d = write_dir(tmp_path / "j", [[person(100, 0.9, missing=("RSmallToe",))]])
    big = load_openpose_dir(d, "c", 30, 640, 480, foot_index="big_toe")
    np.testing.assert_allclose(big.point("LEFT_FOOT_INDEX")[0], big.point("LEFT_BIG_TOE")[0])
    mean = load_openpose_dir(d, "c", 30, 640, 480, foot_index="mean_toes")
    np.testing.assert_allclose(mean.point("LEFT_FOOT_INDEX")[0], [130, 555])
    # Mean of the toes needs both toes; never a jump between one toe and the mean.
    assert np.isnan(mean.point("RIGHT_FOOT_INDEX")[0]).all()
    assert mean.confidence("RIGHT_FOOT_INDEX")[0] == 0
    with pytest.raises(ValueError):
        load_openpose_dir(d, "c", 30, 640, 480, foot_index="toe")


def test_coco18_has_no_feet(tmp_path: Path) -> None:
    d = write_dir(tmp_path / "j", [[person(100, 0.9, model=COCO_18)]])
    kp = load_openpose_dir(d, "c", 30, 640, 480)
    assert kp.meta["openpose_model"] == "COCO_18"
    assert not kp.has("LEFT_HEEL") and kp.has("LEFT_ANKLE")
    np.testing.assert_allclose(kp.point("RIGHT_KNEE")[0], [95, 420])


def test_rejects_bad_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_openpose_dir(tmp_path, "c", 30, 640, 480)
    d = write_dir(tmp_path / "j", [[person(100, 0.9)]])
    with pytest.raises(ValueError, match="fps"):
        load_openpose_dir(d, "c")
    write_dir(d, [[person(100, 0.9)]], prefix="other")
    with pytest.raises(FormatError, match="several videos"):
        load_openpose_dir(d, "c", 30, 640, 480)
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "x_000000000000_keypoints.json").write_text(json.dumps({"people": [{"pose_keypoints_2d": [1, 2, 3, 4]}]}))
    with pytest.raises(FormatError, match="pose values"):
        load_openpose_dir(bad, "c", 30, 640, 480)


def test_unnumbered_files_use_sorted_order(tmp_path: Path) -> None:
    d = tmp_path / "j"
    d.mkdir()
    for name, x in (("b_keypoints.json", 20), ("a_keypoints.json", 10)):
        (d / name).write_text(json.dumps({"people": [person(x, 0.9)]}))
    with pytest.warns(UserWarning, match="no frame number"):
        kp = load_openpose_dir(d, "c", 30, 640, 480)
    np.testing.assert_allclose(kp.point("LEFT_HIP")[:, 0], [15, 25])


def test_normalised_coordinates_warn(tmp_path: Path) -> None:
    p = person(0.5, 0.9)
    p["pose_keypoints_2d"] = [v / 1000 if i % 3 != 2 else v for i, v in enumerate(p["pose_keypoints_2d"])]
    with pytest.warns(UserWarning, match="normalised"):
        load_openpose_dir(write_dir(tmp_path / "j", [[p]]), "c", 30, 640, 480)


def test_read_openpose_frame(tmp_path: Path) -> None:
    d = write_dir(tmp_path / "j", [[person(100, 0.9), {"pose_keypoints_2d": []}]])
    people, version = read_openpose_frame(next(d.iterdir()))
    assert len(people) == 1 and people[0].shape == (25, 3)
    assert version == "1.3"

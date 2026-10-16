"""Tests for gaitbench.capture.realsense without a camera.

Pure helpers run everywhere. Tests needing pyrealsense2 build synthetic
recordings with a software device, so playback, video export, alignment and
deprojection are exercised; recording from a real D455 is not.
"""

from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from gaitbench.capture import realsense as R
from gaitbench.formats import Keypoints


# ---------------------------------------------------------------------------
# Pure helpers (no pyrealsense2 needed)
# ---------------------------------------------------------------------------


def test_package_imports_without_touching_pyrealsense2():
    import gaitbench.capture as cap

    assert callable(cap.record) and callable(cap.bag_to_video)


def test_missing_pyrealsense2_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyrealsense2", None)
    with pytest.raises(ImportError, match="pip install pyrealsense2"):
        R.import_pyrealsense2()
    monkeypatch.setattr(sys, "platform", "darwin")
    assert "pyrealsense2-macosx" in R.install_hint()


def test_validate_mode():
    assert R.validate_mode("color", (1280, 800, 30)) == (1280, 800, 30)
    assert R.validate_mode("color", [848.0, 480.0, 60.0]) == (848, 480, 60)
    assert R.validate_mode("depth", None) is None
    for bad in ((1280, 800), (1280, 800, 0), (1280, -800, 30), ("a", 2, 3), (1280.5, 800, 30), 30):
        with pytest.raises(ValueError):
            R.validate_mode("color", bad)


def test_record_validates_arguments_before_looking_for_a_camera(tmp_path):
    for kwargs in (
        {"seconds": 0},
        {"seconds": float("nan")},
        {"seconds": 5, "color": None, "depth": None},
        {"seconds": 5, "color": (1280, 800)},
        {"seconds": 5, "color_format": "mjpeg"},
        {"seconds": 5, "warmup_s": -1},
    ):
        with pytest.raises(ValueError):
            R.record(tmp_path, **kwargs)


MODES = [
    {"stream": "color", "width": 1280, "height": 800, "fps": 30, "format": "rgb8"},
    {"stream": "color", "width": 1280, "height": 800, "fps": 30, "format": "yuyv"},
    {"stream": "color", "width": 848, "height": 480, "fps": 60, "format": "rgb8"},
    {"stream": "depth", "width": 848, "height": 480, "fps": 30, "format": "z16"},
    {"stream": "gyro", "width": None, "height": None, "fps": 200, "format": "motion_xyz32f"},
]


def test_find_and_describe_modes():
    assert R.find_mode(MODES, "color", 848, 480, 60)["format"] == "rgb8"
    assert R.find_mode(MODES, "color", 1280, 800, 30, fmt="yuyv") is not None
    assert R.find_mode(MODES, "color", 1280, 800, 60) is None
    assert R.find_mode(MODES, "depth", 848, 480, 30, fmt="rgb8") is None
    assert R.describe_modes(MODES, "color") == "1280x800@30 (rgb8, yuyv), 848x480@60 (rgb8)"
    assert R.describe_modes(MODES, "infrared") == "none"


def test_estimate_recording_bytes():
    per_s = R.estimate_recording_bytes(1.0)
    assert per_s == 1280 * 800 * 3 * 30 + 848 * 480 * 2 * 30  # ~116 MB/s
    assert R.estimate_recording_bytes(10.0, depth=None) == 10 * 1280 * 800 * 3 * 30
    assert R.estimate_recording_bytes(1.0, color=(640, 480, 30), depth=None, color_format="yuyv") == 640 * 480 * 2 * 30


def test_dropped_frames():
    assert R.dropped_frames([]) == 0
    assert R.dropped_frames([5]) == 0
    assert R.dropped_frames([1, 2, 3, 4]) == 0
    assert R.dropped_frames([1, 2, 5, 6, 9]) == 4
    assert R.dropped_frames([3, 3, 4]) == 0


def test_depth_error_grows_with_the_square_of_distance():
    e2, e4 = R.depth_rms_error_m(2.0), R.depth_rms_error_m(4.0)
    assert e4 == pytest.approx(4 * e2)
    assert 0.015 < R.depth_rms_error_m(3.5) < 0.03  # a couple of centimetres at the protocol distance
    assert R.depth_rms_error_m(4.0) < 0.02 * 4.0  # within Intel's 2 % at 4 m
    arr = R.depth_rms_error_m(np.array([1.0, 2.0]))
    assert arr.shape == (2,)


def frame_rows(numbers, times_ms):
    return [{"frame": i, "rs_frame_number": n, "timestamp_ms": t, "timestamp_domain": "global_time"}
            for i, (n, t) in enumerate(zip(numbers, times_ms))]


def test_frame_times_table_and_round_trip(tmp_path):
    table = R.frame_times_table(frame_rows([7, 8, 10], [1000.0, 1033.4, 1100.1]))
    assert tuple(table.columns) == R.FRAME_COLUMNS
    np.testing.assert_allclose(table["time_s"], [0.0, 0.0334, 0.1001])
    assert table["sensor_timestamp_us"].isna().all()
    path = R.write_frame_times(table, tmp_path / "x" / "clip.frames.csv")
    back = R.read_frame_times(path)
    pd.testing.assert_frame_equal(back[list(R.FRAME_COLUMNS)], table, check_dtype=False)
    assert R.frame_times_path(tmp_path / "clip.mp4") == tmp_path / "clip.frames.csv"


def test_read_frame_times_rejects_bad_files(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"frame": [0, 1], "time_s": [0.0, 0.1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        R.read_frame_times(bad)
    table = R.frame_times_table(frame_rows([1, 2, 3], [0.0, 33.0, 66.0]))
    table.loc[1, "frame"] = 5
    R.write_frame_times(table, bad)
    with pytest.raises(ValueError, match="frame"):
        R.read_frame_times(bad)
    table = R.frame_times_table(frame_rows([1, 2, 3], [0.0, 33.0, 20.0]))
    R.write_frame_times(table, bad)
    with pytest.warns(UserWarning, match="not strictly increasing"):
        R.read_frame_times(bad)


def test_robust_depth():
    d = np.zeros((20, 30))
    d[5:15, 10:20] = 2.0
    d[10, 15] = 9.0  # a single outlier
    assert R.robust_depth(d, 15, 10) == pytest.approx(2.0)
    assert np.isnan(R.robust_depth(d, 2, 2))  # all holes
    assert R.robust_depth(d, 10, 5) == pytest.approx(2.0)  # corner of the patch: 9 valid of 25
    assert np.isnan(R.robust_depth(d, 10, 5, min_valid=10))
    assert np.isnan(R.robust_depth(d, -1, 5)) and np.isnan(R.robust_depth(d, 30, 5))
    assert np.isnan(R.robust_depth(d, np.nan, 5))
    assert np.isnan(R.robust_depth(d, 15, 10, depth_range_m=(0.3, 1.5)))
    d[d == 2.0] = np.nan
    assert np.isnan(R.robust_depth(d, 12, 8))


def test_color_to_bgr():
    cv2 = pytest.importorskip("cv2")
    rgb = np.zeros((4, 6, 3), np.uint8)
    rgb[..., 0] = 200  # red
    bgr = R.color_to_bgr(rgb.reshape(-1), "rgb8", 6, 4)
    assert bgr.shape == (4, 6, 3) and bgr[0, 0].tolist() == [0, 0, 200]
    assert R.color_to_bgr(bgr, "bgr8", 6, 4)[0, 0].tolist() == [0, 0, 200]
    rgba = np.dstack([rgb, np.full((4, 6), 255, np.uint8)])
    assert R.color_to_bgr(rgba, "rgba8", 6, 4)[0, 0].tolist() == [0, 0, 200]
    yuyv = cv2.cvtColor(np.full((4, 6, 3), 128, np.uint8), cv2.COLOR_BGR2YUV)[..., :2]
    assert R.color_to_bgr(np.ascontiguousarray(yuyv), "yuyv", 6, 4).shape == (4, 6, 3)
    grey = R.color_to_bgr(np.full((4, 6), 77, np.uint8), "y8", 6, 4)
    assert grey[0, 0].tolist() == [77, 77, 77]
    with pytest.raises(ValueError):
        R.color_to_bgr(rgb, "mjpeg", 6, 4)


# ---------------------------------------------------------------------------
# pyrealsense2, no device
# ---------------------------------------------------------------------------


@pytest.fixture
def rs():
    return pytest.importorskip("pyrealsense2")


@pytest.fixture
def no_camera(rs):
    if len(rs.context().query_devices()):
        pytest.skip("a RealSense camera is connected; these tests assume none")


def test_enum_name(rs):
    assert R.enum_name(rs.stream.color) == "color"
    assert R.enum_name(rs.format.z16) == "z16"
    assert R.enum_name(rs.timestamp_domain.global_time) == "global_time"


def test_list_modes_and_record_without_a_camera(no_camera, tmp_path):
    assert R.list_modes() == []
    with pytest.raises(RuntimeError, match="No RealSense camera"):
        R.record(tmp_path, 1.0)
    assert not list(tmp_path.iterdir())


def test_recording_suffix(rs):
    assert R.recording_suffix() in (".db3", ".bag")
    assert R.recording_sidecar_path("a/clip.db3").name == "clip.recording.json"


def test_device_modes_and_info_on_a_software_device(rs):
    dev = rs.software_device()
    dev.register_info(rs.camera_info.name, "Fake D455")
    dev.register_info(rs.camera_info.serial_number, "123")
    sensor = dev.add_sensor("RGB Camera")
    for fps in (30, 60):
        vs = rs.video_stream()
        vs.type, vs.index, vs.uid = rs.stream.color, 0, fps
        vs.width, vs.height, vs.fps, vs.bpp, vs.fmt = 640, 480, fps, 3, rs.format.rgb8
        vs.intrinsics = R.make_intrinsics(640, 480, 400, 400, 320, 240)
        sensor.add_video_stream(vs)
    assert R.device_info(dev) == {"name": "Fake D455", "serial_number": "123"}
    modes = R.device_modes(dev)
    assert [(m["stream"], m["width"], m["height"], m["fps"], m["format"]) for m in modes] == [
        ("color", 640, 480, 60, "rgb8"), ("color", 640, 480, 30, "rgb8")
    ]
    assert modes[0]["sensor"] == "RGB Camera" and modes[0]["serial"] == "123"


def test_intrinsics_round_trip_and_pinhole_deprojection(rs):
    intr = R.make_intrinsics(64, 48, 50.0, 60.0, 30.0, 20.0)
    d = R.intrinsics_to_dict(intr)
    assert d["fx"] == 50.0 and d["model"] == "none" and d["width"] == 64
    depth = np.full((48, 64), 2.5)
    depth[0:3, 0:3] = 0.0
    xy = np.array([[30.0, 20.0], [40.0, 32.0], [np.nan, 5.0], [1.0, 1.0]])
    pts = R.deproject_points(depth, xy, intr, radius_px=1, min_valid=5)
    np.testing.assert_allclose(pts[0], [0.0, 0.0, 2.5], atol=1e-6)
    np.testing.assert_allclose(pts[1], [(40 - 30) / 50 * 2.5, (32 - 20) / 60 * 2.5, 2.5], atol=1e-5)
    assert np.isnan(pts[2]).all() and np.isnan(pts[3]).all()  # missing point; 4 of 9 pixels valid < 5
    masked = R.deproject_points(depth, xy, intr, valid=np.array([False, True, True, True]))
    assert np.isnan(masked[0]).all() and np.isfinite(masked[1]).all()


# ---------------------------------------------------------------------------
# Synthetic recordings: playback, export, deprojection
# ---------------------------------------------------------------------------

N, H, W, FPS = 15, 48, 64, 30
INTR = {"fx": 50.0, "fy": 50.0, "ppx": 32.0, "ppy": 24.0}
# Frame 4 and 9-10 missing, as if dropped by the camera.
NUMBERS = [1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]


def synthetic_arrays():
    color = np.zeros((N, H, W, 3), np.uint8)
    color[..., 0] = (10 * np.arange(N))[:, None, None]  # red channel encodes the frame
    color[..., 1] = 50
    depth = np.empty((N, H, W))
    depth[:] = (1.0 + 0.1 * np.arange(N))[:, None, None]  # every frame at its own distance
    depth[:, 30:40, 50:60] = 0.0  # a hole
    times = 5000.0 + (np.array(NUMBERS) - 1) * 1000.0 / FPS
    return color, depth, times


@pytest.fixture
def recording(rs, tmp_path):
    color, depth, times = synthetic_arrays()
    path = R.write_synthetic_recording(tmp_path / "rec" / "trolley_01.bag", color, depth, FPS, INTR,
                                       timestamps_ms=times, frame_numbers=NUMBERS)
    assert path.suffix == R.recording_suffix() and path.is_file()
    return path


def test_bag_to_video_exports_every_frame_with_hardware_times(recording, tmp_path):
    cv2 = pytest.importorskip("cv2")
    with pytest.warns(UserWarning, match="3 colour frames were dropped"):
        out = R.bag_to_video(recording, tmp_path / "video")
    assert set(out) == {"video", "frames_csv", "camera_json"}
    assert out["video"].name == "trolley_01.mp4" and out["frames_csv"].name == "trolley_01.frames.csv"

    table = R.read_frame_times(out["frames_csv"])
    assert table["frame"].tolist() == list(range(N))
    assert table["rs_frame_number"].tolist() == NUMBERS
    np.testing.assert_allclose(table["time_s"], (np.array(NUMBERS) - 1) / FPS, atol=1e-6)
    assert set(table["timestamp_domain"]) == {"hardware_clock"}

    cap = cv2.VideoCapture(str(out["video"]))
    reds = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        reds.append(int(img[10, 10, 2]))
    assert len(reds) == N
    assert np.all(np.abs(np.array(reds) - 10 * np.arange(N)) <= 4)  # in order, none duplicated or lost
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(FPS)

    camera = json.loads(out["camera_json"].read_text())
    assert camera["n_frames"] == N and camera["dropped_frames"] == 3 and camera["out_of_order_frames"] == 0
    assert camera["device"]["name"] == "Synthetic D455"
    assert camera["color_stream"]["intrinsics"]["fx"] == 50.0
    assert "global shutter" in camera["notes"][0]


def test_bag_to_video_rejects_missing_file(rs, tmp_path):
    with pytest.raises(FileNotFoundError):
        R.bag_to_video(tmp_path / "nope.db3", tmp_path)


def make_keypoints(video_path, frames):
    """Two keypoints per frame: on the optical axis, and in the depth hole."""
    xy = np.zeros((len(frames), 3, 2))
    xy[:, 0] = [32.0, 24.0]
    xy[:, 1] = [42.0, 14.0]
    xy[:, 2] = [55.0, 35.0]
    conf = np.ones((len(frames), 3))
    conf[0, 1] = 0.2
    return Keypoints(
        clip_id="trolley_01",
        names=("LEFT_HEEL", "RIGHT_HEEL", "LEFT_ANKLE"),
        xy=xy,
        conf=conf,
        time_s=np.asarray(frames) / FPS,
        frame_idx=np.asarray(frames),
        meta={"fps": FPS, "width": W, "height": H, "source": str(video_path)},
    )


def test_deproject_keypoints_matches_frames_through_the_csv(recording, tmp_path):
    with pytest.warns(UserWarning):
        out = R.bag_to_video(recording, tmp_path / "video")
    frames = [0, 2, 3, 7, 14]  # video frames, some after the dropped ones
    kp = make_keypoints(out["video"], frames)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # every frame should find its depth
        pts = R.deproject_keypoints(recording, kp, min_conf=0.5)
    assert pts.shape == (5, 3, 3)
    for i, f in enumerate(frames):
        z = 1.0 + 0.1 * f  # depth of *that* recorded frame, not a neighbour's
        np.testing.assert_allclose(pts[i, 0], [0.0, 0.0, z], atol=1e-3)
        if i:
            np.testing.assert_allclose(pts[i, 1], [10 / 50 * z, -10 / 50 * z, z], atol=1e-3)
        assert np.isnan(pts[i, 2]).all()  # in the depth hole
    assert np.isnan(pts[0, 1]).all()  # below min_conf


def test_deproject_keypoints_without_csv_matches_in_order(recording, tmp_path):
    kp = make_keypoints(tmp_path / "elsewhere.mp4", [0, 5, 6])
    with pytest.warns(UserWarning, match="matching video frames to recorded frames in order"):
        pts = R.deproject_keypoints(recording, kp)
    np.testing.assert_allclose(pts[:, 0, 2], [1.0, 1.5, 1.6], atol=1e-3)
    with pytest.raises(FileNotFoundError):
        R.deproject_keypoints(recording, kp, frames_csv=tmp_path / "missing.csv")


def test_deproject_keypoints_checks_image_size(recording, tmp_path):
    kp = make_keypoints(tmp_path / "x.mp4", [0])
    kp.meta["width"] = 1280
    with pytest.raises(ValueError, match="colour stream is 64x48"):
        R.deproject_keypoints(recording, kp)


def test_synthetic_recording_validates_shapes(rs, tmp_path):
    color, depth, _ = synthetic_arrays()
    with pytest.raises(ValueError):
        R.write_synthetic_recording(tmp_path / "a", color[:3], depth, FPS, INTR)
    with pytest.raises(ValueError, match="depth_intrinsics"):
        R.write_synthetic_recording(tmp_path / "a", color, depth[:, :20, :20], FPS, INTR)

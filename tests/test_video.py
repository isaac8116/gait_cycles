"""Tests for gaitbench.video on tiny videos generated with ffmpeg (skipped without ffmpeg)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gaitbench.video import cfr_command, frame_timestamps, is_variable_frame_rate, probe_video

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe not on PATH"
)


def _ffmpeg(*args: str) -> None:
    subprocess.run([FFMPEG, "-v", "error", "-y", *args], check=True, capture_output=True)


@pytest.fixture(scope="module")
def videos(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    if FFMPEG is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not on PATH")
    d = tmp_path_factory.mktemp("videos")
    src = ["-f", "lavfi", "-i", "testsrc=size=64x48:rate=30"]
    enc = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    out = {"cfr": d / "cfr.mp4", "vfr": d / "vfr.mp4", "mjpeg": d / "clip.avi"}
    _ffmpeg(*src, "-t", "1", *enc, str(out["cfr"]))
    # Every third frame interval is doubled: times 0, 1, 3, 4, 6, 7, ... / 30 s.
    _ffmpeg(*src, "-frames:v", "20", "-vf", "setpts='(N+floor(N/2))/30/TB'", "-fps_mode", "passthrough", *enc,
            str(out["vfr"]))
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=96x54:rate=25", "-t", "0.8", "-c:v", "mjpeg", str(out["mjpeg"]))
    try:
        # Portrait phone video: stored landscape, display matrix says rotate 90 degrees clockwise.
        _ffmpeg("-display_rotation", "-90", "-i", str(out["cfr"]), "-c", "copy", str(d / "rot.mp4"))
        out["rot"] = d / "rot.mp4"
    except subprocess.CalledProcessError:
        pass  # ffmpeg < 6 has no -display_rotation
    return out


@needs_ffmpeg
@pytest.mark.parametrize("backend", ["ffprobe", "opencv"])
def test_probe_cfr(videos: dict[str, Path], backend: str) -> None:
    if backend == "opencv":
        pytest.importorskip("cv2")
    info = probe_video(videos["cfr"], backend=backend)
    assert (info["width"], info["height"]) == (64, 48)
    assert info["fps_nominal"] == pytest.approx(30.0)
    assert info["n_frames"] == 30
    assert info["duration_s"] == pytest.approx(1.0, abs=0.04)
    assert info["codec"] in ("h264", "avc1")
    assert info["is_vfr"] is False
    assert info["rotation"] == 0
    assert info["frame_interval_s"]["median"] == pytest.approx(1 / 30, rel=1e-3)
    assert info["notes"] == []


@needs_ffmpeg
@pytest.mark.parametrize("backend", ["ffprobe", "opencv"])
def test_vfr_detected_and_timestamps_exact(videos: dict[str, Path], backend: str) -> None:
    if backend == "opencv":
        pytest.importorskip("cv2")
    info = probe_video(videos["vfr"], backend=backend)
    assert info["is_vfr"] is True
    assert any("Variable frame rate" in n for n in info["notes"])
    ts = frame_timestamps(videos["vfr"], backend=backend)
    n = np.arange(20)
    np.testing.assert_allclose(ts, (n + n // 2) / 30, atol=1e-3)


@needs_ffmpeg
def test_timestamp_methods_agree(videos: dict[str, Path]) -> None:
    frames = frame_timestamps(videos["cfr"], method="frames")
    packets = frame_timestamps(videos["cfr"], method="packets")
    np.testing.assert_allclose(frames, packets, atol=1e-6)
    np.testing.assert_allclose(frames, np.arange(30) / 30, atol=1e-4)
    assert frames[0] == 0.0
    absolute = frame_timestamps(videos["cfr"], relative=False)
    np.testing.assert_allclose(np.diff(absolute), np.diff(frames))
    with pytest.raises(ValueError):
        frame_timestamps(videos["cfr"], method="nope")


@needs_ffmpeg
@pytest.mark.parametrize("backend", ["ffprobe", "opencv"])
def test_rotation_reports_display_size(videos: dict[str, Path], backend: str) -> None:
    if "rot" not in videos:
        pytest.skip("this ffmpeg cannot write a display matrix")
    if backend == "opencv":
        pytest.importorskip("cv2")
    info = probe_video(videos["rot"], backend=backend)
    assert info["rotation"] == 90
    assert (info["width"], info["height"]) == (48, 64)
    assert (info["coded_width"], info["coded_height"]) == (64, 48)
    assert any("Rotation" in n for n in info["notes"])


@needs_ffmpeg
def test_probe_mjpeg_avi(videos: dict[str, Path]) -> None:
    info = probe_video(videos["mjpeg"])
    assert info["codec"] == "mjpeg"
    assert (info["width"], info["height"], info["n_frames"]) == (96, 54, 20)
    assert info["fps_nominal"] == pytest.approx(25.0)
    assert probe_video(videos["mjpeg"], timestamps=False)["is_vfr"] is None


@needs_ffmpeg
def test_cfr_command_makes_constant_rate_copy(videos: dict[str, Path], tmp_path: Path) -> None:
    dst = tmp_path / "cfr_copy.mp4"
    cmd = cfr_command(videos["vfr"], dst, 30, preset="ultrafast", ffmpeg=FFMPEG)
    subprocess.run(cmd, check=True, capture_output=True)
    info = probe_video(dst)
    assert info["is_vfr"] is False
    assert info["fps_nominal"] == pytest.approx(30.0)
    # The source spans 29/30 s, so the copy keeps real time: about 29 frames, not 20.
    assert 28 <= info["n_frames"] <= 30


def test_cfr_command_arguments() -> None:
    cmd = cfr_command("in.mov", "out.mp4", 60)
    assert cmd[0] == "ffmpeg" and cmd[-1] == "out.mp4"
    assert cmd[cmd.index("-vf") + 1] == "fps=60"
    assert "-c:a" in cmd and "0:a:0?" in cmd
    quiet = cfr_command("in.mov", "out.mp4", 29.97, audio=False)
    assert quiet[quiet.index("-vf") + 1] == "fps=29.97" and "-an" in quiet
    assert cfr_command("a", "b", "30000/1001")[cfr_command("a", "b", "30000/1001").index("-vf") + 1] == "fps=30000/1001"
    for bad in (0, -1, "x", float("nan")):
        with pytest.raises(ValueError):
            cfr_command("a", "b", bad)


def test_is_variable_frame_rate() -> None:
    assert not is_variable_frame_rate(np.arange(100) / 60)
    assert not is_variable_frame_rate([0.0, 0.1])
    # 240 fps stored in a 1/600 s time base alternates 3- and 2-tick intervals: rounding, not VFR.
    ticks = np.round(np.arange(48) * 600 / 240) / 600
    assert is_variable_frame_rate(ticks)
    assert not is_variable_frame_rate(ticks, tick_s=1 / 600)
    dropped = np.delete(np.arange(60) / 60, 30)
    assert is_variable_frame_rate(dropped)
    assert is_variable_frame_rate([0.0, 0.1, 0.1, 0.2])
    jitter = np.arange(60) / 60 + np.random.default_rng(0).uniform(-0.0005, 0.0005, 60)
    assert not is_variable_frame_rate(jitter)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        probe_video(tmp_path / "nope.mp4")
    with pytest.raises(FileNotFoundError):
        frame_timestamps(tmp_path / "nope.mp4")
    with pytest.raises(ValueError):
        probe_video(__file__, backend="vlc")


@needs_ffmpeg
def test_falls_back_to_opencv_without_ffprobe(videos: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("cv2")
    monkeypatch.setenv("GAITBENCH_FFPROBE", "/nonexistent/ffprobe")
    info = probe_video(videos["cfr"])
    assert info["backend"].startswith("opencv")
    assert info["n_frames"] == 30
    with pytest.raises(RuntimeError, match="ffprobe"):
        probe_video(videos["cfr"], backend="ffprobe")

"""Tests for gaitbench.video on tiny videos generated with ffmpeg (skipped without ffmpeg)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gaitbench.video import cfr_command, frame_timestamps, is_variable_frame_rate, load_timestamps, probe_video

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
    assert cmd[cmd.index("-vf") + 1] == "setpts=PTS-STARTPTS,fps=60:round=up"
    assert "-c:a" in cmd and "0:a:0?" in cmd and "-af" not in cmd
    quiet = cfr_command("in.mov", "out.mp4", 29.97, audio=False)
    assert quiet[quiet.index("-vf") + 1] == "setpts=PTS-STARTPTS,fps=29.97:round=up" and "-an" in quiet
    ntsc = cfr_command("a", "b", "30000/1001")
    assert ntsc[ntsc.index("-vf") + 1] == "setpts=PTS-STARTPTS,fps=30000/1001:round=up"
    late = cfr_command("a", "b", 60, video_start_s=0.1)
    assert late[late.index("-af") + 1] == "atrim=start=0.1,asetpts=PTS-0.1/TB"
    assert "-af" not in cfr_command("a", "b", 60, audio=False, video_start_s=0.1)
    for bad in (0, -1, "x", float("nan")):
        with pytest.raises(ValueError):
            cfr_command("a", "b", bad)
    with pytest.raises(ValueError):
        cfr_command("a", "b", 60, video_start_s=-0.1)


# Frame index written into the pixels as 10 black/white blocks, so a copy's
# frames can be traced back to the source frame they show.
_W, _H, _BITS, _BLOCK = 320, 64, 10, 32


def _index_frame(i: int) -> np.ndarray:
    img = np.zeros((_H, _W), np.uint8)
    for b in range(_BITS):
        if (i >> b) & 1:
            img[:, b * _BLOCK:(b + 1) * _BLOCK] = 255
    return img


def _shown_indices(path: Path) -> np.ndarray:
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         check=True, capture_output=True).stdout
    imgs = np.frombuffer(raw, np.uint8).reshape(-1, _H, _W)
    inner = imgs[:, 8:-8, :].reshape(len(imgs), _H - 16, _BITS, _BLOCK)[:, :, :, 8:-8].mean(axis=(1, 3))
    return (inner > 127).astype(int) @ (1 << np.arange(_BITS))


@pytest.fixture(scope="module")
def slowmo(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """240 fps sources in a 1/600 s time base (as iPhone slow motion stores them), frame index in the pixels.

    ``plain`` starts with its first frame; in ``late`` the video starts 0.1 s
    after the audio, with a beep in the audio when source frame 120 is shown.
    """
    if FFMPEG is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not on PATH")
    d = tmp_path_factory.mktemp("slowmo")
    raw = d / "raw.mp4"
    proc = subprocess.Popen([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{_W}x{_H}",
                             "-r", "240", "-i", "-", "-c:v", "libx264", "-qp", "0", "-video_track_timescale", "600",
                             str(raw)], stdin=subprocess.PIPE)
    for i in range(480):
        proc.stdin.write(_index_frame(i).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    plain = d / "plain.mov"
    _ffmpeg("-i", str(raw), "-c", "copy", "-video_track_timescale", "600", str(plain))
    late = d / "late.mov"
    beep = "aevalsrc=exprs='if(between(t\\,0.6\\,0.61)\\,sin(2*PI*1000*t)\\,0)':s=48000:d=2.1"
    _ffmpeg("-itsoffset", "0.1", "-i", str(raw), "-f", "lavfi", "-i", beep, "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "pcm_s16le", "-video_track_timescale", "600", str(late))
    return {"plain": plain, "late": late}


@needs_ffmpeg
@pytest.mark.parametrize("source", ["plain", "late"])
@pytest.mark.parametrize("fps, step", [(60, 4), (30, 8)])
def test_cfr_copy_frames_carry_their_source_times(slowmo: dict[str, Path], tmp_path: Path, source: str,
                                                  fps: int, step: int) -> None:
    """Copy frame k shows source frame k * step, at that frame's own time (CFR-1)."""
    src = slowmo[source]
    info = probe_video(src, timestamps=False)
    video_start = info["start_time_s"] - info["file_start_s"]
    assert video_start == pytest.approx(0.1 if source == "late" else 0.0, abs=1e-3)
    dst = tmp_path / "copy.mp4"
    subprocess.run(cfr_command(src, dst, fps, preset="ultrafast", video_start_s=video_start, ffmpeg=FFMPEG),
                   check=True, capture_output=True)
    shown = _shown_indices(dst)
    n = 480 // step
    np.testing.assert_array_equal(shown[:n], np.arange(n) * step)
    copy_t, src_t = frame_timestamps(dst), frame_timestamps(src)
    np.testing.assert_allclose(copy_t[:n], src_t[shown[:n]], atol=1e-4)
    if source == "late":
        # The audio moved with the video: the beep is heard when frame 120 is shown (0.5 s).
        streams = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,start_time", "-of", "json", str(dst)],
            check=True, capture_output=True, text=True).stdout)["streams"]
        start = {s["codec_type"]: float(s["start_time"]) for s in streams}
        pcm = subprocess.run([FFMPEG, "-v", "error", "-i", str(dst), "-map", "0:a", "-f", "f32le", "-ac", "1",
                              "-ar", "48000", "-"], check=True, capture_output=True).stdout
        heard = int(np.argmax(np.abs(np.frombuffer(pcm, np.float32)) > 0.3)) / 48000 + start["audio"] - start["video"]
        assert heard == pytest.approx(src_t[120], abs=2e-3)


def test_load_timestamps_prefers_a_realsense_csv(tmp_path: Path) -> None:
    from gaitbench.capture.realsense import frame_times_path, frame_times_table, write_frame_times

    video = tmp_path / "rs.mp4"
    video.write_bytes(b"not a video")
    rows = [{"frame": i, "rs_frame_number": i, "timestamp_ms": 1000.0 + 40.0 * i} for i in range(3)]
    write_frame_times(frame_times_table(rows), frame_times_path(video))
    times, source, container = load_timestamps(video)
    assert source == "realsense:rs.frames.csv" and container is None
    np.testing.assert_allclose(times, [0.0, 0.04, 0.08])
    frame_times_path(video).unlink()
    with pytest.warns(UserWarning, match="Could not read frame times"):
        assert load_timestamps(video) == (None, "nominal", None)


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

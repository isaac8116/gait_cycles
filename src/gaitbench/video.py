"""Video facts the harness needs: size, frame rate, rotation and the real time of every frame.

Why per-frame times matter: iPhone video can have a **variable frame rate**
(VFR). The camera app drops or stretches frames in low light, so frame *n* is
not always at ``n / fps``. The harness therefore stores the presentation time
of every frame (``Keypoints.time_s``) and scores by time. Use
:func:`frame_timestamps` to read them, or make a constant-frame-rate copy with
:func:`cfr_command` first.

**iPhone slow motion.** A slow-motion ``.mov`` straight from the camera roll
stores frames at their real capture times (about 4.2 ms apart at 240 fps);
the slowed playback you see in Photos is an instruction to the player, not a
change to the frames. So ``probe_video`` reports ``fps_nominal`` ~ 240 and the
real duration, and the timestamps are real time: label and score them as they
are. A slow-motion clip that has been *shared or exported* (AirDrop without
"All Photos Data", Messages, many cloud uploads) is often re-rendered with the
slowdown baked in: it then probes at ~30 fps with a duration about eight times
too long, and its timestamps are no longer real time. Transfer the unmodified
original instead (AirDrop with Options > All Photos Data, Image Capture on the
Mac, or Photos > File > Export > Export Unmodified Original).

**Rotation.** Phones store portrait video as landscape frames plus a rotation
flag. OpenCV (and so myogait) applies the flag when decoding, so keypoints are
in the upright (display) orientation. ``probe_video`` therefore reports the
display ``width``/``height``, with the stored size in ``coded_width``/``coded_height``.

ffprobe is used when it is on ``PATH`` (on macOS: ``brew install ffmpeg``; set
``GAITBENCH_FFPROBE`` to use another binary); otherwise OpenCV is the
fallback. Times from either start at 0 for the first frame, so a constant-rate
video gives exactly ``frame_idx / fps``, the convention ``myogait.extract`` uses.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

# Intervals that differ from the median by more than this fraction mean VFR.
VFR_TOLERANCE = 0.10
# Transfer characteristics of HDR video (iPhone records HLG Dolby Vision by default).
_HDR_TRANSFERS = ("arib-std-b67", "smpte2084")
# Smallest timestamp step assumed for OpenCV, which does not report the time base.
_OPENCV_TICK_S = 1e-3


def ffprobe_path() -> str | None:
    """Path of the ffprobe binary (``$GAITBENCH_FFPROBE`` or ``ffprobe`` on PATH), or None."""
    return shutil.which(os.environ.get("GAITBENCH_FFPROBE", "ffprobe"))


def probe_video(path: str | Path, *, backend: str = "auto", timestamps: bool = True) -> dict[str, Any]:
    """Describe a video file.

    Parameters
    ----------
    path:
        Video file.
    backend:
        ``"auto"`` (ffprobe if available, else OpenCV), ``"ffprobe"`` or ``"opencv"``.
    timestamps:
        Also read every frame's time to decide ``is_vfr`` and count frames
        exactly. With ffprobe this reads packets only (no decoding), so it is
        quick even for 4K clips; with OpenCV every frame is decoded. False
        leaves ``is_vfr`` as None.

    Returns
    -------
    dict
        ``width``, ``height`` (display size, after rotation), ``fps_nominal``,
        ``n_frames``, ``duration_s``, ``codec``, ``is_vfr``, ``rotation``
        (degrees clockwise to display: 0, 90, 180 or 270), plus
        ``coded_width``, ``coded_height``, ``avg_fps``, ``frame_interval_s``
        (min/median/max), ``pix_fmt``, ``is_hdr``, ``device`` (iPhone model
        from QuickTime tags, if any), ``backend`` and ``notes`` (warnings
        worth reading before running pose on the clip). With ffprobe also
        ``start_time_s`` and ``file_start_s``: the first video timestamp and
        the file's (they differ when the audio starts first).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RuntimeError
        If no backend can read the video.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    backend = _resolve_backend(backend)
    info = _probe_ffprobe(path) if backend == "ffprobe" else _probe_opencv(path)
    info["is_vfr"] = None
    info["frame_interval_s"] = None
    if timestamps:
        method = "packets" if backend == "ffprobe" else "frames"
        ts = frame_timestamps(path, backend=backend, method=method)
        _add_timing(info, ts, tick=info.pop("_tick", None) or _OPENCV_TICK_S)
    info.pop("_tick", None)
    info["notes"] = _notes(info)
    return info


def frame_timestamps(
    path: str | Path,
    *,
    backend: str = "auto",
    method: str = "frames",
    relative: bool = True,
) -> np.ndarray:
    """Presentation time of every frame, in decoding-output order.

    Parameters
    ----------
    path:
        Video file.
    backend:
        ``"auto"``, ``"ffprobe"`` or ``"opencv"`` (``CAP_PROP_POS_MSEC`` after
        each grabbed frame).
    method:
        ffprobe only. ``"frames"`` (default) decodes and reads each frame's
        ``best_effort_timestamp_time``, which is what a decoder (and so
        OpenCV and myogait) will see. ``"packets"`` reads packet times without
        decoding (much faster) and sorts them; it can differ at the very start
        of streams with unusual edit lists.
    relative:
        Subtract the first frame's time so the clip starts at 0.

    Returns
    -------
    numpy.ndarray
        ``(n_frames,)`` seconds. Index ``i`` is the time of source frame ``i``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RuntimeError
        If the video cannot be read or has no timestamps.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if method not in ("frames", "packets"):
        raise ValueError(f"method must be 'frames' or 'packets', got {method!r}")
    backend = _resolve_backend(backend)
    ts = _timestamps_ffprobe(path, method) if backend == "ffprobe" else _timestamps_opencv(path)
    if ts.size == 0:
        raise RuntimeError(f"{path}: no video frames with timestamps")
    return ts - ts[0] if relative else ts


def is_variable_frame_rate(timestamps: np.ndarray, *, tolerance: float = VFR_TOLERANCE, tick_s: float = 0.0) -> bool:
    """True if frame intervals vary by more than ``tolerance`` of their median.

    Parameters
    ----------
    timestamps:
        Frame times in seconds.
    tolerance:
        Allowed deviation as a fraction of the median interval.
    tick_s:
        Resolution of the stored times (the stream time base). Deviations
        smaller than 1.5 ticks are rounding, not VFR: 240 fps in a 1/600 s
        time base alternates 3- and 2-tick intervals.

    Returns
    -------
    bool
        Also True if times repeat or go backwards; False for fewer than 3 frames.
    """
    ts = np.asarray(timestamps, dtype=float)
    if ts.size < 3:
        return False
    dt = np.diff(ts)
    if np.any(dt <= 0):
        return True
    median = float(np.median(dt))
    allowed = max(tolerance * median, 1.5 * tick_s)
    return bool(np.any(np.abs(dt - median) > allowed))


def cfr_command(
    src: str | Path,
    dst: str | Path,
    fps: float | str,
    *,
    crf: int = 18,
    preset: str = "medium",
    audio: bool = True,
    video_start_s: float = 0.0,
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    """ffmpeg command that writes a constant-frame-rate H.264 copy of a video.

    The video is first moved to start at 0 (``setpts=PTS-STARTPTS``), then
    the ``fps`` filter with ``round=up`` fills an exact ``1/fps`` grid: output
    frame ``k`` shows the last source frame at or before ``k / fps`` after the
    first source frame (duplicating or dropping where the source was
    irregular). For an integer ratio this is exactly source frame
    ``k * step`` when the stored times land on the output grid, as they do in
    the iPhone's 1/600 s time base (a coarse 1/1000 s base, as in ``.mkv``,
    can round a grid frame late and pick the one before it). So an
    unmodified 240 fps iPhone slow-motion original with
    ``fps=60`` gives a real-time 60 fps copy of frames 0, 4, 8, ..., and the
    copy's :func:`frame_timestamps` are those frames' own times, so events
    found on the copy score against labels made on the original. (ffmpeg's
    default ``round=near`` would show frames 1, 5, 9, ... at the times of
    0, 4, 8, ..., putting every event early by one source frame, and a video
    that starts after the file's first timestamp would be padded with
    repeated frames.) ffmpeg applies the rotation flag when re-encoding, so
    the copy is upright. Audio is kept (re-encoded to AAC) for clap
    synchronisation unless ``audio=False``.

    Parameters
    ----------
    src, dst:
        Input and output paths; ``dst`` should end in ``.mp4`` or ``.mov``.
    fps:
        Output frame rate, e.g. ``60``, ``29.97`` or ``"30000/1001"``.
    crf, preset:
        x264 quality settings; 18 is visually lossless for pose estimation.
    audio:
        Keep the first audio stream if there is one.
    video_start_s:
        How long after the start of the file the first video frame comes
        (:func:`probe_video`'s ``start_time_s - file_start_s``; ``gaitbench
        cfr`` reads it with ffprobe). The audio is trimmed and shifted by the
        same amount, so it stays in sync with the picture after the video is
        moved to start at 0.
    ffmpeg:
        ffmpeg executable.

    Returns
    -------
    list of str
        Arguments for :func:`subprocess.run`.
    """
    rate = _format_rate(fps)
    cmd = [ffmpeg, "-hide_banner", "-y", "-i", str(src), "-map", "0:v:0"]
    if audio:
        cmd += ["-map", "0:a:0?"]
    cmd += ["-vf", f"setpts=PTS-STARTPTS,fps={rate}:round=up", "-c:v", "libx264", "-preset", preset,
            "-crf", str(int(crf)), "-pix_fmt", "yuv420p"]
    if audio:
        offset = float(video_start_s)
        if not np.isfinite(offset) or offset < 0:
            raise ValueError(f"video_start_s must be a finite number >= 0, got {video_start_s!r}")
        if offset > 0:
            cmd += ["-af", f"atrim=start={offset!r},asetpts=PTS-{offset!r}/TB"]
        cmd += ["-c:a", "aac", "-b:a", "128k"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", str(dst)]
    return cmd


def load_timestamps(
    video_path: str | Path,
    timestamps_csv: str | Path | None = None,
) -> tuple[np.ndarray | None, str, np.ndarray | None]:
    """Find the best per-frame times for a video; the labeller and the pose stage both use this.

    Sharing one function keeps labels and keypoints on the same timeline,
    which scoring by time depends on. In order: a RealSense frames CSV next
    to the video (``<stem>.frames.csv``, written by
    :func:`gaitbench.capture.realsense.bag_to_video`; the ``.mp4`` itself is
    written at a constant nominal rate and cannot show dropped frames), then
    :func:`frame_timestamps`, then nothing (the caller falls back to
    ``frame / fps``).

    Parameters
    ----------
    video_path:
        The video.
    timestamps_csv:
        A RealSense frames CSV to use (default: the one next to the video, if any).

    Returns
    -------
    times:
        ``(n_frames,)`` seconds from the first frame, or None if only the
        nominal frame rate is available.
    source:
        ``"realsense:<file>"``, ``"video"`` or ``"nominal"``.
    container_times:
        The container's own frame times (for checking seeks), or None.
    """
    import warnings

    from .capture.realsense import frame_times_path, read_frame_times

    video_path = Path(video_path)
    csv = Path(timestamps_csv) if timestamps_csv is not None else frame_times_path(video_path)
    if timestamps_csv is not None or csv.is_file():
        table = read_frame_times(csv)
        return table["time_s"].to_numpy(dtype=float), f"realsense:{csv.name}", None
    try:
        ts = frame_timestamps(video_path)
    except (RuntimeError, OSError, ValueError) as exc:
        warnings.warn(f"Could not read frame times of {video_path} ({exc}); using frame / fps", stacklevel=2)
        return None, "nominal", None
    return np.asarray(ts, dtype=float), "video", np.asarray(ts, dtype=float)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _resolve_backend(backend: str) -> str:
    if backend not in ("auto", "ffprobe", "opencv"):
        raise ValueError(f"backend must be 'auto', 'ffprobe' or 'opencv', got {backend!r}")
    if backend == "ffprobe" and ffprobe_path() is None:
        raise RuntimeError("ffprobe not found on PATH (macOS: brew install ffmpeg)")
    if backend == "auto":
        return "ffprobe" if ffprobe_path() is not None else "opencv"
    return backend


def _run_ffprobe(path: Path, args: list[str]) -> dict[str, Any]:
    exe = ffprobe_path()
    if exe is None:
        raise RuntimeError("ffprobe not found on PATH")
    cmd = [exe, "-v", "error", *args, "-of", "json", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {proc.stderr.strip() or proc.returncode}")
    return json.loads(proc.stdout or "{}")


def _rate(text: str | None) -> float | None:
    if not text or text in ("0/0", "N/A"):
        return None
    try:
        value = float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _probe_ffprobe(path: Path) -> dict[str, Any]:
    data = _run_ffprobe(path, ["-show_streams", "-show_format"])
    streams = data.get("streams") or []
    # Skip cover art: some containers store a still image as an extra "video" stream.
    video = next(
        (s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")),
        None,
    )
    if video is None:
        raise RuntimeError(f"{path}: no video stream")
    fmt = data.get("format") or {}
    tags = {**(fmt.get("tags") or {}), **(video.get("tags") or {})}

    rotation = 0
    for side in video.get("side_data_list") or []:
        if "rotation" in side:
            # Display-matrix rotation is counter-clockwise; report clockwise like the legacy tag.
            rotation = int(round(-float(side["rotation"]))) % 360
    if not rotation and "rotate" in tags:
        rotation = int(round(float(tags["rotate"]))) % 360

    coded_w, coded_h = int(video.get("width") or 0), int(video.get("height") or 0)
    width, height = (coded_h, coded_w) if rotation in (90, 270) else (coded_w, coded_h)
    r_rate, avg_rate = _rate(video.get("r_frame_rate")), _rate(video.get("avg_frame_rate"))
    # r_frame_rate is the intended rate for normal files but can be a time-base
    # multiple (e.g. 90000/1) for VFR ones; trust it only if it agrees with the average.
    if r_rate and (avg_rate is None or abs(r_rate - avg_rate) <= 0.15 * avg_rate) and r_rate <= 1000:
        fps = r_rate
    else:
        fps = avg_rate or r_rate
    duration = _float(video.get("duration")) or _float(fmt.get("duration"))
    nb_frames = int(video["nb_frames"]) if str(video.get("nb_frames", "")).isdigit() else None
    transfer = video.get("color_transfer")
    return {
        "path": str(path),
        "backend": "ffprobe",
        "width": width,
        "height": height,
        "coded_width": coded_w,
        "coded_height": coded_h,
        "rotation": rotation,
        "fps_nominal": fps,
        "avg_fps": avg_rate,
        "r_frame_rate": video.get("r_frame_rate"),
        "n_frames": nb_frames if nb_frames else (int(round(duration * fps)) if duration and fps else None),
        "n_frames_header": nb_frames,
        "duration_s": duration,
        "codec": video.get("codec_name"),
        "pix_fmt": video.get("pix_fmt"),
        "color_transfer": transfer,
        "is_hdr": transfer in _HDR_TRANSFERS,
        "time_base": video.get("time_base"),
        "start_time_s": _float(video.get("start_time")),
        "file_start_s": _float(fmt.get("start_time")),
        "container": fmt.get("format_name"),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "creation_time": tags.get("creation_time"),
        "device": tags.get("com.apple.quicktime.model"),
        "_tick": float(Fraction(video["time_base"])) if video.get("time_base") else None,
    }


def _timestamps_ffprobe(path: Path, method: str) -> np.ndarray:
    if method == "frames":
        data = _run_ffprobe(path, ["-select_streams", "v:0", "-show_entries", "frame=best_effort_timestamp_time"])
        values = [_float(f.get("best_effort_timestamp_time")) for f in data.get("frames") or []]
        ts = np.array([np.nan if v is None else v for v in values], dtype=float)
        if ts.size and np.isnan(ts).any():
            raise RuntimeError(f"{path}: {int(np.isnan(ts).sum())} frames have no timestamp")
        return ts
    data = _run_ffprobe(path, ["-select_streams", "v:0", "-show_entries", "packet=pts_time,flags"])
    # 'D' marks packets an edit list discards; they are never displayed.
    values = [_float(p.get("pts_time")) for p in data.get("packets") or [] if "D" not in str(p.get("flags", ""))]
    return np.sort(np.array([v for v in values if v is not None], dtype=float))


def _open_capture(path: Path) -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Reading video needs ffprobe on PATH or OpenCV: pip install 'gaitbench[video]'"
        ) from exc
    # Prefer the FFmpeg backend: its timestamp and rotation handling is the one tested here.
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {path}")
    return cap


def _probe_opencv(path: Path) -> dict[str, Any]:
    import cv2

    cap = _open_capture(path)
    try:
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        rotation = int(round(cap.get(getattr(cv2, "CAP_PROP_ORIENTATION_META", -1)) or 0)) % 360
        auto = bool(cap.get(getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", -1)))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or None
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        backend_name = cap.getBackendName()
    finally:
        cap.release()
    codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ").lower() or None
    # OpenCV reports the size after applying the rotation flag when auto-rotation is on.
    if rotation in (90, 270) and auto:
        coded_w, coded_h = height, width
    else:
        coded_w, coded_h = width, height
        if rotation in (90, 270):
            width, height = height, width
    return {
        "path": str(path),
        "backend": f"opencv-{backend_name.lower()}",
        "width": width,
        "height": height,
        "coded_width": coded_w,
        "coded_height": coded_h,
        "rotation": rotation,
        "fps_nominal": fps,
        "avg_fps": None,
        "n_frames": count,
        "n_frames_header": count,
        "duration_s": count / fps if count and fps else None,
        "codec": codec,
        "pix_fmt": None,
        "color_transfer": None,
        "is_hdr": None,
        "container": path.suffix.lstrip(".").lower(),
        "device": None,
    }


def _timestamps_opencv(path: Path) -> np.ndarray:
    import cv2

    cap = _open_capture(path)
    times = []
    try:
        while cap.grab():
            times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    finally:
        cap.release()
    return np.asarray(times, dtype=float)


def _add_timing(info: dict[str, Any], ts: np.ndarray, tick: float) -> None:
    info["n_frames"] = int(ts.size)
    info["is_vfr"] = is_variable_frame_rate(ts, tick_s=tick)
    if ts.size >= 2:
        dt = np.diff(ts)
        info["frame_interval_s"] = {"min": float(dt.min()), "median": float(np.median(dt)), "max": float(dt.max())}
        span = float(ts[-1] - ts[0])
        if span > 0:
            info["avg_fps"] = (ts.size - 1) / span
        if not info.get("fps_nominal"):
            info["fps_nominal"] = 1.0 / float(np.median(dt))


def _notes(info: dict[str, Any]) -> list[str]:
    notes = []
    if info.get("is_vfr"):
        notes.append(
            "Variable frame rate: frame n is not at n/fps. Keep per-frame timestamps "
            "(gaitbench pose does) or make a CFR copy with cfr_command()."
        )
    if info.get("is_hdr"):
        notes.append(
            "HDR video (HLG/PQ): OpenCV decodes it without tone mapping, so colours look washed out. "
            "Record in SDR (Settings > Camera > Record Video > HDR Video off) if pose quality suffers."
        )
    if info.get("rotation"):
        notes.append(
            f"Rotation flag {info['rotation']} deg: decoders show it upright, so width/height are the "
            "upright size and keypoints are in upright pixels."
        )
    return notes


def _format_rate(fps: float | str) -> str:
    if isinstance(fps, str):
        value = _rate(fps)
        if value is None:
            raise ValueError(f"Invalid frame rate {fps!r}")
        return fps
    value = float(fps)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid frame rate {fps!r}")
    return str(int(value)) if value.is_integer() else repr(value)

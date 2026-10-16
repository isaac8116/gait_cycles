"""Intel RealSense D455 capture: record passes, export the colour video, measure keypoints in metres.

Workflow for the trolley test (plan E5):

1. :func:`list_modes` shows what the connected camera can stream (the D455
   colour sensor reaches 1280x800 at 30 fps; check which modes reach 60 fps).
2. :func:`record` writes one pass to a RealSense recording plus a JSON sidecar
   (device serial, firmware, stream modes, intrinsics).
3. :func:`bag_to_video` turns the recording's colour stream into an ``.mp4``
   for the pose stage and the labeller, and writes the **hardware timestamp of
   every frame** to ``<clip>.frames.csv``. The labeller picks that CSV up
   automatically; the pose stage should copy its ``time_s`` column into
   ``Keypoints.time_s`` (see :func:`read_frame_times`), because the ``.mp4``
   itself is written at a constant nominal rate and cannot show dropped frames.
4. :func:`deproject_keypoints` aligns depth to colour and turns 2-D keypoints
   into 3-D points (metres, camera frame), e.g. heel positions at heel strike
   for a metric stride length.

**Not tested on hardware.** No camera was available while this module was
written. Playback, alignment, video export and deprojection are tested on
synthetic recordings made with a software device
(:func:`write_synthetic_recording`); :func:`list_modes` and :func:`record`
follow the pyrealsense2 API but have not been run against a real D455.
Check them on the first session before relying on them.

Why the D455: its colour sensor has a **global shutter**, like the depth
imagers, so a fast-moving foot is not skewed the way it is by the iPhone's
rolling shutter, which is what the rig camera needs. Global shutter does not
remove motion blur: that depends on exposure time, so in bright light set a
short manual exposure (``color_options`` of :func:`record`).

Depth accuracy. Stereo depth error grows with the **square** of distance
(:func:`depth_rms_error_m`): with the D455's 95 mm baseline at 848x480 it is
about 0.8 cm RMS at 2 m, 2.3 cm at 3.5 m (the protocol distance) and 3 cm at
4 m on well-textured surfaces, and worse on dark, shiny or plain ones. Intel
specifies at most 2 % at 4 m. A single foot point is therefore good to a few
centimetres at 3.5 m; average a stance foot over several frames, and keep the
walker within about 4 m.

File format. librealsense 2.57 and later record ROS2 ``.db3`` files; older
versions (including some ``pyrealsense2-macosx`` builds) write ROS1 ``.bag``.
:func:`record` uses whatever the installed SDK writes, and the readers accept
both (converting ``.bag`` to ``.db3`` first if the SDK can no longer read it).

Installing pyrealsense2:

* Linux, Windows: ``pip install pyrealsense2``.
* macOS on Apple Silicon: ``pip install pyrealsense2-macosx`` (a community
  build; it is still imported as ``pyrealsense2``). There is no wheel for Intel
  Macs; build librealsense from source with ``-DBUILD_PYTHON_BINDINGS=ON``.
* macOS 12 and later: librealsense usually needs root to claim the camera
  over USB. If the camera is plugged in but :func:`list_modes` finds nothing,
  run the command with ``sudo``. Use a USB 3 (USB-C) cable straight into the
  Mac, not a hub: over USB 2 the high-resolution modes are missing.
"""

from __future__ import annotations

import contextlib
import functools
import json
import math
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .. import __version__
from ..formats import Keypoints, safe_name

DEFAULT_COLOR: tuple[int, int, int] = (1280, 800, 30)
DEFAULT_DEPTH: tuple[int, int, int] = (848, 480, 30)

# Colour formats bag_to_video can convert to BGR.
COLOR_FORMATS: tuple[str, ...] = ("rgb8", "bgr8", "rgba8", "bgra8", "yuyv", "y8")
_BYTES_PER_PIXEL = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "yuyv": 2, "y8": 1, "z16": 2}

GLOBAL_SHUTTER_NOTE = (
    "Intel RealSense D455: the RGB sensor is global shutter (as are the depth imagers), so moving feet are "
    "not skewed as with a rolling shutter. Motion blur still depends on the exposure time."
)

# Columns of the per-frame CSV written by bag_to_video.
FRAME_COLUMNS: tuple[str, ...] = (
    "frame",
    "rs_frame_number",
    "timestamp_ms",
    "timestamp_domain",
    "time_s",
    "hw_frame_counter",
    "frame_timestamp_us",
    "sensor_timestamp_us",
    "actual_exposure",
)

# D455 stereo geometry (Intel datasheet) for the depth error rule of thumb.
D455_BASELINE_M = 0.095
D455_DEPTH_HFOV_DEG = 87.0


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def install_hint() -> str:
    """How to install pyrealsense2 on this platform."""
    if sys.platform == "darwin":
        return (
            "pyrealsense2 is not installed. On an Apple Silicon Mac: pip install pyrealsense2-macosx "
            "(it is imported as pyrealsense2). Intel Macs have no wheel: build librealsense with "
            "-DBUILD_PYTHON_BINDINGS=ON. Or: pip install 'gaitbench[realsense]'."
        )
    return "pyrealsense2 is not installed: pip install pyrealsense2 (or pip install 'gaitbench[realsense]')."


def import_pyrealsense2() -> Any:
    """Import pyrealsense2, with install hints if it is missing.

    Some builds (notably on macOS) install the compiled module one level down,
    so ``import pyrealsense2`` gives an empty package; the nested module is
    used then.

    Returns
    -------
    module
        The pyrealsense2 extension module.

    Raises
    ------
    ImportError
        With platform-specific install instructions.
    """
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise ImportError(install_hint()) from exc
    if not hasattr(rs, "pipeline"):
        try:
            from pyrealsense2 import pyrealsense2 as rs  # type: ignore[no-redef]
        except ImportError as exc:
            raise ImportError("pyrealsense2 imports but has no pipeline API. " + install_hint()) from exc
    return rs


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Writing video needs OpenCV: pip install opencv-python") from exc
    return cv2


def enum_name(value: Any) -> str:
    """Short name of a pyrealsense2 enum value: ``rs.stream.color`` -> ``"color"``."""
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    return str(value).rsplit(".", 1)[-1]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def validate_mode(stream: str, mode: Sequence[int] | None) -> tuple[int, int, int] | None:
    """Check a ``(width, height, fps)`` request.

    Returns
    -------
    tuple of int or None
        The mode as ints, or None if ``mode`` is None (stream disabled).

    Raises
    ------
    ValueError
        If it is not three positive integers.
    """
    if mode is None:
        return None
    try:
        values = tuple(int(v) for v in mode)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{stream} mode must be (width, height, fps), got {mode!r}") from exc
    if len(values) != 3 or any(v <= 0 for v in values) or any(float(v) != float(m) for v, m in zip(values, mode)):
        raise ValueError(f"{stream} mode must be three positive integers (width, height, fps), got {mode!r}")
    return values  # type: ignore[return-value]


def find_mode(
    modes: Sequence[Mapping[str, Any]],
    stream: str,
    width: int,
    height: int,
    fps: int,
    fmt: str | None = None,
) -> Mapping[str, Any] | None:
    """The first entry of :func:`list_modes` output matching a request, or None."""
    for m in modes:
        if (m.get("stream"), m.get("width"), m.get("height"), m.get("fps")) == (stream, width, height, fps):
            if fmt is None or m.get("format") == fmt:
                return m
    return None


def describe_modes(modes: Sequence[Mapping[str, Any]], stream: str) -> str:
    """Compact list of one stream's modes, e.g. ``"1280x800@30 (rgb8, yuyv), 848x480@60 (rgb8)"``."""
    grouped: dict[tuple[int, int, int], set[str]] = {}
    for m in modes:
        if m.get("stream") == stream and m.get("width"):
            grouped.setdefault((m["width"], m["height"], m["fps"]), set()).add(str(m.get("format")))
    parts = [f"{w}x{h}@{f} ({', '.join(sorted(fmts))})" for (w, h, f), fmts in sorted(grouped.items(), reverse=True)]
    return ", ".join(parts) if parts else "none"


def estimate_recording_bytes(
    seconds: float,
    color: Sequence[int] | None = DEFAULT_COLOR,
    depth: Sequence[int] | None = DEFAULT_DEPTH,
    color_format: str = "rgb8",
) -> int:
    """Uncompressed size of a recording, an upper bound for the disk space it needs.

    At the defaults this is about 116 MB per second (92 MB colour, 24 MB depth).
    """
    total = 0.0
    if color is not None:
        w, h, fps = color
        total += w * h * _BYTES_PER_PIXEL.get(color_format, 3) * fps
    if depth is not None:
        w, h, fps = depth
        total += w * h * 2 * fps
    return int(total * float(seconds))


def dropped_frames(frame_numbers: Sequence[int] | np.ndarray) -> int:
    """Frames missing between consecutive frame numbers (gaps of more than 1)."""
    n = np.asarray(frame_numbers, dtype=np.int64)
    if n.size < 2:
        return 0
    d = np.diff(n)
    return int(np.sum(d[d > 1] - 1))


def depth_rms_error_m(
    distance_m: float | np.ndarray,
    *,
    baseline_m: float = D455_BASELINE_M,
    width_px: int = 848,
    hfov_deg: float = D455_DEPTH_HFOV_DEG,
    subpixel: float = 0.08,
) -> float | np.ndarray:
    """Rule-of-thumb RMS depth error of a stereo camera at a distance.

    ``error = distance**2 * subpixel / (focal_px * baseline)`` (Intel's depth
    tuning guide), with ``focal_px`` from the depth resolution and field of
    view. ``subpixel`` ~0.08 is typical for a well-textured scene; plain or
    dark surfaces are worse.

    Parameters
    ----------
    distance_m:
        Distance(s) from the camera in metres.
    baseline_m, width_px, hfov_deg:
        Camera geometry; the defaults are the D455 at 848x480.
    subpixel:
        Disparity noise in pixels.

    Returns
    -------
    float or numpy.ndarray
        Expected RMS error in metres.
    """
    focal_px = 0.5 * width_px / math.tan(math.radians(hfov_deg) / 2.0)
    z = np.asarray(distance_m, dtype=float)
    err = z**2 * subpixel / (focal_px * baseline_m)
    return float(err) if err.ndim == 0 else err


def frame_times_path(video_path: str | Path) -> Path:
    """Where the per-frame CSV of an exported video lives: ``clip.mp4`` -> ``clip.frames.csv``."""
    return Path(video_path).with_suffix(".frames.csv")


def frame_times_table(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Build the per-frame table (:data:`FRAME_COLUMNS`) from rows of frame facts.

    Each row needs ``frame``, ``rs_frame_number`` and ``timestamp_ms``; other
    columns default to NaN (``timestamp_domain`` to ``""``). ``time_s`` is
    computed as seconds since the first frame.
    """
    df = pd.DataFrame(list(rows))
    for col in FRAME_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col == "timestamp_domain" else np.nan
    df = df.loc[:, list(FRAME_COLUMNS)].copy()
    df["frame"] = df["frame"].astype(np.int64)
    df["rs_frame_number"] = df["rs_frame_number"].astype(np.int64)
    df["timestamp_ms"] = df["timestamp_ms"].astype(float)
    df["timestamp_domain"] = df["timestamp_domain"].fillna("").astype(str)
    df["time_s"] = (df["timestamp_ms"] - df["timestamp_ms"].iloc[0]) / 1000.0 if len(df) else pd.Series(dtype=float)
    for col in ("hw_frame_counter", "frame_timestamp_us", "sensor_timestamp_us", "actual_exposure"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df


def write_frame_times(table: pd.DataFrame, path: str | Path) -> Path:
    """Write a per-frame table to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.loc[:, list(FRAME_COLUMNS)].to_csv(path, index=False)
    return path


def read_frame_times(path: str | Path) -> pd.DataFrame:
    """Read a per-frame CSV written by :func:`bag_to_video`.

    Use ``read_frame_times(p)["time_s"].to_numpy()`` as ``Keypoints.time_s``
    (indexed by ``frame``) to give the pose stage the real capture times.

    Raises
    ------
    ValueError
        If columns are missing, ``frame`` is not ``0..n-1`` or a time is not finite.
    """
    df = pd.read_csv(path)
    missing = [c for c in ("frame", "rs_frame_number", "timestamp_ms", "time_s") if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if not np.array_equal(df["frame"].to_numpy(), np.arange(len(df))):
        raise ValueError(f"{path}: 'frame' must be 0, 1, 2, ... in order")
    if not np.all(np.isfinite(df["time_s"].to_numpy(dtype=float))):
        raise ValueError(f"{path}: time_s has missing values")
    if len(df) > 1 and np.any(np.diff(df["time_s"].to_numpy(dtype=float)) <= 0):
        warnings.warn(f"{path}: frame times are not strictly increasing", stacklevel=2)
    return df


def color_to_bgr(data: np.ndarray, fmt: str, width: int, height: int) -> np.ndarray:
    """Convert a RealSense colour frame buffer to an OpenCV BGR image.

    Parameters
    ----------
    data:
        The frame's pixels (``np.asanyarray(frame.get_data())``).
    fmt:
        Stream format name, one of :data:`COLOR_FORMATS`.
    width, height:
        Frame size in pixels.

    Returns
    -------
    numpy.ndarray
        ``(height, width, 3)`` uint8, contiguous.
    """
    cv2 = _import_cv2()
    a = np.asarray(data, dtype=np.uint8)
    if fmt not in COLOR_FORMATS:
        raise ValueError(f"Cannot convert colour format {fmt!r}; supported: {COLOR_FORMATS}")
    channels = _BYTES_PER_PIXEL[fmt]
    a = a.reshape(height, width, channels) if channels > 1 else a.reshape(height, width)
    if fmt == "rgb8":
        return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    if fmt == "bgr8":
        return np.ascontiguousarray(a)
    if fmt == "rgba8":
        return cv2.cvtColor(a, cv2.COLOR_RGBA2BGR)
    if fmt == "bgra8":
        return cv2.cvtColor(a, cv2.COLOR_BGRA2BGR)
    if fmt == "yuyv":
        return cv2.cvtColor(a, cv2.COLOR_YUV2BGR_YUYV)
    return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)


def robust_depth(
    depth_m: np.ndarray,
    x: float,
    y: float,
    *,
    radius_px: int = 2,
    min_valid: int = 3,
    depth_range_m: tuple[float, float] = (0.3, 10.0),
) -> float:
    """Median depth in a small window around a pixel, ignoring holes.

    Parameters
    ----------
    depth_m:
        ``(H, W)`` depth in metres, 0 where the camera has no depth.
    x, y:
        Pixel coordinates (column, row).
    radius_px:
        Half-size of the window: 2 gives 5x5 pixels (~3 cm at 4 m on the D455's colour grid).
    min_valid:
        Fewest valid pixels needed; fewer gives NaN.
    depth_range_m:
        Depths outside ``(near, far)`` are ignored as invalid.

    Returns
    -------
    float
        Depth in metres, or NaN.
    """
    if not (np.isfinite(x) and np.isfinite(y)):
        return math.nan
    h, w = depth_m.shape[:2]
    xi, yi = int(round(float(x))), int(round(float(y)))
    if not (0 <= xi < w and 0 <= yi < h):
        return math.nan
    r = int(radius_px)
    win = np.asarray(depth_m[max(0, yi - r): yi + r + 1, max(0, xi - r): xi + r + 1], dtype=float)
    near, far = depth_range_m
    v = win[np.isfinite(win) & (win > 0) & (win >= near) & (win <= far)]
    if v.size < max(1, int(min_valid)):
        return math.nan
    return float(np.median(v))


def deproject_points(
    depth_m: np.ndarray,
    xy: np.ndarray,
    intrinsics: Any,
    *,
    valid: np.ndarray | None = None,
    radius_px: int = 2,
    min_valid: int = 3,
    depth_range_m: tuple[float, float] = (0.3, 10.0),
) -> np.ndarray:
    """3-D camera-frame points (metres) for pixels of one frame.

    Parameters
    ----------
    depth_m:
        ``(H, W)`` depth in metres on the same pixel grid as ``xy`` (i.e.
        depth aligned to colour).
    xy:
        ``(K, 2)`` pixel coordinates; NaN rows give NaN.
    intrinsics:
        ``rs.intrinsics`` of that pixel grid (with its distortion model).
    valid:
        Optional ``(K,)`` mask; False rows give NaN.
    radius_px, min_valid, depth_range_m:
        See :func:`robust_depth`.

    Returns
    -------
    numpy.ndarray
        ``(K, 3)``: X right, Y down, Z forward from the camera, in metres
        (the RealSense convention); NaN where there is no depth.
    """
    rs = import_pyrealsense2()
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    out = np.full((xy.shape[0], 3), np.nan)
    for k, (x, y) in enumerate(xy):
        if valid is not None and not valid[k]:
            continue
        z = robust_depth(depth_m, x, y, radius_px=radius_px, min_valid=min_valid, depth_range_m=depth_range_m)
        if math.isfinite(z):
            out[k] = rs.rs2_deproject_pixel_to_point(intrinsics, [float(x), float(y)], z)
    return out


def intrinsics_to_dict(intr: Any) -> dict[str, Any]:
    """JSON-friendly copy of an ``rs.intrinsics``."""
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "ppx": float(intr.ppx),
        "ppy": float(intr.ppy),
        "model": enum_name(intr.model),
        "coeffs": [float(c) for c in intr.coeffs],
    }


def make_intrinsics(
    width: int,
    height: int,
    fx: float,
    fy: float,
    ppx: float,
    ppy: float,
    model: str = "none",
    coeffs: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0),
) -> Any:
    """Build an ``rs.intrinsics`` (e.g. from :func:`intrinsics_to_dict` output)."""
    rs = import_pyrealsense2()
    intr = rs.intrinsics()
    intr.width, intr.height = int(width), int(height)
    intr.fx, intr.fy, intr.ppx, intr.ppy = float(fx), float(fy), float(ppx), float(ppy)
    intr.model = getattr(rs.distortion, model)
    intr.coeffs = [float(c) for c in coeffs]
    return intr


def _version_tuple(text: str) -> tuple[int, ...]:
    parts = []
    for p in str(text).split("."):
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------


_INFO_KEYS = (
    "name",
    "serial_number",
    "firmware_version",
    "recommended_firmware_version",
    "product_line",
    "product_id",
    "usb_type_descriptor",
)


def device_info(device: Any) -> dict[str, str]:
    """Name, serial, firmware, USB type and product line of a device (those it reports)."""
    rs = import_pyrealsense2()
    info: dict[str, str] = {}
    for key in _INFO_KEYS:
        field = getattr(rs.camera_info, key, None)
        if field is not None and device.supports(field):
            info[key] = str(device.get_info(field))
    return info


def device_modes(device: Any, *, video_only: bool = True) -> list[dict[str, Any]]:
    """Every stream profile of one device as a dict (see :func:`list_modes`)."""
    rs = import_pyrealsense2()
    info = device_info(device)
    out = []
    for sensor in device.query_sensors():
        sensor_name = sensor.get_info(rs.camera_info.name) if sensor.supports(rs.camera_info.name) else ""
        for p in sensor.get_stream_profiles():
            width = height = None
            if p.is_video_stream_profile():
                vp = p.as_video_stream_profile()
                width, height = int(vp.width()), int(vp.height())
            elif video_only:
                continue
            out.append({
                "device": info.get("name", ""),
                "serial": info.get("serial_number", ""),
                "sensor": str(sensor_name),
                "stream": enum_name(p.stream_type()),
                "index": int(p.stream_index()),
                "width": width,
                "height": height,
                "fps": int(p.fps()),
                "format": enum_name(p.format()),
            })
    out.sort(key=lambda m: (m["serial"], m["sensor"], m["stream"], m["index"], -(m["width"] or 0), -(m["height"] or 0), -m["fps"], m["format"]))
    return out


def list_modes(*, video_only: bool = True) -> list[dict[str, Any]]:
    """Stream modes of every connected RealSense device.

    Parameters
    ----------
    video_only:
        Leave out the IMU (accelerometer, gyro) streams, which have no image size.

    Returns
    -------
    list of dict
        One per mode: ``device``, ``serial``, ``sensor`` (e.g. ``"RGB Camera"``),
        ``stream`` (``"color"``, ``"depth"``, ``"infrared"``), ``index``,
        ``width``, ``height``, ``fps``, ``format`` (e.g. ``"rgb8"``, ``"z16"``).
        Empty if no device is connected (on macOS, see the module notes on sudo).

    Notes
    -----
    Not tested on a real device.
    """
    rs = import_pyrealsense2()
    ctx = rs.context()
    modes: list[dict[str, Any]] = []
    for dev in ctx.query_devices():
        modes.extend(device_modes(dev, video_only=video_only))
    return modes


@functools.lru_cache(maxsize=1)
def recording_suffix() -> str:
    """File extension the installed librealsense records to: ``".db3"`` (2.57+) or ``".bag"``."""
    rs = import_pyrealsense2()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            rec = rs.recorder(str(Path(tmp) / "probe.bag"), rs.software_device())
        except RuntimeError as exc:
            if "db3" in str(exc):
                return ".db3"
            raise
        del rec
    return ".bag"


def recording_sidecar_path(recording: str | Path) -> Path:
    """JSON sidecar of a recording: ``clip.db3`` -> ``clip.recording.json``."""
    return Path(recording).with_suffix(".recording.json")


def record(
    out_dir: str | Path,
    seconds: float,
    color: Sequence[int] | None = DEFAULT_COLOR,
    depth: Sequence[int] | None = DEFAULT_DEPTH,
    clip_id: str | None = None,
    *,
    serial: str | None = None,
    color_format: str = "rgb8",
    color_options: Mapping[str, float] | None = None,
    warmup_s: float = 1.0,
    overwrite: bool = False,
) -> Path:
    """Record one pass from a connected RealSense camera.

    Parameters
    ----------
    out_dir:
        Folder for the recording and its sidecar.
    seconds:
        Length of the recording.
    color, depth:
        ``(width, height, fps)`` of each stream, or None to leave it out.
        Use the same fps for both so every colour frame has a depth frame.
    clip_id:
        File name stem; default ``realsense_<date>_<time>``.
    serial:
        Camera to use when several are connected (default: the first).
    color_format:
        Colour pixel format, one of :data:`COLOR_FORMATS`.
    color_options:
        Colour sensor options to set first, by ``rs.option`` name in SDK
        units, e.g. ``{"enable_auto_exposure": 0, "exposure": 100}`` for a
        short fixed exposure (check the range in RealSense Viewer). Values are
        checked against the sensor's range.
    warmup_s:
        Stream this long before recording starts, so auto exposure and white
        balance settle (the recorder is paused meanwhile).
    overwrite:
        Replace an existing recording of the same name.

    Returns
    -------
    pathlib.Path
        The recording (``.db3`` or ``.bag``, whichever the SDK writes; see
        :func:`recording_suffix`). The sidecar is :func:`recording_sidecar_path`.

    Raises
    ------
    ValueError
        For bad arguments, or a mode the camera does not offer.
    RuntimeError
        If no camera is connected or it stops delivering frames.
    FileExistsError
        If the recording exists and ``overwrite`` is False.

    Notes
    -----
    Not tested on hardware. Recordings are large (up to ~116 MB/s at the
    default modes, less after the SDK's compression); keep passes short.
    """
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be a positive number")
    color_mode = validate_mode("color", color)
    depth_mode = validate_mode("depth", depth)
    if color_mode is None and depth_mode is None:
        raise ValueError("Enable at least one of color and depth")
    if color_format not in COLOR_FORMATS:
        raise ValueError(f"color_format must be one of {COLOR_FORMATS}, got {color_format!r}")
    if not math.isfinite(float(warmup_s)) or warmup_s < 0:
        raise ValueError("warmup_s must be >= 0")
    clip_id = clip_id or time.strftime("realsense_%Y%m%d_%H%M%S")
    stem = safe_name(clip_id)

    rs = import_pyrealsense2()
    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        hint = " On macOS this usually needs sudo (see gaitbench.capture.realsense)." if sys.platform == "darwin" else ""
        raise RuntimeError("No RealSense camera found. Check the USB 3 cable and port." + hint)
    if serial is not None:
        matches = [d for d in devices if device_info(d).get("serial_number") == str(serial)]
        if not matches:
            raise RuntimeError(f"No camera with serial {serial}; found {[device_info(d).get('serial_number') for d in devices]}")
        dev = matches[0]
    else:
        dev = devices[0]
    info = device_info(dev)
    modes = device_modes(dev)
    for stream, mode, fmt in (("color", color_mode, color_format), ("depth", depth_mode, "z16")):
        if mode is not None and find_mode(modes, stream, *mode, fmt=fmt) is None:
            raise ValueError(
                f"{info.get('name', 'camera')} has no {stream} mode {mode[0]}x{mode[1]}@{mode[2]} {fmt}. "
                f"Available: {describe_modes(modes, stream)}"
            )
    _warn_about_device(info)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}{recording_suffix()}"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (pass overwrite=True to replace it)")
    need = estimate_recording_bytes(seconds + warmup_s, color_mode, depth_mode, color_format)
    free = shutil.disk_usage(out_dir).free
    if free < need:
        warnings.warn(f"Only {free / 1e9:.1f} GB free; an uncompressed recording needs up to {need / 1e9:.1f} GB", stacklevel=2)

    color_sensor = dev.first_color_sensor() if color_mode is not None else None
    if color_options:
        if color_sensor is None:
            raise ValueError("color_options given but the color stream is disabled")
        _set_options(rs, color_sensor, color_options)

    cfg = rs.config()
    if "serial_number" in info:
        cfg.enable_device(info["serial_number"])
    if color_mode is not None:
        w, h, fps = color_mode
        cfg.enable_stream(rs.stream.color, w, h, getattr(rs.format, color_format), fps)
    if depth_mode is not None:
        w, h, fps = depth_mode
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    cfg.enable_record_to_file(str(path))

    pipeline = rs.pipeline(ctx)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    profile = pipeline.start(cfg)
    seen: dict[str, list[tuple[int, float, str]]] = {"color": [], "depth": []}
    notes = [GLOBAL_SHUTTER_NOTE]
    try:
        recorder = None
        if warmup_s > 0:
            try:
                recorder = profile.get_device().as_recorder()
                recorder.pause()
            except (RuntimeError, TypeError) as exc:
                recorder = None
                notes.append(f"Could not pause the recorder for warm-up ({exc}); the first {warmup_s} s are recorded")
            _pull_frames(rs, pipeline, warmup_s, None)
            if recorder is not None:
                recorder.resume()
        t0 = time.monotonic()
        _pull_frames(rs, pipeline, seconds, seen)
        recorded_s = time.monotonic() - t0
        streams = [_profile_info(p) for p in profile.get_streams()]
        extrinsics = _depth_to_color_extrinsics(rs, profile)
        depth_scale = float(dev.first_depth_sensor().get_depth_scale()) if depth_mode is not None else None
        option_values = _read_options(rs, color_sensor) if color_sensor is not None else {}
    finally:
        pipeline.stop()
    # The recorder finishes the file when the last reference to it goes.
    del recorder, profile, pipeline

    sidecar = {
        "clip_id": clip_id,
        "recording": path.name,
        "started_local": started,
        "seconds_requested": seconds,
        "seconds_recorded": round(recorded_s, 3),
        "warmup_s": warmup_s,
        "device": info,
        "streams": streams,
        "depth_scale_m": depth_scale,
        "depth_to_color_extrinsics": extrinsics,
        "color_options_set": dict(color_options or {}),
        "color_options": option_values,
        "frames_seen_by_app": {k: len(v) for k, v in seen.items()},
        # Gaps here are frames the app did not pull in time, not necessarily
        # frames missing from the file; bag_to_video reports those.
        "frame_gaps_seen_by_app": {k: dropped_frames([n for n, _, _ in v]) for k, v in seen.items()},
        "timestamp_domain": {k: (v[0][2] if v else None) for k, v in seen.items()},
        "notes": notes,
        "tested_on_hardware": False,
        "gaitbench_version": __version__,
        "pyrealsense2_version": getattr(rs, "__version__", None),
    }
    recording_sidecar_path(path).write_text(json.dumps(sidecar, indent=2))
    return path


def _warn_about_device(info: Mapping[str, str]) -> None:
    usb = info.get("usb_type_descriptor", "")
    if usb.startswith("2"):
        warnings.warn(f"Camera is on USB {usb}: high-resolution modes need USB 3. Use a USB 3 cable and port.", stacklevel=3)
    fw, rec = info.get("firmware_version"), info.get("recommended_firmware_version")
    if fw and rec and _version_tuple(fw) < _version_tuple(rec):
        warnings.warn(f"Camera firmware {fw} is older than the SDK's recommended {rec}; update with RealSense Viewer", stacklevel=3)


def _set_options(rs: Any, sensor: Any, options: Mapping[str, float]) -> None:
    for name, value in options.items():
        opt = getattr(rs.option, name, None)
        if opt is None:
            raise ValueError(f"Unknown RealSense option {name!r}")
        if not sensor.supports(opt):
            raise ValueError(f"The colour sensor does not support option {name!r}")
        rng = sensor.get_option_range(opt)
        if not rng.min <= float(value) <= rng.max:
            raise ValueError(f"{name}={value} is outside the sensor's range [{rng.min}, {rng.max}]")
        sensor.set_option(opt, float(value))


def _read_options(rs: Any, sensor: Any) -> dict[str, float]:
    values = {}
    for name in ("enable_auto_exposure", "exposure", "gain", "enable_auto_white_balance", "white_balance", "global_time_enabled"):
        opt = getattr(rs.option, name, None)
        try:
            if opt is not None and sensor.supports(opt):
                values[name] = float(sensor.get_option(opt))
        except RuntimeError:
            pass
    return values


def _pull_frames(rs: Any, pipeline: Any, seconds: float, seen: dict[str, list] | None) -> None:
    """Take framesets from a running pipeline for ``seconds``."""
    t0 = time.monotonic()
    misses = 0
    while time.monotonic() - t0 < seconds:
        ok, frames = pipeline.try_wait_for_frames(1000)
        if not ok:
            misses += 1
            if misses >= 5:
                raise RuntimeError("The camera stopped delivering frames (check the USB 3 connection)")
            continue
        misses = 0
        if seen is None:
            continue
        for f in frames:
            key = enum_name(f.get_profile().stream_type())
            if key in seen:
                seen[key].append((int(f.get_frame_number()), float(f.get_timestamp()), enum_name(f.get_frame_timestamp_domain())))


def _profile_info(p: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "stream": enum_name(p.stream_type()),
        "index": int(p.stream_index()),
        "format": enum_name(p.format()),
        "fps": int(p.fps()),
    }
    if p.is_video_stream_profile():
        vp = p.as_video_stream_profile()
        out.update(width=int(vp.width()), height=int(vp.height()), intrinsics=intrinsics_to_dict(vp.get_intrinsics()))
    return out


def _depth_to_color_extrinsics(rs: Any, profile: Any) -> dict[str, list[float]] | None:
    try:
        d = profile.get_stream(rs.stream.depth)
        c = profile.get_stream(rs.stream.color)
        e = d.get_extrinsics_to(c)
    except RuntimeError:
        return None
    return {"rotation": [float(v) for v in e.rotation], "translation_m": [float(v) for v in e.translation]}


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _playback(rs: Any, path: Path, streams: Sequence[Any], workdir: Path | None = None) -> Iterator[tuple[Any, Any, Any]]:
    """Play a recording through a pipeline, in non-real-time mode, without repeating."""
    with contextlib.ExitStack() as stack:
        try:
            pipeline, profile, playback = _start_playback(rs, path, streams)
        except RuntimeError as exc:
            if path.suffix.lower() != ".bag" or not hasattr(rs.context, "convert_bag_to_db3"):
                raise RuntimeError(f"Cannot play {path} with streams {[enum_name(s) for s in streams]}: {exc}") from exc
            # A newer SDK that no longer reads ROS1 bags directly can still convert them.
            target_dir = workdir if workdir is not None else Path(stack.enter_context(tempfile.TemporaryDirectory()))
            converted = target_dir / (path.stem + ".db3")
            if not converted.exists():
                rs.context().convert_bag_to_db3(str(path), str(converted))
            pipeline, profile, playback = _start_playback(rs, converted, streams)
        try:
            yield pipeline, profile, playback
        finally:
            pipeline.stop()


def _start_playback(rs: Any, path: Path, streams: Sequence[Any]) -> tuple[Any, Any, Any]:
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(str(path), repeat_playback=False)
    for s in streams:
        cfg.enable_stream(s)
    # Switch the file's device to non-real-time *before* streaming starts. Set
    # after pipeline.start(), the playback briefly runs in real time and frames
    # the pipeline queue cannot hold are dropped (seen with short files).
    resolved = cfg.resolve(rs.pipeline_wrapper(pipeline))
    resolved.get_device().as_playback().set_real_time(False)
    profile = pipeline.start(cfg)
    playback = profile.get_device().as_playback()
    if playback.is_real_time():
        playback.set_real_time(False)
    return pipeline, profile, playback


def _framesets(rs: Any, pipeline: Any, playback: Any, *, timeout_ms: int = 250, idle_s: float = 10.0) -> Iterator[Any]:
    """Every frameset of a non-repeating playback, until the file ends."""
    got_any = False
    idle_since: float | None = None
    while True:
        ok, frames = pipeline.try_wait_for_frames(timeout_ms)
        if ok:
            got_any = True
            idle_since = None
            yield frames
            continue
        if got_any and playback.current_status() == rs.playback_status.stopped:
            return
        idle_since = idle_since or time.monotonic()
        if time.monotonic() - idle_since > idle_s:
            if got_any:
                warnings.warn("Playback stalled before the end of the file; output may be incomplete", stacklevel=3)
            return


def _frame_row(rs: Any, f: Any, index: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "frame": index,
        "rs_frame_number": int(f.get_frame_number()),
        "timestamp_ms": float(f.get_timestamp()),
        "timestamp_domain": enum_name(f.get_frame_timestamp_domain()),
    }
    md = rs.frame_metadata_value
    for col, key in (
        ("hw_frame_counter", md.frame_counter),
        ("frame_timestamp_us", md.frame_timestamp),
        ("sensor_timestamp_us", md.sensor_timestamp),
        ("actual_exposure", md.actual_exposure),
    ):
        row[col] = float(f.get_frame_metadata(key)) if f.supports_frame_metadata(key) else math.nan
    return row


def bag_to_video(
    bag_path: str | Path,
    out_dir: str | Path,
    *,
    stem: str | None = None,
    fourcc: str = "mp4v",
) -> dict[str, Path]:
    """Export a recording's colour stream to ``.mp4`` with a per-frame timestamp CSV.

    Parameters
    ----------
    bag_path:
        A RealSense recording (``.db3`` or ``.bag``), e.g. from :func:`record`
        or RealSense Viewer.
    out_dir:
        Output folder.
    stem:
        Output file stem (default: the recording's).
    fourcc:
        OpenCV codec code. ``mp4v`` plays everywhere; the frames are
        re-compressed, which pose models tolerate.

    Returns
    -------
    dict
        ``video``: ``<stem>.mp4`` at the nominal frame rate, one video frame
        per recorded colour frame, in order; ``frames_csv``: ``<stem>.frames.csv``
        with :data:`FRAME_COLUMNS` (``frame`` is the video frame index,
        ``time_s`` the hardware timestamp in seconds from the first frame);
        ``camera_json``: ``<stem>.camera.json`` with the colour intrinsics,
        device info, frame count and dropped-frame count.

    Notes
    -----
    Frames dropped during recording leave gaps in ``rs_frame_number`` and
    ``time_s`` but not in the video, so use ``time_s`` for timing, never
    ``frame / fps``. A warning is raised if any were dropped.
    """
    bag_path = Path(bag_path)
    if not bag_path.is_file():
        raise FileNotFoundError(bag_path)
    rs = import_pyrealsense2()
    cv2 = _import_cv2()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_name(stem or bag_path.stem)
    video = out_dir / f"{stem}.mp4"
    frames_csv = frame_times_path(video)
    camera_json = out_dir / f"{stem}.camera.json"

    rows: list[dict[str, Any]] = []
    with _playback(rs, bag_path, [rs.stream.color], workdir=out_dir) as (pipeline, profile, playback):
        cprof = profile.get_stream(rs.stream.color).as_video_stream_profile()
        width, height, fps = int(cprof.width()), int(cprof.height()), int(cprof.fps())
        fmt = enum_name(cprof.format())
        if fmt not in COLOR_FORMATS:
            raise ValueError(f"{bag_path}: colour format {fmt!r} is not supported; supported: {COLOR_FORMATS}")
        info = device_info(profile.get_device())
        stream = _profile_info(cprof)
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*fourcc), float(fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV cannot write {video} with codec {fourcc!r}")
        seen: set[int] = set()
        try:
            for frames in _framesets(rs, pipeline, playback):
                cf = frames.get_color_frame()
                if not cf:
                    continue
                number = int(cf.get_frame_number())
                if number in seen:  # the syncer can deliver a frame twice
                    continue
                seen.add(number)
                writer.write(color_to_bgr(np.asanyarray(cf.get_data()), fmt, width, height))
                rows.append(_frame_row(rs, cf, len(rows)))
        finally:
            writer.release()
    if not rows:
        raise RuntimeError(f"{bag_path}: no colour frames were read")

    table = frame_times_table(rows)
    write_frame_times(table, frames_csv)
    numbers = table["rs_frame_number"].to_numpy()
    dropped = dropped_frames(numbers)
    out_of_order = int(np.sum(np.diff(numbers) < 0)) if len(numbers) > 1 else 0
    if dropped:
        warnings.warn(f"{bag_path.name}: {dropped} colour frames were dropped during recording; use time_s from {frames_csv.name}", stacklevel=2)
    camera = {
        "source": str(bag_path),
        "video": video.name,
        "frames_csv": frames_csv.name,
        "n_frames": len(table),
        "nominal_fps": fps,
        "duration_s": float(table["time_s"].iloc[-1]) if len(table) else 0.0,
        "dropped_frames": dropped,
        "out_of_order_frames": out_of_order,
        "timestamp_domain": str(table["timestamp_domain"].iloc[0]),
        "device": info,
        "color_stream": stream,
        "notes": [GLOBAL_SHUTTER_NOTE],
        "gaitbench_version": __version__,
    }
    camera_json.write_text(json.dumps(camera, indent=2))
    return {"video": video, "frames_csv": frames_csv, "camera_json": camera_json}


def deproject_keypoints(
    bag_path: str | Path,
    kp: Keypoints,
    *,
    frames_csv: str | Path | None = None,
    radius_px: int = 2,
    min_valid: int = 3,
    min_conf: float = 0.0,
    depth_range_m: tuple[float, float] = (0.3, 10.0),
    max_sync_ms: float | None = None,
) -> np.ndarray:
    """3-D positions (metres, camera frame) of 2-D keypoints from the recording's depth.

    Depth is aligned to the colour image with ``rs.align``; for each keypoint
    the median of the valid depths in a small window (:func:`robust_depth`)
    is deprojected through the colour intrinsics with
    ``rs2_deproject_pixel_to_point``.

    Parameters
    ----------
    bag_path:
        The recording the keypoints' video was exported from.
    kp:
        Keypoints of the exported video (pixel coordinates of the colour
        frames; ``frame_idx`` = video frame index).
    frames_csv:
        The frames CSV from :func:`bag_to_video`, which maps video frames to
        recorded frame numbers. Default: next to ``kp.meta['source']`` or next
        to the recording; if none is found, frames are matched in order (with
        a warning), which is wrong if the export skipped any frame.
    radius_px, min_valid, depth_range_m:
        Window and validity rules of :func:`robust_depth`.
    min_conf:
        Keypoints with lower confidence give NaN.
    max_sync_ms:
        Largest colour-depth timestamp difference accepted (default: half the
        longer frame interval of the two streams). The closest depth frame is
        used when the syncer offers several.

    Returns
    -------
    numpy.ndarray
        ``(n_frames, n_keypoints, 3)`` X right, Y down, Z forward in metres
        (RealSense camera convention); NaN where the keypoint is missing, has
        no valid depth, or its frame has no depth frame close enough in time.

    Raises
    ------
    ValueError
        If the keypoints' image size differs from the colour stream's.

    Notes
    -----
    Expected error grows with the square of distance (:func:`depth_rms_error_m`):
    ~2-3 cm per point at 3.5 m. Points on the edge of a foot can pick up
    floor or background depth; the median limits but does not remove this.
    """
    bag_path = Path(bag_path)
    if not bag_path.is_file():
        raise FileNotFoundError(bag_path)
    rs = import_pyrealsense2()
    pos_of_video_frame = {int(f): i for i, f in enumerate(kp.frame_idx)}
    out = np.full((kp.n_frames, len(kp.names), 3), np.nan)
    best_dt: dict[int, float] = {}
    order: dict[int, int] = {}  # frame number -> video index, when matching in order
    align = rs.align(rs.stream.color)

    with _playback(rs, bag_path, [rs.stream.color, rs.stream.depth]) as (pipeline, profile, playback):
        cprof = profile.get_stream(rs.stream.color).as_video_stream_profile()
        dprof = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        kw, kh = kp.meta.get("width"), kp.meta.get("height")
        if kw and kh and (int(kw), int(kh)) != (cprof.width(), cprof.height()):
            raise ValueError(
                f"Keypoints are for {kw}x{kh} images but the colour stream is {cprof.width()}x{cprof.height()}"
            )
        if max_sync_ms is None:
            max_sync_ms = 0.5 * 1000.0 / min(cprof.fps(), dprof.fps())
        number_of = _video_to_frame_number(kp, bag_path, frames_csv)
        if number_of is not None:
            wanted = {number_of[v]: pos for v, pos in pos_of_video_frame.items() if v in number_of}
        for frames in _framesets(rs, pipeline, playback):
            cf = frames.get_color_frame()
            df = frames.get_depth_frame()
            if not cf:
                continue
            number = int(cf.get_frame_number())
            if number_of is None:
                if number not in order:
                    order[number] = len(order)
                pos = pos_of_video_frame.get(order[number])
            else:
                pos = wanted.get(number)
            if pos is None or not df:
                continue
            dt = abs(float(df.get_timestamp()) - float(cf.get_timestamp()))
            if dt > max_sync_ms or dt >= best_dt.get(number, math.inf):
                continue
            aligned = align.process(frames).get_depth_frame()
            if not aligned:
                continue
            depth_m = np.asanyarray(aligned.get_data()).astype(np.float32) * float(aligned.get_units())
            intr = aligned.get_profile().as_video_stream_profile().get_intrinsics()
            valid = kp.conf[pos] >= min_conf if min_conf > 0 else None
            out[pos] = deproject_points(depth_m, kp.xy[pos], intr, valid=valid, radius_px=radius_px,
                                        min_valid=min_valid, depth_range_m=depth_range_m)
            best_dt[number] = dt

    n_without = kp.n_frames - len(best_dt)
    if n_without:
        warnings.warn(f"{n_without} of {kp.n_frames} keypoint frames had no matching depth frame", stacklevel=2)
    return out


def _video_to_frame_number(kp: Keypoints, bag_path: Path, frames_csv: str | Path | None) -> dict[int, int] | None:
    candidates: list[Path] = []
    if frames_csv is not None:
        candidates.append(Path(frames_csv))
    source = kp.meta.get("source")
    if source:
        candidates.append(frame_times_path(source))
    candidates.append(frame_times_path(bag_path))
    for c in candidates:
        if c.is_file():
            table = read_frame_times(c)
            return dict(zip(table["frame"].astype(int), table["rs_frame_number"].astype(int)))
    if frames_csv is not None:
        raise FileNotFoundError(frames_csv)
    warnings.warn(
        "No frames CSV found for these keypoints; matching video frames to recorded frames in order",
        stacklevel=3,
    )
    return None


# ---------------------------------------------------------------------------
# Synthetic recordings (tests and demos)
# ---------------------------------------------------------------------------


def write_synthetic_recording(
    path: str | Path,
    color_rgb: np.ndarray,
    depth_m: np.ndarray,
    fps: int,
    color_intrinsics: Mapping[str, float],
    depth_intrinsics: Mapping[str, float] | None = None,
    *,
    timestamps_ms: Sequence[float] | None = None,
    frame_numbers: Sequence[int] | None = None,
    depth_units: float = 0.001,
    name: str = "Synthetic D455",
    serial: str = "000000000000",
) -> Path:
    """Write a recording from arrays through a software device, as a camera would.

    For testing playback, export and deprojection without hardware. Colour
    and depth share timestamps and frame numbers, and the extrinsics between
    them are identity (the two "sensors" coincide).

    Parameters
    ----------
    path:
        Output file; its suffix is replaced by :func:`recording_suffix`.
    color_rgb:
        ``(N, H, W, 3)`` uint8 RGB frames.
    depth_m:
        ``(N, Hd, Wd)`` depth in metres (0 = no depth).
    fps:
        Nominal rate of both streams.
    color_intrinsics, depth_intrinsics:
        ``fx``, ``fy``, ``ppx``, ``ppy`` (pinhole, no distortion). The depth
        default is the colour's, which requires equal image sizes.
    timestamps_ms:
        ``(N,)`` frame times (default: 1000 ms + n / fps).
    frame_numbers:
        ``(N,)`` frame numbers (default: 1..N); gaps simulate dropped frames.
    depth_units:
        Metres per depth unit (the D455 uses 0.001).
    name, serial:
        Device name and serial stored in the file.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    rs = import_pyrealsense2()
    color_rgb = np.asarray(color_rgb, dtype=np.uint8)
    depth_m = np.asarray(depth_m, dtype=float)
    if color_rgb.ndim != 4 or color_rgb.shape[-1] != 3 or depth_m.ndim != 3 or len(depth_m) != len(color_rgb):
        raise ValueError("color_rgb must be (N, H, W, 3) and depth_m (N, Hd, Wd) with the same N")
    n, ch, cw = color_rgb.shape[:3]
    dh, dw = depth_m.shape[1:]
    if depth_intrinsics is None:
        if (dh, dw) != (ch, cw):
            raise ValueError("depth_intrinsics is required when depth and colour sizes differ")
        depth_intrinsics = color_intrinsics
    ts = np.asarray(timestamps_ms if timestamps_ms is not None else 1000.0 + np.arange(n) * 1000.0 / fps, dtype=float)
    numbers = np.asarray(frame_numbers if frame_numbers is not None else np.arange(1, n + 1), dtype=np.int64)
    path = Path(path).with_suffix(recording_suffix())
    path.parent.mkdir(parents=True, exist_ok=True)

    dev = rs.software_device()
    dev.register_info(rs.camera_info.name, name)
    dev.register_info(rs.camera_info.serial_number, serial)
    depth_sensor = dev.add_sensor("Stereo Module")
    color_sensor = dev.add_sensor("RGB Camera")

    def stream(kind: Any, uid: int, w: int, h: int, bpp: int, fmt: Any, intr: Mapping[str, float]) -> Any:
        vs = rs.video_stream()
        vs.type, vs.index, vs.uid = kind, 0, uid
        vs.width, vs.height, vs.fps, vs.bpp, vs.fmt = w, h, int(fps), bpp, fmt
        vs.intrinsics = make_intrinsics(w, h, intr["fx"], intr["fy"], intr["ppx"], intr["ppy"])
        return vs

    dprof = depth_sensor.add_video_stream(stream(rs.stream.depth, 0, dw, dh, 2, rs.format.z16, depth_intrinsics))
    cprof = color_sensor.add_video_stream(stream(rs.stream.color, 1, cw, ch, 3, rs.format.rgb8, color_intrinsics))
    depth_sensor.add_read_only_option(rs.option.depth_units, float(depth_units))
    identity = rs.extrinsics()
    identity.rotation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    identity.translation = [0.0, 0.0, 0.0]
    dprof.register_extrinsics_to(cprof, identity)
    cprof.register_extrinsics_to(dprof, identity)
    dev.create_matcher(rs.matchers.default)

    recorder = rs.recorder(str(path), dev)
    depth_sensor.open(dprof)
    color_sensor.open(cprof)
    sync = rs.syncer()
    depth_sensor.start(sync)
    color_sensor.start(sync)
    raw_depth = np.clip(np.round(np.nan_to_num(depth_m, nan=0.0) / depth_units), 0, 65535).astype(np.uint16)
    for i in range(n):
        for sensor, prof, pixels, bpp in (
            (depth_sensor, dprof, np.ascontiguousarray(raw_depth[i]), 2),
            (color_sensor, cprof, np.ascontiguousarray(color_rgb[i]), 3),
        ):
            f = rs.software_video_frame()
            f.pixels = pixels
            f.stride = pixels.shape[1] * bpp
            f.bpp = bpp
            f.timestamp = float(ts[i])
            f.domain = rs.timestamp_domain.hardware_clock
            f.frame_number = int(numbers[i])
            f.profile = prof.as_video_stream_profile()
            if sensor is depth_sensor:
                f.depth_units = float(depth_units)
            sensor.on_video_frame(f)
    depth_sensor.stop()
    color_sensor.stop()
    depth_sensor.close()
    color_sensor.close()
    del recorder  # writes and closes the file
    return path

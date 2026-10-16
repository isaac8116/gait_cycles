"""Capture from the Intel RealSense D455 (global-shutter colour + depth).

pyrealsense2 is imported only when a function needs it, so this package
imports without it. See :mod:`gaitbench.capture.realsense` for install notes
(macOS uses the ``pyrealsense2-macosx`` wheel) and for what was not tested on
hardware.
"""

from .realsense import (
    COLOR_FORMATS,
    DEFAULT_COLOR,
    DEFAULT_DEPTH,
    FRAME_COLUMNS,
    bag_to_video,
    deproject_keypoints,
    depth_rms_error_m,
    frame_times_path,
    import_pyrealsense2,
    list_modes,
    read_frame_times,
    record,
    recording_suffix,
    write_synthetic_recording,
)

__all__ = [
    "COLOR_FORMATS",
    "DEFAULT_COLOR",
    "DEFAULT_DEPTH",
    "FRAME_COLUMNS",
    "bag_to_video",
    "deproject_keypoints",
    "depth_rms_error_m",
    "frame_times_path",
    "import_pyrealsense2",
    "list_modes",
    "read_frame_times",
    "record",
    "recording_suffix",
    "write_synthetic_recording",
]

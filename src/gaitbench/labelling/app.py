"""A frame-stepping OpenCV labeller for heel strikes and toe offs.

Run it with :func:`run_labeller` (or the ``gaitbench label`` command). It
shows one video frame at a time; the rater steps to the frame where a foot
lands or lifts and presses a key. Labels are written in the labels format of
:mod:`gaitbench.formats` (``tool = "human"``, ``detector`` = rater).

Keys (letters are the primary keys on every platform)::

    d / a      +1 / -1 frame
    D / A      +10 / -10 frames
    w / s      next / previous labelled event
    1          left heel strike  (L HS)
    2          left toe off      (L TO)
    3          right heel strike (R HS)
    4          right toe off     (R TO)
    x          delete the event nearest the current frame (within 3 frames)
    u / r      undo / redo
    z          toggle 2x zoom on the feet (needs keypoints_path)
    h          show / hide the key help
    Enter      save
    q / Esc    save and quit

Arrow keys (right/left = +1/-1 frame, up/down = +10/-10) work only where the
OpenCV build reports them; ``cv2.waitKeyEx`` returns different codes for them
on macOS, Linux and Windows, and some builds do not report them at all. If
``d`` suddenly jumps 10 frames, Caps Lock is on. Clicking the whole-clip
timeline strip jumps to that frame.

Left and right are the **walker's own** sides. When the walker moves left to
right across the screen, their right side faces the camera.

**Blind labelling.** The labeller never shows any detector's events, or the
pose keypoints, so a rater cannot be pulled towards what a tool found.
Keypoints, if given, are used only to place the zoom window on the feet.

Pressing a key for an event that is already labelled within 2 frames on the
same side moves it rather than adding a second one. On quit (``q``, Esc, or
closing the window) unsaved labels are saved. Every 20 edits the labels are
also written to ``<out_csv>.autosave``; if that file is newer than
``out_csv`` when the labeller starts, it is resumed from (crash recovery).
An existing ``out_csv`` is otherwise loaded, so labelling can be resumed.
Next to the labels a small JSON file (``<out_csv stem>.json``) records the
video, the timestamp source and the frame count.

Frame times come from, in order: a RealSense frames CSV next to the video
(written by :func:`gaitbench.capture.realsense.bag_to_video`), then
:func:`gaitbench.video.frame_timestamps` (real per-frame times, needed for
variable-frame-rate iPhone clips), then ``frame / fps`` as a last resort,
which is wrong for VFR video and says so on screen.

macOS: OpenCV windows must be created on the main thread, so call
:func:`run_labeller` from the main thread (a plain script or the CLI, not a
worker thread). ``opencv-python-headless`` has no windows at all; install
``opencv-python`` instead.

Frames are read by index. Seeking is slow for some codecs (HEVC from
iPhones) and, for variable-frame-rate files, OpenCV's seek can land on the
wrong frame; the reader caches a window of decoded frames, reads forward
instead of seeking where it can, and checks each seek against the frame
times, falling back to decoding from the start if seeks prove inexact.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .. import __version__
from ..formats import SIDE_PREFIX, SIDES, Keypoints
from .state import Edit, LabelSession

# Key name (as returned by decode_key) -> action.
KEY_BINDINGS: dict[str, str] = {
    "d": "step:1",
    "a": "step:-1",
    "D": "step:10",
    "A": "step:-10",
    "right": "step:1",
    "left": "step:-1",
    "up": "step:10",
    "down": "step:-10",
    "w": "next_event",
    "s": "prev_event",
    "1": "add:L:HS",
    "2": "add:L:TO",
    "3": "add:R:HS",
    "4": "add:R:TO",
    "x": "delete",
    "u": "undo",
    "r": "redo",
    "z": "zoom",
    "h": "help",
    "enter": "save",
    "q": "quit",
    "esc": "quit",
}

# ASCII only: OpenCV's Hershey fonts cannot draw other characters.
HELP_LINES: tuple[str, ...] = (
    "d / a      +1 / -1 frame",
    "D / A      +10 / -10 frames",
    "w / s      next / previous labelled event",
    "1 / 2      LEFT heel strike / LEFT toe off",
    "3 / 4      RIGHT heel strike / RIGHT toe off",
    "x          delete nearest event (within 3 frames)",
    "u / r      undo / redo",
    "z          2x zoom on the feet (needs keypoints)",
    "h          show / hide this help",
    "Enter      save",
    "q / Esc    save and quit",
    "click the top timeline to jump there",
    "",
    "Sides are the WALKER'S own left and right.",
    "Walking left->right on screen: their RIGHT side faces you.",
)

AUTOSAVE_EVERY = 20
DELETE_DISTANCE_FRAMES = 3
MESSAGE_SECONDS = 2.5

# BGR colours: blue for left, orange for right (distinguishable with colour blindness).
SIDE_COLOURS: dict[str, tuple[int, int, int]] = {"L": (235, 150, 40), "R": (30, 140, 255)}

# cv2.waitKeyEx codes. Windows puts arrows in the high word; X11 (GTK, Qt on
# Linux) uses keysyms and adds modifier state above bit 16; Cocoa returns the
# Unicode "function key" characters.
_ARROWS_FULL = {0x250000: "left", 0x260000: "up", 0x270000: "right", 0x280000: "down"}
_KEYS_LOW = {
    0xFF51: "left", 0xFF52: "up", 0xFF53: "right", 0xFF54: "down",
    0xF702: "left", 0xF700: "up", 0xF703: "right", 0xF701: "down",
    13: "enter", 10: "enter", 0xFF0D: "enter", 0xFF8D: "enter",
    27: "esc", 0xFF1B: "esc",
}
_KEYPAD_DIGITS = {0xFFB0 + d: str(d) for d in range(10)}

FOOT_PARTS: tuple[str, ...] = ("ANKLE", "HEEL", "FOOT_INDEX", "BIG_TOE", "SMALL_TOE")


# ---------------------------------------------------------------------------
# Keys and actions (pure, testable without a window)
# ---------------------------------------------------------------------------


def decode_key(code: int | None) -> str | None:
    """Turn a ``cv2.waitKeyEx`` code into a key name.

    Parameters
    ----------
    code:
        The value returned by ``cv2.waitKeyEx`` (-1 for no key).

    Returns
    -------
    str or None
        A printable character (``"d"``, ``"D"``, ``"1"``), or one of
        ``"enter"``, ``"esc"``, ``"left"``, ``"right"``, ``"up"``, ``"down"``;
        None for no key or a key with no name.
    """
    if code is None or code < 0:
        return None
    if code in _ARROWS_FULL:
        return _ARROWS_FULL[code]
    low = code & 0xFFFF
    if low in _KEYS_LOW:
        return _KEYS_LOW[low]
    if low in _KEYPAD_DIGITS:
        return _KEYPAD_DIGITS[low]
    if 32 < low < 127:
        return chr(low)
    return None


@dataclass
class ViewState:
    """What the window shows, apart from the session itself."""

    zoom: bool = False
    help: bool = False
    quit: bool = False
    save_requested: bool = False
    message: str = ""
    message_until: float = 0.0

    def set_message(self, text: str, seconds: float = MESSAGE_SECONDS) -> None:
        self.message = text
        self.message_until = time.monotonic() + seconds

    def visible_message(self) -> str:
        return self.message if time.monotonic() < self.message_until else ""


def describe_edit(edit: Edit) -> str:
    """One-line description of an edit, e.g. ``"add L HS at 120 (moved from 121)"``."""
    if edit.kind == "add":
        new = edit.added[0]
        text = f"add {new.code} at {new.frame}"
        if edit.removed:
            text += " (moved from " + ", ".join(str(e.frame) for e in edit.removed) + ")"
        return text
    return "delete " + ", ".join(f"{e.code} at {e.frame}" for e in edit.removed)


def handle_action(session: LabelSession, view: ViewState, action: str, *, zoom_available: bool = False) -> str:
    """Apply one action from :data:`KEY_BINDINGS` to the session or view.

    Parameters
    ----------
    session:
        The labelling session to change.
    view:
        Display state; ``zoom``, ``help``, ``save_requested`` and ``quit`` are set here.
    action:
        An action string such as ``"step:-10"`` or ``"add:L:HS"``.
    zoom_available:
        Whether keypoints were loaded, so the foot zoom can work.

    Returns
    -------
    str
        A short status message for the screen (empty if nothing worth saying).

    Raises
    ------
    ValueError
        For an unknown action.
    """
    if action.startswith("step:"):
        before = session.current
        after = session.step(int(action.split(":", 1)[1]))
        if after == before:
            return "start of clip" if after == 0 else "end of clip"
        return ""
    if action == "next_event":
        return "" if session.next_event() is not None else "no labelled events after this frame"
    if action == "prev_event":
        return "" if session.prev_event() is not None else "no labelled events before this frame"
    if action.startswith("add:"):
        _, side, event = action.split(":")
        edit = session.add(side, event)
        if edit is None:
            return f"{side} {event} is already on frame {session.current}"
        return describe_edit(edit)
    if action == "delete":
        removed = session.remove_nearest(DELETE_DISTANCE_FRAMES)
        if removed is None:
            return f"no event within {DELETE_DISTANCE_FRAMES} frames"
        return f"deleted {removed.code} at {removed.frame}"
    if action in ("undo", "redo"):
        edit = session.undo() if action == "undo" else session.redo()
        if edit is None:
            return f"nothing to {action}"
        # Jump to the change so the rater can see what was undone.
        session.goto(edit.frame)
        return f"{action}: {describe_edit(edit)}"
    if action == "zoom":
        if not zoom_available:
            view.zoom = False
            return "zoom needs keypoints (run with keypoints_path)"
        view.zoom = not view.zoom
        return "zoom on" if view.zoom else "zoom off"
    if action == "help":
        view.help = not view.help
        return ""
    if action == "save":
        view.save_requested = True
        return ""
    if action == "quit":
        view.quit = True
        return ""
    raise ValueError(f"Unknown action {action!r}")


# ---------------------------------------------------------------------------
# Frame reading
# ---------------------------------------------------------------------------


class FrameSource(Protocol):
    """Something that decodes frames in order and can be repositioned."""

    def seek(self, index: int) -> None:
        """Position so that the next :meth:`read` returns frame ``index``."""

    def read(self) -> np.ndarray | None:
        """Decode the next frame (None at the end or on error)."""


class FrameCache:
    """Random access to video frames by index on top of a sequential decoder.

    Stepping forward reads the next frame (fast). A short jump forward reads
    through the gap instead of seeking. Any other request seeks to a little
    before the target and decodes forward, caching those frames, so stepping
    backwards afterwards is served from memory instead of seeking every time.

    Parameters
    ----------
    source:
        A :class:`FrameSource`.
    n_frames:
        Number of frames in the video.
    size:
        Frames kept in memory (a 1080p frame is about 6 MB).
    backfill:
        Frames decoded before a seek target.
    max_skip:
        Largest forward gap read through rather than seeking.
    """

    def __init__(self, source: FrameSource, n_frames: int, *, size: int = 32, backfill: int = 12, max_skip: int = 30) -> None:
        if size < backfill + 1:
            raise ValueError("size must exceed backfill")
        self._source = source
        self.n_frames = int(n_frames)
        self.size = int(size)
        self.backfill = int(backfill)
        self.max_skip = int(max_skip)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._next: int | None = None
        self.n_seeks = 0
        self.n_decoded = 0

    def get(self, index: int) -> np.ndarray | None:
        """Frame ``index`` (BGR array), or None if it cannot be decoded."""
        index = int(index)
        if not 0 <= index < self.n_frames:
            return None
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        if self._next is None or not 0 <= index - self._next <= self.max_skip:
            start = max(0, index - self.backfill)
            self._source.seek(start)
            self.n_seeks += 1
            self._next = start
        while self._next is not None and self._next <= index:
            img = self._source.read()
            if img is None:
                self._next = None
                break
            self.n_decoded += 1
            self._put(self._next, img)
            self._next += 1
        return self._cache.get(index)

    def _put(self, index: int, img: np.ndarray) -> None:
        self._cache[index] = img
        self._cache.move_to_end(index)
        while len(self._cache) > self.size:
            self._cache.popitem(last=False)


class _VideoSource:
    """OpenCV decoder that checks its seeks against known frame times."""

    def __init__(self, cv2: Any, path: Path, container_times: np.ndarray | None = None) -> None:
        self._cv2 = cv2
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV cannot open {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.exact_seek = True
        self._times = None
        if container_times is not None and len(container_times) > 2:
            self._times = np.asarray(container_times, dtype=float)
            self._tol_s = 0.5 * float(np.median(np.diff(self._times)))
        # Read frame 0 once to learn its time, which OpenCV may not report as 0.
        ok, img = self.cap.read()
        self._pending = img if ok else None
        self._t0_ms = float(self.cap.get(cv2.CAP_PROP_POS_MSEC)) if ok else 0.0

    def read(self) -> np.ndarray | None:
        if self._pending is not None:
            img, self._pending = self._pending, None
            return img
        ok, img = self.cap.read()
        return img if ok else None

    def seek(self, index: int) -> None:
        cv2 = self._cv2
        self._pending = None
        if index <= 0 or not self.exact_seek:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for _ in range(index):
                if not self.cap.grab():
                    break
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        if self._times is None or index >= len(self._times):
            return
        ok, img = self.cap.read()
        if not ok:
            return
        actual = (float(self.cap.get(cv2.CAP_PROP_POS_MSEC)) - self._t0_ms) / 1000.0
        expected = float(self._times[index] - self._times[0])
        if abs(actual - expected) <= self._tol_s:
            self._pending = img
            return
        # Seek landed on another frame (typical of VFR files): decode from the
        # start from now on. Slower, but frame indices stay exact.
        warnings.warn(
            f"Seeking is inexact for this video (asked for frame {index} at {expected:.3f} s, got {actual:.3f} s); "
            "jumps will decode from the start, which is slower. A constant-frame-rate copy seeks faster.",
            stacklevel=2,
        )
        self.exact_seek = False
        self.seek(index)

    def release(self) -> None:
        self.cap.release()


# ---------------------------------------------------------------------------
# Geometry helpers (pure)
# ---------------------------------------------------------------------------


def foot_crop_box(
    kp: Keypoints,
    frame: int,
    image_size: tuple[int, int],
    *,
    zoom: float = 2.0,
    window: int = 3,
    min_conf: float = 0.1,
    margin: float = 0.25,
) -> tuple[int, int, int, int] | None:
    """Crop rectangle around both feet, with the frame's aspect ratio.

    Parameters
    ----------
    kp:
        Keypoints of the clip (only foot points are used).
    frame:
        Source-video frame index.
    image_size:
        ``(width, height)`` of the video frames. Keypoints are rescaled if
        ``kp.meta`` records another size.
    zoom:
        Magnification when the feet fit: the box is ``1/zoom`` of the frame.
        It grows (less zoom) if the feet are further apart than that.
    window:
        Use foot points from frames within this many frames of ``frame``, and
        their median as the centre, so the box does not jitter while stepping
        or vanish on a frame with a dropped point.
    min_conf:
        Ignore points with lower confidence.
    margin:
        Extra space around the feet, as a fraction of their extent.

    Returns
    -------
    tuple of int or None
        ``(x0, y0, x1, y1)`` in pixels, inside the image; None if no foot
        point is available near ``frame``.
    """
    width, height = int(image_size[0]), int(image_size[1])
    near = np.abs(kp.frame_idx - int(frame)) <= window
    if not near.any():
        return None
    pts = []
    for side in SIDES:
        for part in FOOT_PARTS:
            name = SIDE_PREFIX[side] + part
            if kp.has(name):
                pts.append(kp.masked_point(name, min_conf)[near])
    if not pts:
        return None
    p = np.concatenate(pts, axis=0)
    p = p[np.all(np.isfinite(p), axis=1)]
    if p.size == 0:
        return None
    kw, kh = kp.meta.get("width"), kp.meta.get("height")
    if kw and kh and (int(kw), int(kh)) != (width, height):
        p = p * np.array([width / float(kw), height / float(kh)])
    cx, cy = np.median(p, axis=0)
    span_x = float(np.ptp(p[:, 0])) * (1.0 + 2.0 * margin)
    span_y = float(np.ptp(p[:, 1])) * (1.0 + 2.0 * margin)
    scale = max(1.0 / zoom, span_x / width, span_y / height)
    scale = min(scale, 1.0)
    bw, bh = scale * width, scale * height
    x0 = float(np.clip(cx - bw / 2.0, 0.0, width - bw))
    y0 = float(np.clip(cy - bh / 2.0, 0.0, height - bh))
    x0i, y0i = int(round(x0)), int(round(y0))
    x1i, y1i = min(width, x0i + int(round(bw))), min(height, y0i + int(round(bh)))
    return x0i, y0i, x1i, y1i


def frame_to_x(frame: float, n_frames: int, x0: int, x1: int) -> int:
    """Pixel column of ``frame`` on a timeline drawn from ``x0`` to ``x1``."""
    if n_frames <= 1:
        return x0
    return int(round(x0 + (x1 - x0) * float(frame) / (n_frames - 1)))


def x_to_frame(x: float, n_frames: int, x0: int, x1: int) -> int:
    """Frame under pixel column ``x`` of a timeline drawn from ``x0`` to ``x1`` (clamped)."""
    if n_frames <= 1 or x1 <= x0:
        return 0
    f = round((float(x) - x0) / (x1 - x0) * (n_frames - 1))
    return int(min(max(f, 0), n_frames - 1))


@dataclass(frozen=True)
class _Layout:
    width: int
    image_h: int
    panel_h: int
    x0: int
    x1: int
    whole_y: tuple[int, int]
    local_y: tuple[int, int]
    font: float


def _layout(width: int, image_h: int) -> _Layout:
    font = float(np.clip(width / 1280.0 * 0.55, 0.4, 0.8))
    strip = int(round(24 * font / 0.55))
    panel_h = 3 * strip + 24
    x0, x1 = 12, width - 12
    whole = (image_h + 8, image_h + 8 + strip)
    local = (whole[1] + 16 + strip // 2, whole[1] + 16 + strip // 2 + strip)
    return _Layout(width, image_h, panel_h, x0, x1, whole, local, font)


def canvas_layout(shape: tuple[int, ...]) -> _Layout:
    """Layout of a canvas returned by :func:`render`, from its shape (for mouse clicks)."""
    height, width = int(shape[0]), int(shape[1])
    return _layout(width, height - _layout(width, 0).panel_h)


def display_size(frame_size: tuple[int, int], max_size: tuple[int, int] = (1280, 720)) -> tuple[int, int]:
    """Size to show a ``(width, height)`` frame at so it fits ``max_size`` (never enlarged)."""
    w, h = frame_size
    s = min(1.0, max_size[0] / float(w), max_size[1] / float(h))
    return max(1, int(round(w * s))), max(1, int(round(h * s)))


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def render(
    session: LabelSession,
    image: np.ndarray | None,
    view: ViewState,
    *,
    crop: tuple[int, int, int, int] | None = None,
    max_size: tuple[int, int] = (1280, 720),
    frame_size: tuple[int, int] | None = None,
    note: str = "",
) -> np.ndarray:
    """Draw the frame with the info bar, nearby labels, timelines and help.

    Parameters
    ----------
    session:
        Labels and current frame.
    image:
        BGR frame at full resolution, or None if it could not be decoded.
    view:
        Zoom/help flags and the status message.
    crop:
        ``(x0, y0, x1, y1)`` to magnify (see :func:`foot_crop_box`).
    max_size:
        Largest ``(width, height)`` for the picture area.
    frame_size:
        ``(width, height)`` of the video, used when ``image`` is None.
    note:
        Persistent warning shown under the info bar (e.g. nominal timestamps).

    Returns
    -------
    numpy.ndarray
        The BGR canvas to pass to ``cv2.imshow``: the picture plus a timeline
        panel below it.
    """
    cv2 = _import_cv2()
    if image is None:
        fw, fh = frame_size or (1280, 720)
        image = np.zeros((fh, fw, 3), np.uint8)
        _text(cv2, image, f"frame {session.current} could not be decoded", (20, fh // 2), 1.0, (0, 0, 255))
    full_h, full_w = image.shape[:2]
    dw, dh = display_size((full_w, full_h), max_size)
    if crop is not None:
        x0, y0, x1, y1 = crop
        image = image[y0:y1, x0:x1]
    interp = cv2.INTER_LINEAR if crop is not None else cv2.INTER_AREA
    picture = cv2.resize(image, (dw, dh), interpolation=interp) if image.shape[:2] != (dh, dw) else image.copy()
    if picture.ndim == 2:
        picture = cv2.cvtColor(picture, cv2.COLOR_GRAY2BGR)

    lay = _layout(dw, dh)
    canvas = np.zeros((dh + lay.panel_h, dw, 3), np.uint8)
    canvas[:dh] = picture
    canvas[dh:] = (32, 32, 32)
    fs = lay.font
    line = int(round(26 * fs / 0.55))

    # Info bar.
    bar_h = 2 * line + 8 + (line if note else 0)
    _shade(cv2, canvas, (0, 0, dw, bar_h), 0.55)
    counts = "  ".join(f"{k} {v}" for k, v in session.counts().items())
    flags = ("  ZOOM" if view.zoom else "") + ("  [unsaved]" if session.dirty else "")
    _text(cv2, canvas, f"{session.clip_id}   frame {session.current} / {session.n_frames - 1}   "
          f"t = {session.current_time:.4f} s   rater {session.rater}", (10, line), fs, (255, 255, 255))
    _text(cv2, canvas, f"{counts}{flags}   h = help", (10, 2 * line), fs, (210, 210, 210))
    if note:
        _text(cv2, canvas, note, (10, 3 * line), fs, (80, 200, 255))

    # Labels near the current frame, top right.
    near = session.events_near(window=_local_halfwidth(session))
    rows = near[:12]
    if rows:
        col_w = int(round(190 * fs / 0.55))
        top = bar_h + 6
        _shade(cv2, canvas, (dw - col_w - 10, top, dw - 4, top + line * len(rows) + 8), 0.55)
        for i, e in enumerate(rows):
            d = e.frame - session.current
            mark = ">" if d == 0 else " "
            txt = f"{mark}{e.code}  {e.frame} ({d:+d})"
            colour = (255, 255, 255) if d == 0 else SIDE_COLOURS[e.side]
            _text(cv2, canvas, txt, (dw - col_w - 4, top + line * (i + 1)), fs, colour, 2 if d == 0 else 1)
    here = session.events_at()
    if here:
        label = "  ".join(e.code for e in here)
        _text(cv2, canvas, label, (10, dh - 16), fs * 1.6, (255, 255, 255), 3)

    message = view.visible_message()
    if message:
        size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0]
        x = max(10, (dw - size[0]) // 2)
        _shade(cv2, canvas, (x - 8, dh - 2 * line - 20, x + size[0] + 8, dh - line - 10), 0.6)
        _text(cv2, canvas, message, (x, dh - line - 18), fs, (120, 255, 120))

    _draw_timelines(cv2, canvas, session, lay)

    if view.help:
        _draw_help(cv2, canvas, dh, fs, line)
    return canvas


def _local_halfwidth(session: LabelSession) -> int:
    """Half-width of the local timeline: about 0.75 s, at least 20 frames."""
    if session.n_frames < 2:
        return 20
    dt = float(np.median(np.diff(session.timestamps)))
    return int(max(20, round(0.75 / dt))) if dt > 0 else 20


def _draw_timelines(cv2: Any, canvas: np.ndarray, session: LabelSession, lay: _Layout) -> None:
    n = session.n_frames
    # Whole clip.
    y0, y1 = lay.whole_y
    cv2.rectangle(canvas, (lay.x0, y0), (lay.x1, y1), (70, 70, 70), -1)
    for e in session.events:
        x = frame_to_x(e.frame, n, lay.x0, lay.x1)
        top = y0 if e.event == "HS" else (y0 + y1) // 2
        cv2.line(canvas, (x, top), (x, y1), SIDE_COLOURS[e.side], 2)
    xc = frame_to_x(session.current, n, lay.x0, lay.x1)
    cv2.line(canvas, (xc, y0 - 4), (xc, y1 + 4), (255, 255, 255), 2)

    # Local window around the current frame.
    half = _local_halfwidth(session)
    lo, hi = session.current - half, session.current + half
    ly0, ly1 = lay.local_y
    cv2.rectangle(canvas, (lay.x0, ly0), (lay.x1, ly1), (55, 55, 55), -1)
    span = hi - lo
    px_per_frame = (lay.x1 - lay.x0) / float(span)
    tick_every = 1 if px_per_frame >= 5 else (5 if px_per_frame >= 1 else 25)
    for f in range(max(lo, 0), min(hi, n - 1) + 1):
        if f % tick_every == 0:
            x = int(round(lay.x0 + (f - lo) * px_per_frame))
            tall = f % (tick_every * 10) == 0
            cv2.line(canvas, (x, ly1 - (8 if tall else 4)), (x, ly1), (130, 130, 130), 1)
    for e in session.events_near(window=half):
        x = int(round(lay.x0 + (e.frame - lo) * px_per_frame))
        top = ly0 if e.event == "HS" else (ly0 + ly1) // 2
        cv2.line(canvas, (x, top), (x, ly1), SIDE_COLOURS[e.side], 3)
        _text(cv2, canvas, e.code.replace(" ", ""), (x + 3, ly0 - 3), lay.font * 0.75, SIDE_COLOURS[e.side])
    xc = int(round(lay.x0 + (session.current - lo) * px_per_frame))
    cv2.line(canvas, (xc, ly0 - 2), (xc, ly1 + 2), (255, 255, 255), 1)


def _draw_help(cv2: Any, canvas: np.ndarray, image_h: int, fs: float, line: int) -> None:
    h, w = canvas.shape[:2]
    box_w = min(w - 40, int(round(620 * fs / 0.55)))
    box_h = line * (len(HELP_LINES) + 1) + 10
    x0 = (w - box_w) // 2
    y0 = max(10, (image_h - box_h) // 2)
    _shade(cv2, canvas, (x0, y0, x0 + box_w, y0 + box_h), 0.8)
    key_col = int(round(130 * fs / 0.55))
    for i, text in enumerate(HELP_LINES):
        y = y0 + line * (i + 1)
        # The font is proportional, so split "keys   description" into two columns.
        parts = text.split("  ", 1)
        if len(parts) == 2 and parts[0].strip():
            _text(cv2, canvas, parts[0].strip(), (x0 + 16, y), fs, (120, 220, 255))
            _text(cv2, canvas, parts[1].strip(), (x0 + 16 + key_col, y), fs, (255, 255, 255))
        else:
            _text(cv2, canvas, text, (x0 + 16, y), fs, (255, 255, 255))


def _shade(cv2: Any, canvas: np.ndarray, rect: tuple[int, int, int, int], alpha: float) -> None:
    x0, y0, x1, y1 = (int(v) for v in rect)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(canvas.shape[1], x1), min(canvas.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = canvas[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = (roi * (1.0 - alpha)).astype(np.uint8)


def _text(cv2: Any, img: np.ndarray, text: str, org: tuple[int, int], scale: float,
          colour: tuple[int, int, int], thickness: int = 1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Timestamps and files
# ---------------------------------------------------------------------------


def load_timestamps(
    video_path: str | Path,
    timestamps_csv: str | Path | None = None,
) -> tuple[np.ndarray | None, str, np.ndarray | None]:
    """Find the best per-frame times for a video.

    Parameters
    ----------
    video_path:
        The video being labelled.
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
    video_path = Path(video_path)
    from ..capture.realsense import frame_times_path, read_frame_times

    csv = Path(timestamps_csv) if timestamps_csv is not None else frame_times_path(video_path)
    if timestamps_csv is not None or csv.is_file():
        table = read_frame_times(csv)
        return table["time_s"].to_numpy(dtype=float), f"realsense:{csv.name}", None
    try:
        from ..video import frame_timestamps
    except ImportError:
        return None, "nominal", None
    try:
        ts = frame_timestamps(video_path)
    except (RuntimeError, OSError, ValueError) as exc:
        warnings.warn(f"Could not read frame times of {video_path} ({exc}); using frame / fps", stacklevel=2)
        return None, "nominal", None
    return np.asarray(ts, dtype=float), "video", np.asarray(ts, dtype=float)


def autosave_path(out_csv: str | Path) -> Path:
    """Where autosaves of ``out_csv`` go: ``<out_csv>.autosave``."""
    return Path(str(out_csv) + ".autosave")


def sidecar_path(out_csv: str | Path) -> Path:
    """The JSON written next to a labels file: same name, ``.json`` suffix."""
    return Path(out_csv).with_suffix(".json")


def _write_sidecar(path: Path, session: LabelSession, info: dict[str, Any]) -> None:
    record = {
        "clip_id": session.clip_id,
        "rater": session.rater,
        "n_frames": session.n_frames,
        "n_events": session.n_events,
        "counts": session.counts(),
        "sequence_issues": session.sequence_issues(),
        "blind": True,
        "gaitbench_version": __version__,
        "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **info,
    }
    path.write_text(json.dumps(record, indent=2))


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The labeller needs OpenCV with GUI support: pip install opencv-python "
            "(not opencv-python-headless)"
        ) from exc
    return cv2


def _window_visible(cv2: Any, name: str) -> bool | None:
    """True/False if the backend reports the window's visibility, None if it cannot."""
    try:
        v = cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE)
    except Exception:  # noqa: BLE001 - some backends raise once the window is gone
        return False
    if v is None or v < 0:
        return None
    return v >= 1


# ---------------------------------------------------------------------------
# The GUI
# ---------------------------------------------------------------------------


def run_labeller(
    video_path: str | Path,
    out_csv: str | Path,
    rater: str,
    clip_id: str | None = None,
    keypoints_path: str | Path | None = None,
    start_frame: int = 0,
    *,
    timestamps_csv: str | Path | None = None,
    cache_frames: int = 32,
    max_size: tuple[int, int] = (1280, 720),
) -> Path | None:
    """Open the labelling window for one video and one rater.

    Parameters
    ----------
    video_path:
        Video to label (the original file, so frame numbers match the pose stage).
    out_csv:
        Labels file, normally ``gaitbench.formats.labels_path(root, clip_id, rater)``.
        If it exists, labelling resumes from it.
    rater:
        Rater's name or initials.
    clip_id:
        Clip name; defaults to the keypoints' clip id, else the video file stem.
    keypoints_path:
        Optional keypoints ``.npz`` of this video, used only to place the
        foot zoom (``z``). Keypoints are never drawn.
    start_frame:
        Frame shown first.
    timestamps_csv:
        Per-frame times from :func:`gaitbench.capture.realsense.bag_to_video`
        (found automatically next to the video).
    cache_frames:
        Decoded frames kept in memory.
    max_size:
        Largest picture size on screen; frames are shrunk to fit.

    Returns
    -------
    pathlib.Path or None
        ``out_csv`` if it holds labels when the window closes, else None.

    Raises
    ------
    FileNotFoundError
        If the video does not exist.
    RuntimeError
        If OpenCV cannot open the video or a window, or (macOS) if called
        off the main thread.
    gaitbench.formats.FormatError
        If ``out_csv`` belongs to another clip or rater.
    """
    cv2 = _import_cv2()
    if sys.platform == "darwin" and threading.current_thread() is not threading.main_thread():
        raise RuntimeError("On macOS the labeller must run on the main thread (OpenCV/Cocoa windows)")
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    out_csv = Path(out_csv)
    kp = Keypoints.load(keypoints_path) if keypoints_path is not None else None
    if clip_id is None:
        clip_id = kp.clip_id if kp is not None else video_path.stem
    if kp is not None and kp.clip_id != clip_id:
        warnings.warn(f"Keypoints are for clip {kp.clip_id!r}, labelling {clip_id!r}", stacklevel=2)

    print(f"Reading frame times of {video_path.name} ...", flush=True)
    times, time_source, container = load_timestamps(video_path, timestamps_csv)
    source = _VideoSource(cv2, video_path, container_times=container)
    note = ""
    if times is None:
        if source.frame_count <= 0 or source.fps <= 0:
            raise RuntimeError(f"{video_path}: OpenCV reports no frame count or fps, and no frame times are available")
        times = np.arange(source.frame_count) / source.fps
        note = "times are frame/fps (no ffprobe): wrong for variable-frame-rate video"
    elif source.frame_count and abs(source.frame_count - len(times)) > 1:
        print(f"Note: OpenCV reports {source.frame_count} frames, the frame times list {len(times)}; using the latter.")

    session = LabelSession(clip_id, rater, timestamps=times, start_frame=start_frame)
    autosave = autosave_path(out_csv)
    force_save = False
    resume_from = None
    if autosave.is_file() and (not out_csv.is_file() or autosave.stat().st_mtime > out_csv.stat().st_mtime):
        resume_from, force_save = autosave, True
    elif out_csv.is_file():
        resume_from = out_csv
    if resume_from is not None:
        n = session.load(resume_from)
        print(f"Resumed {n} labels from {resume_from}")

    sidecar_info = {
        "video": str(video_path),
        "timestamps_source": time_source,
        "keypoints": str(keypoints_path) if keypoints_path is not None else None,
    }
    frames = FrameCache(source, session.n_frames, size=max(cache_frames, 13))
    view = ViewState()
    window = f"gaitbench label - {clip_id} - {rater}"
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_KEEPRATIO", 0))
    except cv2.error as exc:
        source.release()
        raise RuntimeError(
            f"OpenCV cannot open a window ({exc}). Install opencv-python (with GUI) instead of "
            "opencv-python-headless."
        ) from exc

    clicks: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    cv2.setMouseCallback(window, on_mouse)
    print("Labeller open. Press h in the window for keys; q or Esc saves and quits.")
    if note:
        print("Warning:", note)

    def save_main() -> None:
        session.save(out_csv)
        _write_sidecar(sidecar_path(out_csv), session, sidecar_info)
        issues = session.sequence_issues()
        view.set_message(f"saved {session.n_events} events to {out_csv.name}"
                         + (f"  ({len(issues)} sequence warnings)" if issues else ""))
        for issue in issues:
            print("Check:", issue)

    last_autosave = session.n_edits
    seen_visible = False
    redraw = True
    shown_message = ""
    layout: _Layout | None = None
    try:
        while not view.quit:
            msg_now = view.visible_message()
            if msg_now != shown_message:
                redraw = True
            if redraw:
                img = frames.get(session.current)
                crop = None
                if view.zoom and kp is not None and img is not None:
                    crop = foot_crop_box(kp, session.current, (img.shape[1], img.shape[0]))
                canvas = render(session, img, view, crop=crop, max_size=max_size,
                                frame_size=_frame_size(source), note=note)
                layout = canvas_layout(canvas.shape)
                cv2.imshow(window, canvas)
                shown_message = view.visible_message()
                redraw = False
            code = cv2.waitKeyEx(30)
            visible = _window_visible(cv2, window)
            if visible:
                seen_visible = True
            elif visible is False and seen_visible:
                break  # window closed with the mouse: save below
            while clicks and layout is not None:
                x, y = clicks.pop(0)
                if layout.whole_y[0] - 6 <= y <= layout.whole_y[1] + 6:
                    session.goto(x_to_frame(x, session.n_frames, layout.x0, layout.x1))
                    redraw = True
            key = decode_key(code)
            if key is None:
                continue
            action = KEY_BINDINGS.get(key)
            if action is None:
                view.set_message(f"key '{key}' does nothing (h for help)")
                redraw = True
                continue
            message = handle_action(session, view, action, zoom_available=kp is not None)
            if message:
                view.set_message(message)
            if view.zoom and kp is not None and action == "zoom":
                img = frames.get(session.current)
                if img is not None and foot_crop_box(kp, session.current, (img.shape[1], img.shape[0])) is None:
                    view.set_message("no foot keypoints near this frame; showing the whole frame")
            if view.save_requested:
                view.save_requested = False
                save_main()
                force_save = False
            if session.n_edits - last_autosave >= AUTOSAVE_EVERY:
                session.save(autosave, mark_saved=False)
                last_autosave = session.n_edits
            redraw = True
        if session.dirty or force_save:
            save_main()
            print(f"Saved {session.n_events} labels to {out_csv}")
        if autosave.is_file() and out_csv.is_file():
            autosave.unlink()
    except BaseException:
        if session.dirty:
            session.save(autosave, mark_saved=False)
            print(f"Labeller stopped by an error; unsaved labels written to {autosave}", file=sys.stderr)
        raise
    finally:
        source.release()
        try:
            cv2.destroyWindow(window)
            cv2.waitKey(1)  # lets the GUI backend actually close the window (macOS, GTK)
        except Exception:  # noqa: BLE001 - closing a window that is already gone
            pass
    return out_csv if out_csv.is_file() else None


def _frame_size(source: _VideoSource) -> tuple[int, int]:
    cv2 = source._cv2
    w = int(source.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(source.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    return w, h

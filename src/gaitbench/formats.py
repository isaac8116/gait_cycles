"""File formats shared by every stage of the harness.

Three kinds of file move between stages:

* **Keypoints** (``keypoints/<clip>__<backend>.npz``): one pose backend's output
  for one clip, in source-image pixels.
* **Events** (``events/<clip>__<backend>__<detector>.csv``): heel strikes (HS)
  and toe offs (TO) found by one detector on one keypoints file.
* **Labels** (``labels/<clip>__<rater>.csv``): the same columns as events,
  written by a person (``tool = "human"``) or by a simulator
  (``tool = "synthetic"``). Labels are the ground truth the scorer uses.

Conventions used everywhere:

* Image coordinates are pixels with the origin at the top-left corner and
  y pointing down, exactly as the video stores them.
* Keypoint names follow MediaPipe (also what myogait uses), plus the toe
  points that OpenPose, RTMW and Sapiens provide.
* ``frame`` is the index of the frame in the source video, and ``time_s`` is
  that frame's presentation time. Times are stored per frame because phone
  video (iPhone in particular) can have a variable frame rate.
* Side ``"L"``/``"R"`` is the walker's own left/right, never the image side.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

CANONICAL_KEYPOINTS: tuple[str, ...] = (
    "NOSE",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
    "LEFT_BIG_TOE", "RIGHT_BIG_TOE",
    "LEFT_SMALL_TOE", "RIGHT_SMALL_TOE",
)

SIDES: tuple[str, ...] = ("L", "R")
EVENT_TYPES: tuple[str, ...] = ("HS", "TO")
SIDE_PREFIX: dict[str, str] = {"L": "LEFT_", "R": "RIGHT_"}

EVENT_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "tool",
    "tool_version",
    "backend",
    "detector",
    "side",
    "event",
    "frame",
    "time_s",
    "confidence",
)

_FILENAME_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class FormatError(ValueError):
    """A file or object does not follow the harness formats."""


# ---------------------------------------------------------------------------
# Keypoints
# ---------------------------------------------------------------------------


@dataclass
class Keypoints:
    """One pose backend's keypoints for one clip.

    Attributes
    ----------
    clip_id:
        Name of the clip, shared by every file derived from it.
    names:
        Keypoint names, a subset of :data:`CANONICAL_KEYPOINTS` in any order.
    xy:
        ``(n_frames, n_keypoints, 2)`` float array of pixel coordinates.
        Missing points are NaN.
    conf:
        ``(n_frames, n_keypoints)`` confidence in [0, 1]; 0 where missing.
    time_s:
        ``(n_frames,)`` presentation time of each frame, in seconds.
    frame_idx:
        ``(n_frames,)`` index of each frame in the source video.
    meta:
        Free-form metadata. Expected keys: ``fps`` (nominal), ``width``,
        ``height``, ``source`` (video path), ``backend``, ``tool``,
        ``tool_version``. Must be JSON-serialisable.
    """

    clip_id: str
    names: tuple[str, ...]
    xy: np.ndarray
    conf: np.ndarray
    time_s: np.ndarray
    frame_idx: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.names = tuple(str(n) for n in self.names)
        self.xy = np.asarray(self.xy, dtype=float)
        self.conf = np.asarray(self.conf, dtype=float)
        self.time_s = np.asarray(self.time_s, dtype=float)
        self.frame_idx = np.asarray(self.frame_idx, dtype=np.int64)
        self.meta = dict(self.meta)
        self.validate()

    # -- shape and content checks ------------------------------------------

    def validate(self) -> None:
        """Raise :class:`FormatError` if the arrays are inconsistent."""
        unknown = [n for n in self.names if n not in CANONICAL_KEYPOINTS]
        if unknown:
            raise FormatError(f"Unknown keypoint names: {unknown}")
        if len(set(self.names)) != len(self.names):
            raise FormatError("Duplicate keypoint names")
        n_frames, n_kp = len(self.time_s), len(self.names)
        if self.xy.shape != (n_frames, n_kp, 2):
            raise FormatError(f"xy has shape {self.xy.shape}, expected {(n_frames, n_kp, 2)}")
        if self.conf.shape != (n_frames, n_kp):
            raise FormatError(f"conf has shape {self.conf.shape}, expected {(n_frames, n_kp)}")
        if self.frame_idx.shape != (n_frames,):
            raise FormatError(f"frame_idx has shape {self.frame_idx.shape}, expected {(n_frames,)}")
        if n_frames > 1 and np.any(np.diff(self.time_s) <= 0):
            raise FormatError("time_s must be strictly increasing")
        if n_frames > 1 and np.any(np.diff(self.frame_idx) <= 0):
            raise FormatError("frame_idx must be strictly increasing")
        if np.any((self.conf < 0) | (self.conf > 1)):
            raise FormatError("conf must lie in [0, 1]")
        try:
            json.dumps(self.meta)
        except TypeError as exc:
            raise FormatError(f"meta is not JSON-serialisable: {exc}") from exc

    # -- access ------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        return len(self.time_s)

    @property
    def fps(self) -> float:
        """Nominal frame rate: ``meta['fps']`` if set, else from the timestamps."""
        if self.meta.get("fps"):
            return float(self.meta["fps"])
        if self.n_frames > 1:
            return float(1.0 / np.median(np.diff(self.time_s)))
        raise FormatError("Cannot infer fps from a single frame and no meta['fps']")

    def has(self, name: str) -> bool:
        return name in self.names

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"Keypoint {name!r} not in this file ({self.clip_id})") from exc

    def point(self, name: str) -> np.ndarray:
        """``(n_frames, 2)`` pixel coordinates of one keypoint (NaN where missing)."""
        return self.xy[:, self.index(name), :]

    def confidence(self, name: str) -> np.ndarray:
        return self.conf[:, self.index(name)]

    def side_point(self, side: str, part: str) -> np.ndarray:
        """Coordinates of e.g. ``side_point('L', 'HEEL')`` -> ``LEFT_HEEL``."""
        return self.point(SIDE_PREFIX[side] + part)

    def masked_point(self, name: str, min_conf: float = 0.0) -> np.ndarray:
        """Coordinates with points below ``min_conf`` set to NaN."""
        pt = self.point(name).copy()
        pt[self.confidence(name) < min_conf] = np.nan
        return pt

    def subset(self, names: Iterable[str]) -> "Keypoints":
        names = tuple(names)
        idx = [self.index(n) for n in names]
        return Keypoints(
            clip_id=self.clip_id,
            names=names,
            xy=self.xy[:, idx, :],
            conf=self.conf[:, idx],
            time_s=self.time_s,
            frame_idx=self.frame_idx,
            meta=self.meta,
        )

    # -- files -------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            clip_id=np.array(self.clip_id),
            names=np.array(self.names),
            xy=self.xy,
            conf=self.conf,
            time_s=self.time_s,
            frame_idx=self.frame_idx,
            meta_json=np.array(json.dumps(self.meta)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Keypoints":
        with np.load(Path(path), allow_pickle=False) as z:
            missing = {"clip_id", "names", "xy", "conf", "time_s", "frame_idx", "meta_json"} - set(z.files)
            if missing:
                raise FormatError(f"{path}: missing arrays {sorted(missing)}")
            return cls(
                clip_id=str(z["clip_id"]),
                names=tuple(str(n) for n in z["names"]),
                xy=z["xy"],
                conf=z["conf"],
                time_s=z["time_s"],
                frame_idx=z["frame_idx"],
                meta=json.loads(str(z["meta_json"])),
            )


# ---------------------------------------------------------------------------
# Events and labels
# ---------------------------------------------------------------------------


def empty_events() -> pd.DataFrame:
    return validate_events(pd.DataFrame({c: [] for c in EVENT_COLUMNS}))


def make_events(
    records: Iterable[Mapping[str, Any]],
    *,
    defaults: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build an events table from dict records, filling missing columns from ``defaults``.

    Each record needs at least ``side``, ``event`` and ``frame`` or ``time_s``.
    """
    defaults = dict(defaults or {})
    rows = [{**defaults, **dict(r)} for r in records]
    if not rows:
        df = empty_events()
        for col, val in defaults.items():
            if col in df.columns:
                df[col] = df[col].astype(object)
        return df
    df = pd.DataFrame(rows)
    for col in EVENT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan if col in ("frame", "time_s", "confidence") else ""
    return validate_events(df)


def validate_events(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the standard columns, dtypes and order; raise on bad values."""
    missing = [c for c in EVENT_COLUMNS if c not in df.columns]
    if missing:
        raise FormatError(f"Events table is missing columns {missing}")
    out = df.loc[:, list(EVENT_COLUMNS)].copy()
    for col in ("clip_id", "tool", "tool_version", "backend", "detector", "side", "event"):
        out[col] = out[col].fillna("").astype(str)
    bad_side = sorted(set(out["side"]) - set(SIDES))
    if bad_side:
        raise FormatError(f"side must be one of {SIDES}, got {bad_side}")
    bad_event = sorted(set(out["event"]) - set(EVENT_TYPES))
    if bad_event:
        raise FormatError(f"event must be one of {EVENT_TYPES}, got {bad_event}")
    out["time_s"] = pd.to_numeric(out["time_s"], errors="raise").astype(float)
    frame = pd.to_numeric(out["frame"], errors="raise")
    if frame.isna().any():
        raise FormatError("frame must be set for every event")
    if not np.allclose(frame, np.round(frame)):
        raise FormatError("frame must be an integer frame index")
    out["frame"] = frame.round().astype(np.int64)
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").astype(float)
    return out.sort_values(["clip_id", "side", "event", "frame"], kind="stable").reset_index(drop=True)


_TEXT_COLUMNS = ("clip_id", "tool", "tool_version", "backend", "detector", "side", "event")


def read_events(path: str | Path) -> pd.DataFrame:
    # Read IDs as text so a clip or rater called "01" does not come back as 1.
    df = pd.read_csv(path, dtype={c: str for c in _TEXT_COLUMNS}, keep_default_na=False, na_values={
        "frame": [""], "time_s": [""], "confidence": ["", "nan", "NaN"],
    })
    return validate_events(df)


def write_events(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_events(df).to_csv(path, index=False)
    return path


# Labels are events written by a person or a simulator; the format is identical.
read_labels = read_events
write_labels = write_events


def frames_to_times(frames: Sequence[int] | np.ndarray, kp: Keypoints) -> np.ndarray:
    """Look up the presentation time of source-video frame indices in ``kp``.

    Frames between stored frames are interpolated; frames outside the stored
    range are extrapolated at the nominal frame rate.
    """
    frames = np.asarray(frames, dtype=float)
    if kp.n_frames == 0:
        raise FormatError("No frames to look up")
    if kp.n_frames == 1:
        return kp.time_s[0] + (frames - kp.frame_idx[0]) / kp.fps
    t = np.interp(frames, kp.frame_idx, kp.time_s)
    lo, hi = frames < kp.frame_idx[0], frames > kp.frame_idx[-1]
    t[lo] = kp.time_s[0] + (frames[lo] - kp.frame_idx[0]) / kp.fps
    t[hi] = kp.time_s[-1] + (frames[hi] - kp.frame_idx[-1]) / kp.fps
    return t


# ---------------------------------------------------------------------------
# File naming
# ---------------------------------------------------------------------------


def safe_name(name: str) -> str:
    """Make a detector or backend name safe for a filename (``myogait:gk_bike`` -> ``myogait-gk_bike``)."""
    cleaned = _FILENAME_UNSAFE.sub("-", str(name)).strip("-")
    if not cleaned:
        raise FormatError(f"Name {name!r} has no filename-safe characters")
    return cleaned


def keypoints_path(root: str | Path, clip_id: str, backend: str) -> Path:
    return Path(root) / "keypoints" / f"{safe_name(clip_id)}__{safe_name(backend)}.npz"


def events_path(root: str | Path, clip_id: str, backend: str, detector: str) -> Path:
    return Path(root) / "events" / f"{safe_name(clip_id)}__{safe_name(backend)}__{safe_name(detector)}.csv"


def labels_path(root: str | Path, clip_id: str, rater: str) -> Path:
    return Path(root) / "labels" / f"{safe_name(clip_id)}__{safe_name(rater)}.csv"

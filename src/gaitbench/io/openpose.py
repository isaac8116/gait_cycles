"""Read a directory of OpenPose ``*_keypoints.json`` files into harness keypoints.

OpenPose writes one JSON file per video frame, named
``<video>_<12-digit frame number>_keypoints.json``, each holding a ``people``
list with a flat ``pose_keypoints_2d`` array ``[x0, y0, c0, x1, y1, c1, ...]``
in source-image pixels (origin top-left, y down, the harness convention).
A point OpenPose did not find is written as ``(0, 0, 0)``.

Mapping to the harness names (:data:`gaitbench.formats.CANONICAL_KEYPOINTS`):

* BODY_25 ``RHeel``/``LHeel`` -> ``RIGHT_HEEL``/``LEFT_HEEL``, ``*BigToe`` ->
  ``*_BIG_TOE``, ``*SmallToe`` -> ``*_SMALL_TOE``; limbs and the nose map by
  name. ``Neck``, ``MidHip``, eyes and ears have no harness name and are
  dropped (the detectors locate the pelvis from the two hips).
* ``*_FOOT_INDEX`` is **not** written by default. MediaPipe's foot index is a
  point at the toe tip that OpenPose does not have; inventing it would make
  an OpenPose file look as if it carried a fourth foot point. The detectors
  fall back from ``FOOT_INDEX`` to ``BIG_TOE`` themselves and the myogait
  bridge derives ``FOOT_INDEX`` as the mean of the toes, so nothing needs it.
  ``foot_index="big_toe"`` or ``"mean_toes"`` writes it anyway, recorded in
  ``meta['foot_index_source']``.
* The older 18-point COCO model is read too; it has no feet.

OpenPose does not track people between frames (``person_id`` is -1 unless
tracking was switched on), so which ``people`` entry is the walker has to be
decided per frame. See :func:`load_openpose_dir`, ``person="best"``.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..formats import CANONICAL_KEYPOINTS, FormatError, Keypoints

BODY_25: tuple[str, ...] = (
    "Nose", "Neck",
    "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist",
    "MidHip",
    "RHip", "RKnee", "RAnkle",
    "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar",
    "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel",
)

COCO_18: tuple[str, ...] = (
    "Nose", "Neck",
    "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist",
    "RHip", "RKnee", "RAnkle",
    "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar",
)

_MODELS: dict[int, tuple[str, tuple[str, ...]]] = {
    len(BODY_25): ("BODY_25", BODY_25),
    len(COCO_18): ("COCO_18", COCO_18),
}

_TO_CANONICAL: dict[str, str] = {
    "Nose": "NOSE",
    "RShoulder": "RIGHT_SHOULDER", "LShoulder": "LEFT_SHOULDER",
    "RElbow": "RIGHT_ELBOW", "LElbow": "LEFT_ELBOW",
    "RWrist": "RIGHT_WRIST", "LWrist": "LEFT_WRIST",
    "RHip": "RIGHT_HIP", "LHip": "LEFT_HIP",
    "RKnee": "RIGHT_KNEE", "LKnee": "LEFT_KNEE",
    "RAnkle": "RIGHT_ANKLE", "LAnkle": "LEFT_ANKLE",
    "RHeel": "RIGHT_HEEL", "LHeel": "LEFT_HEEL",
    "RBigToe": "RIGHT_BIG_TOE", "LBigToe": "LEFT_BIG_TOE",
    "RSmallToe": "RIGHT_SMALL_TOE", "LSmallToe": "LEFT_SMALL_TOE",
}

# Points that decide which person is the walker: a gait clip is about the legs,
# and a bystander is often only partly in view.
_LOWER_BODY: tuple[str, ...] = ("RHip", "LHip", "RKnee", "LKnee", "RAnkle", "LAnkle")

_FRAME_RE = re.compile(r"^(?P<prefix>.*?)_?(?P<frame>\d+)_keypoints\.json$")

FOOT_INDEX_CHOICES: tuple[str, ...] = ("big_toe", "mean_toes")


def read_openpose_frame(path: str | Path) -> tuple[list[np.ndarray], str]:
    """Read one OpenPose JSON file.

    Parameters
    ----------
    path:
        A ``*_keypoints.json`` file.

    Returns
    -------
    people, version:
        ``people`` holds one ``(n_points, 3)`` array of ``x, y, confidence``
        per detected person (raw values, zeros where missing); ``version`` is
        the file's ``version`` field (the JSON layout version, not the
        OpenPose release).

    Raises
    ------
    FormatError
        If the file is not OpenPose output or a person's array length does
        not match a known model.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise FormatError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(data, dict) or "people" not in data:
        raise FormatError(f"{path}: no 'people' list; not an OpenPose keypoints file")
    people = []
    for i, person in enumerate(data["people"]):
        flat = np.asarray(person.get("pose_keypoints_2d") or [], dtype=float)
        if flat.size == 0:
            continue
        if flat.size % 3 or flat.size // 3 not in _MODELS:
            raise FormatError(
                f"{path}: person {i} has {flat.size} pose values; expected "
                f"{', '.join(str(3 * n) for n in _MODELS)} (BODY_25 or COCO_18)"
            )
        people.append(flat.reshape(-1, 3))
    return people, str(data.get("version", ""))


def load_openpose_dir(
    json_dir: str | Path,
    clip_id: str,
    fps: float | None = None,
    width: float | None = None,
    height: float | None = None,
    person: str | int = "best",
    video_path: str | Path | None = None,
    *,
    backend: str = "openpose",
    foot_index: str | None = None,
    close_conf: float = 0.15,
) -> Keypoints:
    """Load one video's OpenPose JSON files as :class:`~gaitbench.formats.Keypoints`.

    Parameters
    ----------
    json_dir:
        Directory of ``<video>_<frame>_keypoints.json`` files for **one** video.
    clip_id:
        Clip name for the result.
    fps, width, height:
        Nominal frame rate and frame size of the source video. Any left as
        None is read from ``video_path`` with :func:`gaitbench.video.probe_video`.
    person:
        ``"best"`` (default) picks the walker in every frame: the person with
        the highest mean confidence over hips, knees and ankles, except that
        among people within ``close_conf`` of that best score the one nearest
        the previous pick's hip centre wins, so a briefly better-scored
        bystander does not steal the track. An integer picks that index of
        the ``people`` list in every frame (only meaningful with one person
        or OpenPose tracking switched on).
    video_path:
        Source video. When given, ``time_s`` comes from the video's real
        frame timestamps (:func:`gaitbench.video.frame_timestamps`), which
        matters for variable-frame-rate phone video; otherwise
        ``time_s = frame_idx / fps``.
    backend:
        Backend name stored in ``meta['backend']`` (and so in file names).
    foot_index:
        None (default) writes no ``*_FOOT_INDEX`` point; ``"big_toe"`` copies
        the big toe; ``"mean_toes"`` uses the mean of big and small toe. See
        the module docstring.
    close_conf:
        Score margin within which continuity beats confidence (``person="best"``).

    Returns
    -------
    Keypoints
        One row per JSON file, in frame order. ``frame_idx`` is the frame
        number in the file name. Frames with no (selected) person are all NaN
        with confidence 0. Confidences are clipped to [0, 1].

    Raises
    ------
    FileNotFoundError
        If ``json_dir`` holds no ``*_keypoints.json`` files.
    FormatError
        If the files belong to several videos, frame numbers repeat, or a
        file is not OpenPose output.
    ValueError
        If ``fps``/``width``/``height`` are missing and no video is given.
    """
    json_dir = Path(json_dir)
    files = sorted(json_dir.glob("*_keypoints.json"))
    if not files:
        raise FileNotFoundError(f"No *_keypoints.json files in {json_dir}")
    if foot_index is not None and foot_index not in FOOT_INDEX_CHOICES:
        raise ValueError(f"foot_index must be None or one of {FOOT_INDEX_CHOICES}, got {foot_index!r}")
    if isinstance(person, str) and person != "best":
        raise ValueError(f"person must be 'best' or an integer index, got {person!r}")

    video_info: dict[str, Any] = {}
    if fps is None or width is None or height is None:
        if video_path is None:
            raise ValueError("Give fps, width and height, or video_path to read them from the video")
        from ..video import probe_video

        video_info = probe_video(video_path, timestamps=False)
        fps = video_info["fps_nominal"] if fps is None else fps
        width = video_info["width"] if width is None else width
        height = video_info["height"] if height is None else height
    fps, width, height = float(fps), float(width), float(height)
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"fps, width and height must be positive, got {fps}, {width}, {height}")

    frame_idx = _frame_numbers(files)
    order = np.argsort(frame_idx, kind="stable")
    files = [files[i] for i in order]
    frame_idx = frame_idx[order]

    frames: list[list[np.ndarray]] = []
    versions: set[str] = set()
    model_names: set[str] = set()
    for f in files:
        people, version = read_openpose_frame(f)
        versions.add(version)
        model_names.update(_MODELS[len(p)][0] for p in people)
        frames.append(people)
    if len(model_names) > 1:
        raise FormatError(f"{json_dir}: files mix pose models {sorted(model_names)}")
    model_name = model_names.pop() if model_names else "BODY_25"
    op_names = dict(_MODELS.values())[model_name]

    picks, selection_info = _select_people(frames, op_names, person, close_conf, height)

    names = tuple(n for n in CANONICAL_KEYPOINTS if n in {_TO_CANONICAL[o] for o in op_names if o in _TO_CANONICAL})
    col = {_TO_CANONICAL[o]: j for j, o in enumerate(op_names) if o in _TO_CANONICAL}
    n = len(files)
    xy = np.full((n, len(names), 2), np.nan)
    conf = np.zeros((n, len(names)))
    for i, pick in enumerate(picks):
        if pick is None:
            continue
        pts = frames[i][pick]
        for k, name in enumerate(names):
            x, y, c = pts[col[name]]
            if c > 0 and np.isfinite(x) and np.isfinite(y):
                xy[i, k] = (x, y)
                conf[i, k] = min(float(c), 1.0)

    if np.nanmax(np.abs(xy), initial=0.0) <= 1.5 and max(width, height) > 10:
        warnings.warn(
            f"{json_dir}: every coordinate is <= 1.5; the JSON looks normalised "
            "(OpenPose --keypoint_scale), but the harness expects pixels",
            stacklevel=2,
        )

    foot_source = "none"
    if foot_index is not None and model_name == "BODY_25":
        names, xy, conf = _add_foot_index(names, xy, conf, foot_index)
        foot_source = foot_index

    time_s, time_source = _times(frame_idx, fps, video_path)
    meta: dict[str, Any] = {
        "fps": fps,
        "width": width,
        "height": height,
        "source": str(video_path) if video_path is not None else str(json_dir),
        "json_dir": str(json_dir),
        "backend": backend,
        "tool": "openpose",
        # The JSON does not say which OpenPose release wrote it.
        "tool_version": "",
        "openpose_json_version": ",".join(sorted(v for v in versions if v)),
        "openpose_model": model_name,
        "importer": "gaitbench.io.openpose",
        "importer_version": __version__,
        "time_source": time_source,
        "foot_index_source": foot_source,
        "n_files": n,
        **selection_info,
    }
    if video_info:
        meta["video_probe"] = {k: video_info.get(k) for k in ("codec", "rotation", "n_frames", "duration_s")}
    return Keypoints(clip_id=clip_id, names=names, xy=xy, conf=conf, time_s=time_s, frame_idx=frame_idx, meta=meta)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _frame_numbers(files: list[Path]) -> np.ndarray:
    """Frame number of each file, from its name; files must come from one video."""
    matches = [_FRAME_RE.match(f.name) for f in files]
    if all(m is None for m in matches):
        warnings.warn(
            f"{files[0].parent}: file names carry no frame number; using sorted order as frame index",
            stacklevel=3,
        )
        return np.arange(len(files), dtype=np.int64)
    bad = [f.name for f, m in zip(files, matches) if m is None]
    if bad:
        raise FormatError(f"{files[0].parent}: some files have no frame number: {bad[:3]}")
    prefixes = sorted({m["prefix"] for m in matches})
    if len(prefixes) > 1:
        raise FormatError(
            f"{files[0].parent}: JSON files from several videos ({prefixes[:4]}); "
            "give one directory per video"
        )
    idx = np.array([int(m["frame"]) for m in matches], dtype=np.int64)
    if len(np.unique(idx)) != len(idx):
        raise FormatError(f"{files[0].parent}: repeated frame numbers")
    return idx


def _select_people(
    frames: list[list[np.ndarray]],
    op_names: tuple[str, ...],
    person: str | int,
    close_conf: float,
    height: float,
) -> tuple[list[int | None], dict[str, Any]]:
    """Choose one ``people`` entry per frame; return the picks and a summary for ``meta``."""
    n_people = [len(p) for p in frames]
    info: dict[str, Any] = {
        "person_selection": str(person),
        "max_people_per_frame": max(n_people, default=0),
        "n_frames_multi_person": int(sum(k > 1 for k in n_people)),
        "n_frames_no_person": int(sum(k == 0 for k in n_people)),
    }
    if not isinstance(person, str):
        index = int(person)
        picks = [index if index < len(p) else None for p in frames]
        info["n_frames_no_person"] = int(sum(p is None for p in picks))
        return picks, info

    lower = [op_names.index(n) for n in _LOWER_BODY]
    hips = [op_names.index(n) for n in ("RHip", "LHip")]
    picks: list[int | None] = []
    prev_centre: np.ndarray | None = None
    n_continuity = 0
    n_jumps = 0
    for people in frames:
        if not people:
            picks.append(None)
            continue
        # Overall confidence breaks ties between people with no visible legs.
        scores = np.array([p[lower, 2].clip(0, 1).mean() + 1e-3 * p[:, 2].clip(0, 1).mean() for p in people])
        centres = [_hip_centre(p, hips) for p in people]
        best = int(np.argmax(scores))
        pick = best
        close = [i for i in range(len(people)) if scores[i] >= scores[best] - close_conf]
        if prev_centre is not None and len(close) > 1:
            dists = {i: float(np.linalg.norm(centres[i] - prev_centre)) for i in close if centres[i] is not None}
            if dists:
                pick = min(dists, key=dists.get)
                n_continuity += pick != best
        centre = centres[pick]
        if centre is not None:
            # A pick far from the last one usually means the track switched person.
            if prev_centre is not None and np.linalg.norm(centre - prev_centre) > 0.2 * height:
                n_jumps += 1
            prev_centre = centre
        picks.append(pick)
    info["n_frames_continuity_pick"] = n_continuity
    info["n_hip_jumps"] = n_jumps
    return picks, info


def _hip_centre(points: np.ndarray, hips: list[int]) -> np.ndarray | None:
    valid = [points[j, :2] for j in hips if points[j, 2] > 0]
    return np.mean(valid, axis=0) if valid else None


def _add_foot_index(
    names: tuple[str, ...], xy: np.ndarray, conf: np.ndarray, how: str
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    extra_names, extra_xy, extra_conf = [], [], []
    for prefix in ("LEFT_", "RIGHT_"):
        big = names.index(prefix + "BIG_TOE")
        small = names.index(prefix + "SMALL_TOE")
        if how == "big_toe":
            p, c = xy[:, big], conf[:, big]
        else:
            # Both toes must be present, so the point never jumps between one toe and the mean.
            p = (xy[:, big] + xy[:, small]) / 2.0
            c = np.where(np.isnan(p).any(axis=1), 0.0, np.minimum(conf[:, big], conf[:, small]))
        extra_names.append(prefix + "FOOT_INDEX")
        extra_xy.append(p)
        extra_conf.append(c)
    all_names = names + tuple(extra_names)
    order = [all_names.index(n) for n in CANONICAL_KEYPOINTS if n in all_names]
    xy = np.concatenate([xy, np.stack(extra_xy, axis=1)], axis=1)[:, order]
    conf = np.concatenate([conf, np.stack(extra_conf, axis=1)], axis=1)[:, order]
    return tuple(all_names[i] for i in order), xy, conf


def _times(frame_idx: np.ndarray, fps: float, video_path: str | Path | None) -> tuple[np.ndarray, str]:
    if video_path is not None:
        from ..video import frame_timestamps

        ts = frame_timestamps(video_path)
        if len(frame_idx) and frame_idx.max() < len(ts) and np.all(np.diff(ts[frame_idx]) > 0):
            return ts[frame_idx], "timestamps"
        warnings.warn(
            f"{video_path}: {len(ts)} frame timestamps do not cover JSON frame {int(frame_idx.max())}; "
            "using frame_idx / fps",
            stacklevel=3,
        )
    return frame_idx / fps, "nominal_fps"

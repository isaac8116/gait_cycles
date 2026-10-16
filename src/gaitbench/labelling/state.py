"""The labelling session: which events a rater has marked on which frames.

This module is the pure logic behind the frame-stepping labeller
(:mod:`gaitbench.labelling.app`). It knows nothing about video or windows, so
it can be tested and reused (a notebook, another GUI) without OpenCV.

A session holds one rater's labels for one clip:

* ``n_frames`` frames, numbered as in the source video (frame 0 is the first
  decoded frame), each with its presentation time in seconds. Times are per
  frame because iPhone video can have a variable frame rate.
* a current frame, moved with :meth:`LabelSession.goto`, :meth:`~LabelSession.step`,
  :meth:`~LabelSession.next_event` and :meth:`~LabelSession.prev_event`;
* a set of events (side ``"L"``/``"R"`` = the walker's own side, event
  ``"HS"``/``"TO"``, frame), edited with :meth:`~LabelSession.add` and
  :meth:`~LabelSession.remove_nearest`, with unlimited undo and redo.

Saved files use the labels format of :mod:`gaitbench.formats`
(``tool = "human"``, ``detector`` = the rater), so the scorer reads them like
any other truth. Comparing two raters is the scoring module's job
(:func:`gaitbench.scoring.agreement`), not this one's.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .. import __version__
from ..formats import EVENT_TYPES, SIDES, FormatError, make_events, read_labels, write_labels

# An identical side/event this close to a new label is taken to be the same
# event placed again (the rater correcting it), not a second event.
REPLACE_WINDOW_FRAMES = 2


@dataclass(frozen=True, order=True)
class LabelledEvent:
    """One labelled event. Ordered by frame, then side, then event."""

    frame: int
    side: str
    event: str

    @property
    def code(self) -> str:
        """Short name for display, e.g. ``"L HS"``."""
        return f"{self.side} {self.event}"


@dataclass(frozen=True)
class Edit:
    """One undoable change: the events it removed and the events it added.

    ``frame`` is where the change happened, so a GUI can jump there on undo.
    """

    kind: str  # "add" or "remove"
    frame: int
    added: tuple[LabelledEvent, ...] = ()
    removed: tuple[LabelledEvent, ...] = ()


class LabelSession:
    """One rater's heel-strike and toe-off labels for one clip.

    Parameters
    ----------
    clip_id:
        Clip name; written to every label row.
    rater:
        Rater's name or initials; written to the ``detector`` column and used
        in the labels file name (:func:`gaitbench.formats.labels_path`).
    n_frames:
        Number of frames in the video. Optional if ``timestamps`` is given.
    timestamps:
        ``(n_frames,)`` presentation time of every frame in seconds, e.g. from
        :func:`gaitbench.video.frame_timestamps`. If omitted, times are
        ``frame / fps``.
    fps:
        Nominal frame rate, used only when ``timestamps`` is omitted.
    start_frame:
        Initial current frame (clamped to the clip).

    Raises
    ------
    ValueError
        If the frame count or times are missing, inconsistent or not finite.
    """

    def __init__(
        self,
        clip_id: str,
        rater: str,
        n_frames: int | None = None,
        timestamps: Sequence[float] | np.ndarray | None = None,
        *,
        fps: float | None = None,
        start_frame: int = 0,
    ) -> None:
        if not str(clip_id).strip():
            raise ValueError("clip_id must not be empty")
        if not str(rater).strip():
            raise ValueError("rater must not be empty")
        self.clip_id = str(clip_id)
        self.rater = str(rater)
        if timestamps is not None:
            ts = np.array(timestamps, dtype=float).reshape(-1)
            if n_frames is not None and int(n_frames) != ts.size:
                raise ValueError(f"n_frames={n_frames} but {ts.size} timestamps were given")
        elif n_frames is not None and fps:
            if fps <= 0:
                raise ValueError("fps must be positive")
            ts = np.arange(int(n_frames)) / float(fps)
        else:
            raise ValueError("Give timestamps, or n_frames and fps")
        if ts.size == 0:
            raise ValueError("The clip has no frames")
        if not np.all(np.isfinite(ts)):
            raise ValueError("timestamps must be finite")
        if ts.size > 1 and np.any(np.diff(ts) <= 0):
            # Labels are still usable (frames are the primary key), but times
            # that do not increase usually mean the wrong timestamp source.
            warnings.warn(f"{self.clip_id}: frame timestamps are not strictly increasing", stacklevel=2)
        self._timestamps = ts
        self._timestamps.flags.writeable = False
        self._current = 0
        self._events: list[LabelledEvent] = []
        self._undo: list[Edit] = []
        self._redo: list[Edit] = []
        self._n_edits = 0
        self._saved_at = 0
        self.goto(start_frame)

    # -- read-only state -----------------------------------------------------

    @property
    def n_frames(self) -> int:
        return int(self._timestamps.size)

    @property
    def timestamps(self) -> np.ndarray:
        """Read-only ``(n_frames,)`` frame times in seconds."""
        return self._timestamps

    @property
    def current(self) -> int:
        """The current frame index."""
        return self._current

    @property
    def current_time(self) -> float:
        return self.time_of(self._current)

    @property
    def events(self) -> tuple[LabelledEvent, ...]:
        """All events, sorted by frame."""
        return tuple(sorted(self._events))

    @property
    def n_events(self) -> int:
        return len(self._events)

    @property
    def n_edits(self) -> int:
        """Number of changes so far (adds, removals, undos and redos)."""
        return self._n_edits

    @property
    def dirty(self) -> bool:
        """True if there are changes since the last :meth:`save` or :meth:`load`."""
        return self._n_edits != self._saved_at

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def time_of(self, frame: int) -> float:
        """Presentation time of ``frame`` in seconds."""
        return float(self._timestamps[self._check_frame(frame)])

    # -- navigation ------------------------------------------------------------

    def goto(self, frame: int) -> int:
        """Make ``frame`` current (clamped to the clip) and return it."""
        self._current = int(min(max(int(frame), 0), self.n_frames - 1))
        return self._current

    def step(self, n: int) -> int:
        """Move ``n`` frames (negative = back), clamped, and return the new frame."""
        return self.goto(self._current + int(n))

    def next_event(self) -> int | None:
        """Jump to the first labelled frame after the current one.

        Returns
        -------
        int or None
            The new current frame, or None (and no move) if there is none.
        """
        later = [e.frame for e in self._events if e.frame > self._current]
        return self.goto(min(later)) if later else None

    def prev_event(self) -> int | None:
        """Jump to the last labelled frame before the current one (None if there is none)."""
        earlier = [e.frame for e in self._events if e.frame < self._current]
        return self.goto(max(earlier)) if earlier else None

    # -- queries ---------------------------------------------------------------

    def events_near(self, frame: int | None = None, window: int = 30) -> list[LabelledEvent]:
        """Events within ``window`` frames of ``frame`` (default: the current frame), by frame."""
        f = self._current if frame is None else int(frame)
        return sorted(e for e in self._events if abs(e.frame - f) <= window)

    def events_at(self, frame: int | None = None) -> list[LabelledEvent]:
        """Events exactly on ``frame`` (default: the current frame)."""
        return self.events_near(frame, window=0)

    def nearest_event(self, frame: int | None = None, max_distance_frames: int | None = None) -> LabelledEvent | None:
        """The event closest to ``frame``; ties go to the most recently added.

        Parameters
        ----------
        frame:
            Reference frame (default: the current frame).
        max_distance_frames:
            Ignore events further away than this.
        """
        f = self._current if frame is None else int(frame)
        best: LabelledEvent | None = None
        best_d = math.inf
        # Iterating in insertion order with "<=" makes the latest-added win ties,
        # which is what a rater pressing delete right after a mistake expects.
        for e in self._events:
            d = abs(e.frame - f)
            if max_distance_frames is not None and d > max_distance_frames:
                continue
            if d <= best_d:
                best, best_d = e, d
        return best

    def counts(self) -> dict[str, int]:
        """Number of events of each kind, e.g. ``{"L HS": 3, "L TO": 3, "R HS": 2, "R TO": 3}``."""
        out = {f"{s} {ev}": 0 for s in SIDES for ev in EVENT_TYPES}
        for e in self._events:
            out[e.code] += 1
        return out

    def sequence_issues(self) -> list[str]:
        """Plausibility warnings: on each side, heel strikes and toe offs should alternate.

        Two heel strikes on one side with no toe off between them (or the
        reverse) usually mean a missed or mislabelled event, often left and
        right swapped. These are warnings, not errors: a clip can start or end
        mid-cycle, and a rater may skip an event they cannot see.

        Returns
        -------
        list of str
            One human-readable line per problem, empty if none.
        """
        issues = []
        for side in SIDES:
            seq = sorted(e for e in self._events if e.side == side)
            for a, b in zip(seq, seq[1:]):
                if a.event == b.event:
                    other = "TO" if a.event == "HS" else "HS"
                    issues.append(f"{side}: {a.event} at frame {a.frame} then {b.event} at {b.frame} with no {other} between")
        return issues

    # -- editing ---------------------------------------------------------------

    def add(self, side: str, event: str) -> Edit | None:
        """Label ``side``/``event`` at the current frame.

        An existing label of the same side and event within
        :data:`REPLACE_WINDOW_FRAMES` frames is replaced (moved here), so
        pressing a key again a frame later corrects a label instead of adding
        a duplicate.

        Returns
        -------
        Edit or None
            The change made, or None if that exact label already exists here.

        Raises
        ------
        ValueError
            If ``side`` or ``event`` is not a valid code.
        """
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        if event not in EVENT_TYPES:
            raise ValueError(f"event must be one of {EVENT_TYPES}, got {event!r}")
        new = LabelledEvent(frame=self._current, side=side, event=event)
        if new in self._events:
            return None
        removed = tuple(
            e for e in self._events
            if e.side == side and e.event == event and abs(e.frame - new.frame) <= REPLACE_WINDOW_FRAMES
        )
        edit = Edit(kind="add", frame=new.frame, added=(new,), removed=removed)
        self._apply(edit)
        self._record(edit)
        return edit

    def remove_nearest(self, max_distance_frames: int = 3) -> LabelledEvent | None:
        """Delete the event nearest the current frame, if within ``max_distance_frames``.

        Returns
        -------
        LabelledEvent or None
            The deleted event, or None if nothing was close enough.
        """
        target = self.nearest_event(max_distance_frames=max_distance_frames)
        if target is None:
            return None
        edit = Edit(kind="remove", frame=target.frame, removed=(target,))
        self._apply(edit)
        self._record(edit)
        return target

    def undo(self) -> Edit | None:
        """Revert the last change. Returns it, or None if there is nothing to undo."""
        if not self._undo:
            return None
        edit = self._undo.pop()
        self._apply(Edit(kind=edit.kind, frame=edit.frame, added=edit.removed, removed=edit.added))
        self._redo.append(edit)
        self._n_edits += 1
        return edit

    def redo(self) -> Edit | None:
        """Re-apply the last undone change. Returns it, or None if there is nothing to redo."""
        if not self._redo:
            return None
        edit = self._redo.pop()
        self._apply(edit)
        self._undo.append(edit)
        self._n_edits += 1
        return edit

    def clear(self) -> Edit | None:
        """Remove every event as one undoable change (None if there were none)."""
        if not self._events:
            return None
        edit = Edit(kind="remove", frame=self._current, removed=tuple(self._events))
        self._apply(edit)
        self._record(edit)
        return edit

    # -- files -----------------------------------------------------------------

    def to_events_df(self) -> pd.DataFrame:
        """The labels as a validated events table (:data:`gaitbench.formats.EVENT_COLUMNS`).

        ``tool`` is ``"human"``, ``tool_version`` the gaitbench version,
        ``backend`` empty, ``detector`` the rater, ``time_s`` the frame's
        timestamp and ``confidence`` NaN.
        """
        records = [
            {"side": e.side, "event": e.event, "frame": e.frame, "time_s": self.time_of(e.frame)}
            for e in sorted(self._events)
        ]
        return make_events(
            records,
            defaults={
                "clip_id": self.clip_id,
                "tool": "human",
                "tool_version": __version__,
                "backend": "",
                "detector": self.rater,
                "confidence": np.nan,
            },
        )

    def save(self, path: str | Path, *, mark_saved: bool = True) -> Path:
        """Write the labels CSV with :func:`gaitbench.formats.write_labels`.

        Parameters
        ----------
        path:
            Output file, normally :func:`gaitbench.formats.labels_path`.
        mark_saved:
            Clear :attr:`dirty`. Pass False for autosaves, which are not the
            real file.

        Returns
        -------
        pathlib.Path
            The file written.
        """
        out = write_labels(self.to_events_df(), path)
        if mark_saved:
            self._saved_at = self._n_edits
        return out

    def load(self, path: str | Path) -> int:
        """Replace the events with those in a labels file, to resume labelling.

        Undo history is cleared and the session is marked clean.

        Parameters
        ----------
        path:
            A labels CSV written by :meth:`save` (or any file in the labels format).

        Returns
        -------
        int
            Number of events loaded.

        Raises
        ------
        FormatError
            If the file is not in the labels format, or belongs to another
            clip or rater (labelling is blind: a rater does not continue
            someone else's file).
        ValueError
            If a frame lies outside this clip.
        """
        df = read_labels(path)
        clips = sorted(set(df["clip_id"]))
        if clips and clips != [self.clip_id]:
            raise FormatError(f"{path}: labels are for clip {clips}, not {self.clip_id!r}")
        raters = sorted(set(df["detector"]))
        if raters and raters != [self.rater]:
            raise FormatError(f"{path}: labels are by rater {raters}, not {self.rater!r}")
        events = [
            LabelledEvent(frame=int(r.frame), side=str(r.side), event=str(r.event))
            for r in df.itertuples(index=False)
        ]
        bad = [e.frame for e in events if not 0 <= e.frame < self.n_frames]
        if bad:
            raise ValueError(f"{path}: frames {bad} are outside this clip (0..{self.n_frames - 1})")
        self._warn_time_mismatch(df, path)
        self._events = sorted(set(events))
        self._undo.clear()
        self._redo.clear()
        self._saved_at = self._n_edits
        return len(self._events)

    @classmethod
    def from_labels(
        cls,
        path: str | Path,
        n_frames: int | None = None,
        timestamps: Sequence[float] | np.ndarray | None = None,
        *,
        fps: float | None = None,
    ) -> "LabelSession":
        """Open a session for the clip and rater named inside a labels file, and load it."""
        df = read_labels(path)
        if df.empty:
            raise FormatError(f"{path}: no labels, so the clip and rater are unknown")
        session = cls(str(df["clip_id"].iloc[0]), str(df["detector"].iloc[0]), n_frames, timestamps, fps=fps)
        session.load(path)
        return session

    # -- internals -------------------------------------------------------------

    def _check_frame(self, frame: int) -> int:
        f = int(frame)
        if not 0 <= f < self.n_frames:
            raise IndexError(f"frame {f} is outside 0..{self.n_frames - 1}")
        return f

    def _apply(self, edit: Edit) -> None:
        for e in edit.removed:
            self._events.remove(e)
        self._events.extend(edit.added)

    def _record(self, edit: Edit) -> None:
        self._undo.append(edit)
        self._redo.clear()
        self._n_edits += 1

    def _warn_time_mismatch(self, df: pd.DataFrame, path: str | Path) -> None:
        """Warn if the file's times disagree with this session's timestamps.

        Frames are what is loaded; times are recomputed on save. A mismatch
        means the file was labelled with another timestamp source (e.g.
        nominal fps instead of the real VFR times), which is worth knowing.
        """
        if df.empty or self.n_frames < 2:
            return
        frames = df["frame"].to_numpy()
        expected = self._timestamps[frames]
        tol = 0.5 * float(np.median(np.diff(self._timestamps)))
        diff = np.abs(df["time_s"].to_numpy(dtype=float) - expected)
        n_off = int(np.sum(diff > tol))
        if n_off:
            warnings.warn(
                f"{path}: {n_off} of {len(df)} label times differ from this video's frame times by more than "
                f"half a frame (max {np.nanmax(diff):.4f} s); times will be rewritten from the video on save",
                stacklevel=3,
            )

    def __repr__(self) -> str:
        return (
            f"LabelSession(clip_id={self.clip_id!r}, rater={self.rater!r}, n_frames={self.n_frames}, "
            f"current={self._current}, n_events={self.n_events})"
        )

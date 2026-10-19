"""Tests for gaitbench.labelling: the LabelSession logic and the labeller's pure helpers (no windows)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from gaitbench import __version__
from gaitbench.formats import EVENT_COLUMNS, FormatError, read_labels, validate_events, write_labels
from gaitbench.labelling import KEY_BINDINGS, LabelSession, LabelledEvent
from gaitbench.labelling.app import (
    FrameCache,
    ViewState,
    autosave_path,
    canvas_layout,
    decode_key,
    describe_edit,
    display_size,
    foot_crop_box,
    frame_to_x,
    handle_action,
    load_timestamps,
    sidecar_path,
    x_to_frame,
)
from gaitbench.labelling.state import REPLACE_WINDOW_FRAMES
from gaitbench.synthetic import simulate_walk


def session(n: int = 200, fps: float = 60.0, **kw) -> LabelSession:
    return LabelSession("clip1", "alice", n, fps=fps, **kw)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construct_from_fps_and_from_timestamps():
    s = session(100, fps=50.0)
    assert s.n_frames == 100
    assert s.time_of(50) == pytest.approx(1.0)
    ts = np.cumsum(np.full(10, 0.02)) - 0.02
    s2 = LabelSession("c", "r", timestamps=ts)
    assert s2.n_frames == 10 and s2.time_of(3) == pytest.approx(0.06)


def test_timestamps_are_copied_and_read_only():
    ts = np.arange(5) / 10.0
    s = LabelSession("c", "r", timestamps=ts)
    ts[0] = 99.0
    assert s.time_of(0) == 0.0
    with pytest.raises(ValueError):
        s.timestamps[0] = 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # nothing to define the frames
        {"n_frames": 10},  # no fps
        {"n_frames": 3, "timestamps": [0.0, 0.1]},  # mismatch
        {"timestamps": []},
        {"timestamps": [0.0, np.nan, 0.2]},
        {"n_frames": 10, "fps": -5.0},
    ],
)
def test_construct_rejects_bad_frames(kwargs):
    with pytest.raises(ValueError):
        LabelSession("c", "r", **kwargs)


def test_construct_rejects_empty_names():
    with pytest.raises(ValueError):
        LabelSession("", "r", 10, fps=30)
    with pytest.raises(ValueError):
        LabelSession("c", " ", 10, fps=30)


def test_non_increasing_timestamps_warn():
    with pytest.warns(UserWarning, match="not strictly increasing"):
        LabelSession("c", "r", timestamps=[0.0, 0.1, 0.1, 0.2])


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


def test_goto_and_step_clamp():
    s = session(100)
    assert s.current == 0
    assert s.goto(42) == 42 and s.current == 42
    assert s.step(10) == 52
    assert s.step(-100) == 0
    assert s.goto(1000) == 99
    assert s.step(1) == 99
    assert s.current_time == pytest.approx(99 / 60.0)


def test_start_frame_is_clamped():
    assert session(100, start_frame=500).current == 99


def test_next_and_prev_event():
    s = session()
    assert s.next_event() is None and s.prev_event() is None
    for f, code in ((10, ("L", "HS")), (40, ("R", "TO")), (70, ("L", "TO"))):
        s.goto(f)
        s.add(*code)
    s.goto(0)
    assert s.next_event() == 10
    assert s.next_event() == 40
    assert s.next_event() == 70
    assert s.next_event() is None and s.current == 70  # no move when there is none
    assert s.prev_event() == 40
    s.goto(25)
    assert s.prev_event() == 10
    assert s.prev_event() is None and s.current == 10


def test_next_event_skips_events_on_the_current_frame():
    s = session()
    s.goto(20)
    s.add("L", "HS")
    s.add("R", "TO")
    s.goto(30)
    s.add("R", "HS")
    s.goto(20)
    assert s.next_event() == 30


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------


def test_add_at_current_frame():
    s = session()
    s.goto(12)
    edit = s.add("L", "HS")
    assert edit is not None and edit.kind == "add" and edit.frame == 12
    assert s.events == (LabelledEvent(12, "L", "HS"),)
    assert s.n_edits == 1 and s.dirty


def test_add_same_label_twice_is_a_no_op():
    s = session()
    s.goto(12)
    s.add("L", "HS")
    assert s.add("L", "HS") is None
    assert s.n_events == 1 and s.n_edits == 1


@pytest.mark.parametrize("offset", [-2, -1, 1, 2])
def test_add_replaces_same_side_event_within_window(offset):
    assert REPLACE_WINDOW_FRAMES == 2
    s = session()
    s.goto(50)
    s.add("R", "TO")
    s.goto(50 + offset)
    edit = s.add("R", "TO")
    assert edit.removed == (LabelledEvent(50, "R", "TO"),)
    assert s.events == (LabelledEvent(50 + offset, "R", "TO"),)


def test_add_outside_window_or_other_code_does_not_replace():
    s = session()
    s.goto(50)
    s.add("R", "TO")
    s.goto(53)
    s.add("R", "TO")  # 3 frames away: a second event
    s.goto(51)
    s.add("L", "TO")  # other side
    s.add("R", "HS")  # other event type
    assert [e.frame for e in s.events] == [50, 51, 51, 53]
    assert s.n_events == 4


def test_add_replaces_every_close_duplicate():
    s = session()
    s.goto(10)
    s.add("L", "HS")
    s.goto(13)
    s.add("L", "HS")
    s.goto(12)  # within 2 of 10 and within 1 of 13
    edit = s.add("L", "HS")
    assert set(edit.removed) == {LabelledEvent(10, "L", "HS"), LabelledEvent(13, "L", "HS")}
    assert s.events == (LabelledEvent(12, "L", "HS"),)


@pytest.mark.parametrize("side,event", [("X", "HS"), ("L", "IC"), ("left", "HS")])
def test_add_rejects_bad_codes(side, event):
    with pytest.raises(ValueError):
        session().add(side, event)


def test_remove_nearest():
    s = session()
    for f in (10, 20):
        s.goto(f)
        s.add("L", "HS")
    s.goto(22)
    removed = s.remove_nearest(max_distance_frames=3)
    assert removed == LabelledEvent(20, "L", "HS")
    assert s.events == (LabelledEvent(10, "L", "HS"),)
    s.goto(14)
    assert s.remove_nearest(max_distance_frames=3) is None
    assert s.n_events == 1
    assert s.remove_nearest(max_distance_frames=4) == LabelledEvent(10, "L", "HS")


def test_remove_nearest_tie_prefers_latest_added():
    s = session()
    s.goto(30)
    s.add("R", "HS")
    s.add("L", "TO")
    assert s.remove_nearest() == LabelledEvent(30, "L", "TO")
    s.goto(28)
    s.add("L", "HS")
    s.goto(32)
    s.add("R", "TO")
    s.goto(30)  # R HS at 0 frames wins over the two at 2
    assert s.remove_nearest() == LabelledEvent(30, "R", "HS")
    assert s.remove_nearest() == LabelledEvent(32, "R", "TO")


def test_undo_redo_add_and_remove():
    s = session()
    s.goto(10)
    s.add("L", "HS")
    s.goto(20)
    s.add("L", "TO")
    s.remove_nearest()
    assert s.events == (LabelledEvent(10, "L", "HS"),)
    edit = s.undo()
    assert edit.kind == "remove" and edit.frame == 20
    assert s.events == (LabelledEvent(10, "L", "HS"), LabelledEvent(20, "L", "TO"))
    s.undo()
    s.undo()
    assert s.events == () and not s.can_undo and s.undo() is None
    assert s.redo().frame == 10
    assert s.redo().frame == 20
    assert s.redo().kind == "remove"
    assert s.events == (LabelledEvent(10, "L", "HS"),)
    assert not s.can_redo and s.redo() is None
    assert s.n_edits == 9  # 3 edits, 3 undos, 3 redos


def test_undo_of_a_replacement_restores_the_original():
    s = session()
    s.goto(40)
    s.add("R", "HS")
    s.goto(41)
    s.add("R", "HS")
    s.undo()
    assert s.events == (LabelledEvent(40, "R", "HS"),)
    s.redo()
    assert s.events == (LabelledEvent(41, "R", "HS"),)


def test_new_edit_clears_redo():
    s = session()
    s.goto(10)
    s.add("L", "HS")
    s.undo()
    assert s.can_redo
    s.goto(15)
    s.add("R", "HS")
    assert not s.can_redo and s.redo() is None


def test_clear_is_undoable():
    s = session()
    assert s.clear() is None
    for f in (5, 9):
        s.goto(f)
        s.add("L", "HS")
    s.clear()
    assert s.n_events == 0
    s.undo()
    assert [e.frame for e in s.events] == [5, 9]


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def test_events_near_at_and_counts():
    s = session()
    for f, code in ((10, ("L", "HS")), (12, ("R", "TO")), (60, ("L", "TO"))):
        s.goto(f)
        s.add(*code)
    s.goto(11)
    assert [e.frame for e in s.events_near(window=5)] == [10, 12]
    assert s.events_at(12) == [LabelledEvent(12, "R", "TO")]
    assert s.events_at() == []
    assert s.counts() == {"L HS": 1, "L TO": 1, "R HS": 0, "R TO": 1}
    assert s.nearest_event(55).frame == 60
    assert s.nearest_event(55, max_distance_frames=2) is None


def test_sequence_issues():
    s = session(400)
    for f, code in ((10, ("L", "HS")), (40, ("L", "TO")), (70, ("L", "HS")), (130, ("L", "HS")), (90, ("R", "TO"))):
        s.goto(f)
        s.add(*code)
    issues = s.sequence_issues()
    assert len(issues) == 1
    assert issues[0].startswith("L: HS at frame 70 then HS at 130")


# ---------------------------------------------------------------------------
# Events table, save and load
# ---------------------------------------------------------------------------


def vfr_session() -> LabelSession:
    rng = np.random.default_rng(1)
    ts = np.cumsum(rng.uniform(1 / 70, 1 / 50, size=300))
    ts -= ts[0]
    s = LabelSession("p01_F240_LtoR_normal_01", "bob", timestamps=ts)
    for f, code in ((15, ("L", "HS")), (52, ("L", "TO")), (48, ("R", "HS")), (90, ("R", "TO")), (79, ("L", "HS"))):
        s.goto(f)
        s.add(*code)
    return s


def test_to_events_df_follows_the_labels_format():
    s = vfr_session()
    df = s.to_events_df()
    assert tuple(df.columns) == EVENT_COLUMNS
    assert len(df) == 5
    assert set(df["tool"]) == {"human"}
    assert set(df["tool_version"]) == {__version__}
    assert set(df["backend"]) == {""}
    assert set(df["detector"]) == {"bob"}
    assert set(df["clip_id"]) == {"p01_F240_LtoR_normal_01"}
    assert df["confidence"].isna().all()
    for r in df.itertuples():
        assert r.time_s == pytest.approx(s.timestamps[r.frame])
    pd.testing.assert_frame_equal(df, validate_events(df))


def test_empty_session_gives_an_empty_valid_table(tmp_path):
    s = session()
    df = s.to_events_df()
    assert len(df) == 0 and tuple(df.columns) == EVENT_COLUMNS
    path = s.save(tmp_path / "empty.csv")
    assert len(read_labels(path)) == 0
    s2 = session()
    assert s2.load(path) == 0


def test_save_load_round_trip_through_read_labels(tmp_path):
    s = vfr_session()
    path = s.save(tmp_path / "labels" / "clip__bob.csv")
    assert path.is_file() and not s.dirty
    back = read_labels(path)
    pd.testing.assert_frame_equal(back, s.to_events_df(), check_dtype=False)

    resumed = LabelSession(s.clip_id, "bob", timestamps=s.timestamps)
    assert resumed.load(path) == 5
    assert resumed.events == s.events
    assert not resumed.dirty and not resumed.can_undo
    resumed.goto(200)
    resumed.add("R", "HS")
    assert resumed.dirty
    resumed.save(path)
    assert len(read_labels(path)) == 6


def test_autosave_does_not_mark_clean(tmp_path):
    s = session()
    s.add("L", "HS")
    s.save(tmp_path / "x.csv.autosave", mark_saved=False)
    assert s.dirty
    s.save(tmp_path / "x.csv")
    assert not s.dirty


def test_load_rejects_other_clip_rater_or_frames(tmp_path):
    s = vfr_session()
    path = s.save(tmp_path / "a.csv")
    with pytest.raises(FormatError, match="clip"):
        LabelSession("other_clip", "bob", timestamps=s.timestamps).load(path)
    with pytest.raises(FormatError, match="rater"):
        LabelSession(s.clip_id, "alice", timestamps=s.timestamps).load(path)
    with pytest.raises(ValueError, match="outside"):
        LabelSession(s.clip_id, "bob", 50, fps=60).load(path)


def test_load_warns_when_times_disagree(tmp_path):
    s = vfr_session()
    path = s.save(tmp_path / "a.csv")
    nominal = LabelSession(s.clip_id, "bob", s.n_frames, fps=30.0)  # very different times
    with pytest.warns(UserWarning, match="differ from this video's frame times"):
        nominal.load(path)
    same = LabelSession(s.clip_id, "bob", timestamps=s.timestamps)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        same.load(path)


def test_load_drops_exact_duplicates(tmp_path):
    s = session()
    s.goto(10)
    s.add("L", "HS")
    df = s.to_events_df()
    path = write_labels(pd.concat([df, df], ignore_index=True), tmp_path / "dup.csv")
    s2 = session()
    assert s2.load(path) == 1


def test_from_labels(tmp_path):
    s = vfr_session()
    path = s.save(tmp_path / "a.csv")
    s2 = LabelSession.from_labels(path, timestamps=s.timestamps)
    assert (s2.clip_id, s2.rater, s2.events) == (s.clip_id, "bob", s.events)
    with pytest.raises(FormatError):
        LabelSession.from_labels(session().save(tmp_path / "empty.csv"), 10, fps=30)


def test_time_of_out_of_range():
    with pytest.raises(IndexError):
        session(10).time_of(10)


# ---------------------------------------------------------------------------
# Keys and actions
# ---------------------------------------------------------------------------


def test_decode_key_letters_and_specials():
    assert decode_key(-1) is None
    assert decode_key(None) is None
    assert decode_key(ord("d")) == "d"
    assert decode_key(ord("D")) == "D"
    assert decode_key(ord("1")) == "1"
    assert decode_key(13) == "enter" and decode_key(10) == "enter"
    assert decode_key(27) == "esc"
    assert decode_key(32) is None  # space is unbound and unnamed


def test_decode_key_platform_codes():
    shift_numlock = (0x1 | 0x10) << 16  # GTK puts modifier state above bit 16
    assert decode_key(ord("D") | shift_numlock) == "D"
    assert decode_key(0xFF0D | (0x10 << 16)) == "enter"
    assert decode_key(0xFF53) == "right" and decode_key(0xFF51 | (0x10 << 16)) == "left"  # X11
    assert decode_key(0x270000) == "right" and decode_key(0x250000) == "left"  # Windows
    assert decode_key(0xF703) == "right" and decode_key(0xF702) == "left"  # macOS Cocoa
    assert decode_key(0xF700) == "up" and decode_key(0xF701) == "down"
    assert decode_key(0xFFB3) == "3"  # keypad digit on X11


def test_key_bindings_cover_the_documented_keys():
    expected = {
        "d": "step:1", "a": "step:-1", "D": "step:10", "A": "step:-10",
        "w": "next_event", "s": "prev_event",
        "1": "add:L:HS", "2": "add:L:TO", "3": "add:R:HS", "4": "add:R:TO",
        "x": "delete", "u": "undo", "r": "redo", "z": "zoom", "h": "help",
        "enter": "save", "q": "quit", "esc": "quit",
    }
    for key, action in expected.items():
        assert KEY_BINDINGS[key] == action


def test_every_binding_is_a_valid_action():
    for action in set(KEY_BINDINGS.values()):
        handle_action(session(), ViewState(), action, zoom_available=True)


def test_handle_action_drives_the_session():
    s, v = session(100), ViewState()
    assert handle_action(s, v, KEY_BINDINGS["D"]) == "" and s.current == 10
    assert handle_action(s, v, KEY_BINDINGS["a"]) == "" and s.current == 9
    assert "L HS" in handle_action(s, v, KEY_BINDINGS["1"])
    assert "already" in handle_action(s, v, KEY_BINDINGS["1"])
    handle_action(s, v, KEY_BINDINGS["d"])
    assert "moved from 9" in handle_action(s, v, KEY_BINDINGS["1"])
    handle_action(s, v, KEY_BINDINGS["4"])
    assert s.events == (LabelledEvent(10, "L", "HS"), LabelledEvent(10, "R", "TO"))
    assert "deleted R TO" in handle_action(s, v, KEY_BINDINGS["x"])
    s.goto(80)
    assert handle_action(s, v, KEY_BINDINGS["u"]).startswith("undo") and s.current == 10  # jumps to the change
    assert s.n_events == 2
    s.goto(0)
    assert "start of clip" in handle_action(s, v, KEY_BINDINGS["A"])
    s.goto(99)
    assert "end of clip" in handle_action(s, v, KEY_BINDINGS["d"])
    assert "no labelled events after" in handle_action(s, v, KEY_BINDINGS["w"])
    assert handle_action(s, v, KEY_BINDINGS["s"]) == "" and s.current == 10
    assert handle_action(s, v, KEY_BINDINGS["r"]) == "redo: delete R TO at 10"
    assert "nothing to redo" in handle_action(s, v, KEY_BINDINGS["r"])


def test_handle_action_view_flags():
    s, v = session(), ViewState()
    assert "needs keypoints" in handle_action(s, v, "zoom", zoom_available=False) and not v.zoom
    handle_action(s, v, "zoom", zoom_available=True)
    assert v.zoom
    handle_action(s, v, "help")
    assert v.help
    handle_action(s, v, "save")
    assert v.save_requested and not v.quit
    handle_action(s, v, "quit")
    assert v.quit
    with pytest.raises(ValueError):
        handle_action(s, v, "launch")


def test_view_message_expires():
    v = ViewState()
    v.set_message("hi", seconds=60)
    assert v.visible_message() == "hi"
    v.set_message("gone", seconds=-1)
    assert v.visible_message() == ""


def test_describe_edit():
    s = session()
    s.goto(5)
    s.add("L", "HS")
    s.goto(6)
    assert describe_edit(s.add("L", "HS")) == "add L HS at 6 (moved from 5)"
    s.remove_nearest()
    assert describe_edit(s.undo()) == "delete L HS at 6"


# ---------------------------------------------------------------------------
# Frame cache
# ---------------------------------------------------------------------------


class FakeSource:
    """Sequential 'decoder' whose frames are filled with their own index."""

    def __init__(self, n: int, fail_at: int | None = None) -> None:
        self.n, self.pos, self.fail_at = n, 0, fail_at
        self.seeks: list[int] = []
        self.reads = 0

    def seek(self, index: int) -> None:
        self.seeks.append(index)
        self.pos = index

    def read(self):
        if self.pos >= self.n or self.pos == self.fail_at:
            return None
        img = np.full((2, 2, 3), self.pos % 256, np.uint8)
        self.pos += 1
        self.reads += 1
        return img


def test_frame_cache_reads_forward_without_seeking():
    src = FakeSource(500)
    cache = FrameCache(src, 500, size=32, backfill=12, max_skip=30)
    assert cache.get(0)[0, 0, 0] == 0
    for i in range(1, 60):
        assert cache.get(i)[0, 0, 0] == i % 256
    assert src.seeks == [0]
    cache.get(80)  # 20 ahead: read through
    assert src.seeks == [0]
    assert src.reads == 81


def test_frame_cache_backward_steps_hit_the_backfill():
    src = FakeSource(1000)
    cache = FrameCache(src, 1000, size=32, backfill=12)
    assert cache.get(400)[0, 0, 0] == 400 % 256
    assert src.seeks == [388]
    for i in range(399, 387, -1):  # step back 12 frames: all cached
        assert cache.get(i)[0, 0, 0] == i % 256
    assert src.seeks == [388]
    cache.get(387)  # one more step back: seek again with backfill
    assert src.seeks == [388, 375]
    assert cache.get(800) is not None and src.seeks[-1] == 788


def test_frame_cache_bounds_and_failures():
    src = FakeSource(100, fail_at=50)
    cache = FrameCache(src, 100, size=16, backfill=4)
    assert cache.get(-1) is None and cache.get(100) is None
    assert cache.get(49) is not None
    assert cache.get(50) is None  # decode error
    assert cache.get(60) is not None  # recovers with a seek
    assert len(cache._cache) <= 16
    with pytest.raises(ValueError):
        FrameCache(src, 100, size=4, backfill=4)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def foot_points(kp, frame):
    names = [n for n in kp.names if any(p in n for p in ("ANKLE", "HEEL", "FOOT_INDEX", "TOE"))]
    i = int(np.flatnonzero(kp.frame_idx == frame)[0])
    return np.array([kp.point(n)[i] for n in names])


def test_foot_crop_box_contains_both_feet_with_frame_aspect():
    walk = simulate_walk(duration_s=3.0, pan_speed_mps=1.3)
    kp = walk.keypoints
    w, h = kp.meta["width"], kp.meta["height"]
    for frame in (10, 60, 120):
        box = foot_crop_box(kp, frame, (w, h), window=0)
        x0, y0, x1, y1 = box
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
        assert (x1 - x0) / (y1 - y0) == pytest.approx(w / h, rel=0.01)
        assert (x1 - x0) <= w / 2 + 1  # 2x zoom when the feet fit
        pts = foot_points(kp, frame)
        assert np.all(pts[:, 0] >= x0) and np.all(pts[:, 0] <= x1)
        assert np.all(pts[:, 1] >= y0) and np.all(pts[:, 1] <= y1)


def test_foot_crop_box_rescales_and_handles_missing():
    kp = simulate_walk(duration_s=1.0).keypoints
    w, h = kp.meta["width"], kp.meta["height"]
    full = foot_crop_box(kp, 30, (w, h))
    half = foot_crop_box(kp, 30, (w // 2, h // 2))  # video at half the keypoints' size
    assert half == pytest.approx(tuple(v / 2 for v in full), abs=2)
    assert foot_crop_box(kp, 10_000, (w, h)) is None
    blank = kp.subset(kp.names)
    blank.xy[:] = np.nan
    blank.conf[:] = 0.0
    assert foot_crop_box(blank, 30, (w, h)) is None
    no_feet = kp.subset(["NOSE", "LEFT_HIP"])
    assert foot_crop_box(no_feet, 30, (w, h)) is None


def test_timeline_mapping_round_trip():
    n, x0, x1 = 2400, 12, 1268
    for f in (0, 1, 500, 1199, 2399):
        assert abs(x_to_frame(frame_to_x(f, n, x0, x1), n, x0, x1) - f) <= 1
    assert frame_to_x(0, n, x0, x1) == x0 and frame_to_x(n - 1, n, x0, x1) == x1
    assert x_to_frame(-50, n, x0, x1) == 0 and x_to_frame(5000, n, x0, x1) == n - 1
    assert x_to_frame(100, 1, x0, x1) == 0


def test_display_size_fits_and_never_enlarges():
    assert display_size((1920, 1080)) == (1280, 720)
    assert display_size((1280, 800), (1280, 720)) == (1152, 720)
    assert display_size((640, 480)) == (640, 480)


def test_file_helpers(tmp_path):
    out = tmp_path / "labels" / "clip__bob.csv"
    assert autosave_path(out).name == "clip__bob.csv.autosave"
    assert sidecar_path(out).name == "clip__bob.json"


# ---------------------------------------------------------------------------
# OpenCV-dependent pieces (no windows are opened)
# ---------------------------------------------------------------------------


def test_render_draws_without_a_window():
    pytest.importorskip("cv2")
    from gaitbench.labelling.app import render

    s = vfr_session()
    s.goto(50)
    v = ViewState()
    img = np.full((1080, 1920, 3), 90, np.uint8)
    canvas = render(s, img, v)
    lay = canvas_layout(canvas.shape)
    assert canvas.shape[1] == 1280 and canvas.shape[0] == 720 + lay.panel_h
    assert lay.image_h == 720 and lay.whole_y[0] > 720
    v.help, v.zoom = True, True
    v.set_message("saved")
    zoomed = render(s, img, v, crop=(400, 300, 1360, 840), note="times are nominal")
    assert zoomed.shape == canvas.shape
    missing = render(s, None, ViewState(), frame_size=(640, 480))
    assert missing.shape[1] == 640
    assert img.mean() == 90  # the source frame is not drawn on


def write_test_video(path, n=90, size=(160, 120), fps=30.0):
    """Video whose frame i has grey level 2*i, for checking frame-accurate reads."""
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4v here")
    for i in range(n):
        writer.write(np.full((size[1], size[0], 3), 2 * i, np.uint8))
    writer.release()
    return path


def test_video_source_and_cache_read_the_right_frames(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from gaitbench.labelling.app import _VideoSource

    path = write_test_video(tmp_path / "v.mp4")
    times, source_name, container = load_timestamps(path)
    assert source_name == "video" and len(times) == 90
    assert times[30] == pytest.approx(1.0, abs=1e-3)
    src = _VideoSource(cv2, path, container_times=container)
    cache = FrameCache(src, 90, size=20, backfill=5)
    for i in (0, 1, 2, 60, 59, 58, 20, 89, 45, 44):
        img = cache.get(i)
        assert img is not None
        assert abs(int(img[60, 80, 1]) - 2 * i) <= 3, i
    assert src.exact_seek
    src.release()


def test_video_source_falls_back_when_seeks_are_inexact(tmp_path):
    """OpenCV seeks a VFR file by index/fps and lands on the wrong frame; the reader must notice."""
    import shutil
    import subprocess

    cv2 = pytest.importorskip("cv2")
    from gaitbench.labelling.app import _VideoSource

    if shutil.which("ffmpeg") is None:
        pytest.skip("needs ffmpeg to make a variable-frame-rate video")
    cfr = write_test_video(tmp_path / "cfr.mp4", n=90)
    vfr = tmp_path / "vfr.mp4"
    # Frames from 30 on are shown 1 s later: a gap, as when an iPhone drops frames.
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(cfr), "-vf", "setpts='(N+gte(N\\,30)*30)/30/TB'",
         "-fps_mode", "vfr", "-c:v", "libx264", "-g", "15", "-pix_fmt", "yuv420p", str(vfr)],
        capture_output=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"ffmpeg could not make the test video: {proc.stderr[-200:]!r}")
    times, source_name, container = load_timestamps(vfr)
    assert source_name == "video" and times[30] - times[29] == pytest.approx(31 / 30, abs=1e-3)
    src = _VideoSource(cv2, vfr, container_times=container)
    cache = FrameCache(src, len(times), size=20, backfill=5)
    with pytest.warns(UserWarning, match="Seeking is inexact"):
        for i in (70, 69, 50, 10, 85):
            assert abs(int(cache.get(i)[60, 80, 1]) - 2 * i) <= 3, i
    assert not src.exact_seek
    src.release()


def test_load_timestamps_prefers_a_frames_csv(tmp_path):
    from gaitbench.capture.realsense import frame_times_path, frame_times_table, write_frame_times

    video = tmp_path / "rs_clip.mp4"
    video.write_bytes(b"not really a video")
    rows = [{"frame": i, "rs_frame_number": n, "timestamp_ms": 5000.0 + 1000.0 * t}
            for i, (n, t) in enumerate([(10, 0.0), (11, 0.0333), (13, 0.1), (14, 0.1333)])]
    write_frame_times(frame_times_table(rows), frame_times_path(video))
    times, source, container = load_timestamps(video)
    assert source == "realsense:rs_clip.frames.csv" and container is None
    np.testing.assert_allclose(times, [0.0, 0.0333, 0.1, 0.1333])


# ---------------------------------------------------------------------------
# The whole labeller loop, with OpenCV's window functions replaced by a script
# ---------------------------------------------------------------------------


class ScriptedGUI:
    """Stands in for cv2's window functions: feeds key codes, records what was shown."""

    def __init__(self, keys, crash_after=None):
        self.keys = list(keys)
        self.crash_after = crash_after
        self.n_keys = 0
        self.shown = []
        self.destroyed = False

    def install(self, monkeypatch, cv2):
        monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
        monkeypatch.setattr(cv2, "setMouseCallback", lambda *a, **k: None)
        monkeypatch.setattr(cv2, "imshow", lambda name, img: self.shown.append(img.shape))
        monkeypatch.setattr(cv2, "getWindowProperty", lambda *a: 1.0)
        monkeypatch.setattr(cv2, "destroyWindow", lambda *a: setattr(self, "destroyed", True))
        monkeypatch.setattr(cv2, "waitKey", lambda *a: -1)
        monkeypatch.setattr(cv2, "waitKeyEx", self.wait_key)

    def wait_key(self, delay=0):
        if self.crash_after is not None and self.n_keys >= self.crash_after:
            raise KeyboardInterrupt
        if not self.keys:
            return ord("q")
        self.n_keys += 1
        k = self.keys.pop(0)
        return k if isinstance(k, int) else ord(k)


def test_run_labeller_saves_resumes_and_recovers(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    import json

    from gaitbench.labelling import run_labeller

    video = write_test_video(tmp_path / "walk.mp4", n=90)
    out = tmp_path / "labels" / "walk__alice.csv"

    # First session: label, save with Enter, keep going, quit with q.
    gui = ScriptedGUI(["D", "D", "1", "d", "d", "4", 13, "D", "3", "a", "2", "h", "z", "q"])
    gui.install(monkeypatch, cv2)
    assert run_labeller(video, out, "alice") == out
    labels = read_labels(out)
    assert list(zip(labels["side"], labels["event"], labels["frame"])) == [
        ("L", "HS", 20), ("L", "TO", 31), ("R", "HS", 32), ("R", "TO", 22)
    ]
    assert set(labels["clip_id"]) == {"walk"} and set(labels["detector"]) == {"alice"}
    assert labels.loc[labels["frame"] == 20, "time_s"].iloc[0] == pytest.approx(20 / 30.0, abs=1e-3)
    meta = json.loads(sidecar_path(out).read_text())
    assert meta["timestamps_source"] == "video" and meta["n_events"] == 4 and meta["blind"] is True
    assert gui.shown and gui.destroyed
    assert not autosave_path(out).exists()

    # Second session resumes the file; start at frame 31 and delete the L TO there.
    gui = ScriptedGUI(["x", "q"])
    gui.install(monkeypatch, cv2)
    run_labeller(video, out, "alice", start_frame=31)
    assert len(read_labels(out)) == 3

    # Third session crashes after 25 edits: the autosave holds them and the next run resumes from it.
    keys = []
    for _ in range(25):
        keys += ["d", "3"]  # each "3" a frame later moves the R HS: one edit per press
    gui = ScriptedGUI(keys, crash_after=len(keys))
    gui.install(monkeypatch, cv2)
    with pytest.raises(KeyboardInterrupt):
        run_labeller(video, out, "alice", start_frame=40)
    assert autosave_path(out).exists()
    assert len(read_labels(out)) == 3  # the real file is untouched

    gui = ScriptedGUI(["q"])
    gui.install(monkeypatch, cv2)
    run_labeller(video, out, "alice")
    final = read_labels(out)
    assert len(final) == 4 and final["frame"].max() == 65
    assert not autosave_path(out).exists()


def test_run_labeller_quit_without_labels_writes_nothing(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    from gaitbench.labelling import run_labeller

    video = write_test_video(tmp_path / "walk.mp4", n=30)
    gui = ScriptedGUI(["d", "d", "q"])
    gui.install(monkeypatch, cv2)
    assert run_labeller(video, tmp_path / "out.csv", "bob") is None
    assert not (tmp_path / "out.csv").exists()


def test_run_labeller_rejects_another_raters_file(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    from gaitbench.labelling import run_labeller

    video = write_test_video(tmp_path / "walk.mp4", n=30)
    s = LabelSession("walk", "bob", 30, fps=30)
    s.add("L", "HS")
    out = s.save(tmp_path / "walk__bob.csv")
    ScriptedGUI(["q"]).install(monkeypatch, cv2)
    with pytest.raises(FormatError, match="rater"):
        run_labeller(video, out, "alice")
    with pytest.raises(FileNotFoundError):
        run_labeller(tmp_path / "missing.mp4", out, "bob")


def test_run_labeller_zoom_uses_keypoints(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    from gaitbench.labelling import run_labeller

    video = write_test_video(tmp_path / "walk.mp4", n=60, size=(320, 180))
    walk = simulate_walk(duration_s=1.0, fps=30.0, width=320, height=180, px_per_m=60.0, clip_id="walk")
    kp_path = walk.keypoints.save(tmp_path / "walk__synthetic.npz")
    gui = ScriptedGUI(["z", "d", "1", "q"])
    gui.install(monkeypatch, cv2)
    out = run_labeller(video, tmp_path / "walk__carol.csv", "carol", keypoints_path=kp_path)
    assert read_labels(out)["frame"].tolist() == [1]
    assert len({shape for shape in gui.shown}) == 1  # zoom keeps the canvas size


# ---------------------------------------------------------------------------
# Frame-time tolerances and the OpenCV repair hint (review findings)
# ---------------------------------------------------------------------------


def tick600(n: int = 480) -> np.ndarray:
    """240 fps in a 1/600 s time base: 2- and 3-tick intervals (3.33 and 5 ms)."""
    return np.floor(np.arange(n) * 2.5) / 600


@pytest.mark.parametrize(
    "ts",
    [tick600(480), tick600(479), np.r_[0.0, 1 / 30, 1 / 30 + 0.01, 2 / 30 + 0.01, 3 / 30 + 0.01]],
    ids=["tick600-even", "tick600-odd", "vfr-short-interval"],
)
def test_frame_time_tolerance_never_reaches_a_neighbour(ts):
    from gaitbench.labelling.state import frame_time_tolerance

    tol = frame_time_tolerance(ts)
    gaps = np.diff(ts)
    assert np.all(tol[1:] < gaps) and np.all(tol[:-1] < gaps)
    np.testing.assert_allclose(frame_time_tolerance(ts, [0, 2]), tol[[0, 2]])
    assert np.isinf(frame_time_tolerance([0.5])).all()


def test_load_warns_when_label_times_are_an_adjacent_frames(tmp_path):
    """Half the median interval (16.7 ms here) would accept the next frame's time, 10 ms away."""
    ts = np.r_[0.0, 1 / 30, 1 / 30 + 0.01, np.arange(3, 30) / 30 + 0.01]
    s = LabelSession("c", "bob", timestamps=ts)
    s.goto(1)
    s.add("L", "HS")
    df = s.to_events_df()
    df["time_s"] = ts[2]  # the times of a timeline one frame off
    path = write_labels(df, tmp_path / "off.csv")
    with pytest.warns(UserWarning, match="differ from this video's frame times"):
        LabelSession("c", "bob", timestamps=ts).load(path)


def test_local_timeline_width_uses_the_mean_rate():
    from gaitbench.labelling.app import _local_halfwidth

    for n in (480, 479):
        assert _local_halfwidth(LabelSession("c", "r", timestamps=tick600(n))) == 180  # 0.75 s at 240 fps
    assert _local_halfwidth(LabelSession("c", "r", timestamps=[0.0])) == 20


class FakeCapture:
    """cv2.VideoCapture over known frame times; a seek to frame i lands on frame i + ``seek_error``."""

    def __init__(self, times, seek_error=0):
        self.times, self.seek_error, self.pos = np.asarray(times), seek_error, 0

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == "POS_MSEC":
            return 1000.0 * self.times[max(self.pos - 1, 0)]
        return {"FPS": 30.0, "COUNT": len(self.times)}.get(prop, 0.0)

    def set(self, prop, value):
        self.pos = int(value) + (self.seek_error if value else 0)

    def read(self):
        self.pos += 1
        return self.pos <= len(self.times), np.full((4, 4, 3), self.pos - 1, np.uint8)

    def grab(self):
        return self.read()[0]

    def release(self):
        pass


def test_video_source_rejects_a_seek_to_an_adjacent_frame():
    from types import SimpleNamespace

    from gaitbench.labelling.app import _VideoSource

    ts = np.r_[np.arange(0, 20) / 30, 19 / 30 + 0.01, np.arange(20, 40) / 30 + 0.01]
    for error, exact in ((0, True), (1, False)):
        cv2 = SimpleNamespace(CAP_PROP_FPS="FPS", CAP_PROP_FRAME_COUNT="COUNT", CAP_PROP_POS_MSEC="POS_MSEC",
                              CAP_PROP_POS_FRAMES="POS_FRAMES", VideoCapture=lambda path: FakeCapture(ts, error))
        src = _VideoSource(cv2, "fake.mp4", container_times=ts)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            src.seek(19)  # landing on frame 20 is only 10 ms off, under half the median interval
        assert src.exact_seek is exact
        assert any("Seeking is inexact" in str(w.message) for w in caught) is not exact
        assert int(src.read()[0, 0, 0]) == 19


def test_run_labeller_window_error_gives_the_opencv_repair(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    from gaitbench import env
    from gaitbench.labelling import run_labeller

    video = write_test_video(tmp_path / "walk.mp4", n=10)

    def no_window(*args, **kwargs):
        raise cv2.error("The function is not implemented. Rebuild the library with GTK+ 2.x or Cocoa support")

    monkeypatch.setattr(cv2, "namedWindow", no_window)
    monkeypatch.setattr(env, "package_version", lambda name: "5.0.0.93" if name.startswith("opencv") else None)
    with pytest.raises(RuntimeError, match=r"pip install --force-reinstall --no-deps opencv-python==5\.0\.0\.93"):
        run_labeller(video, tmp_path / "walk__a.csv", "a")

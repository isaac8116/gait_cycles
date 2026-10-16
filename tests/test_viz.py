"""Tests for gaitbench.viz (headless; no pyplot window is ever opened)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from matplotlib.figure import Figure

from gaitbench.formats import make_events
from gaitbench.synthetic import simulate_walk
from gaitbench.viz import foot_minus_pelvis, plot_signals, walking_direction


@pytest.fixture(scope="module")
def walk():
    return simulate_walk(duration_s=4.0, direction=-1, pan_speed_mps=1.3, clip_id="viz")


def test_plot_with_detectors_and_truth(walk, tmp_path: Path) -> None:
    truth = walk.truth
    shifted = truth.assign(time_s=truth["time_s"] + 0.01, detector="late")
    out = tmp_path / "sub" / "plot.png"
    fig = plot_signals(walk.keypoints, {"exact": truth, "late": shifted}, truth=truth, out_path=out)
    assert isinstance(fig, Figure)
    assert out.is_file() and out.stat().st_size > 10_000
    assert len(fig.axes) == 4  # signals + raster for each side
    raster = fig.axes[1]
    assert [t.get_text() for t in raster.get_yticklabels()] == ["exact", "late"]
    # One vertical truth line per left-side event in both left panels.
    n_left = int((truth["side"] == "L").sum())
    assert len(fig.axes[0].lines) == 3 + n_left  # heel, toe, zero line + truth


def test_plot_without_events(walk) -> None:
    fig = plot_signals(walk.keypoints, {})
    assert len(fig.axes) == 2
    assert "right-to-left" in fig._suptitle.get_text()


def test_events_of_other_clips_are_ignored_and_frames_used(walk) -> None:
    other = make_events([{"side": "L", "event": "HS", "frame": 10, "time_s": 0.1}], defaults={"clip_id": "other"})
    no_time = make_events([{"side": "L", "event": "HS", "frame": 30}], defaults={"clip_id": "viz"})
    fig = plot_signals(walk.keypoints, {"other": other, "frames": no_time})
    raster = fig.axes[1]
    xs = [line.get_xdata() for line in raster.lines if len(line.get_xdata())]
    assert len(xs) == 1 and xs[0][0] == pytest.approx(0.5)  # frame 30 at 60 fps


def test_signal_sign_and_direction(walk) -> None:
    direction, method = walking_direction(walk.keypoints)
    if method != "raw":
        assert direction == -1
    heel, point = foot_minus_pelvis(walk.keypoints, "L", "heel", direction=-1)
    assert point == "LEFT_HEEL"
    hs = walk.truth[(walk.truth["side"] == "L") & (walk.truth["event"] == "HS")]["frame"].to_numpy()
    # At heel strike the heel is in front of the pelvis: positive once sign-corrected.
    assert np.all(heel[hs] > 0)


def test_ankle_fallback(walk) -> None:
    kp = walk.keypoints.subset([n for n in walk.keypoints.names if "HEEL" not in n and "TOE" not in n
                                and "FOOT" not in n])
    signal, point = foot_minus_pelvis(kp, "R", "toe")
    assert point == "RIGHT_ANKLE" and np.isfinite(signal).any()
    fig = plot_signals(kp, {}, direction=1, title="ankles only")
    assert fig._suptitle.get_text() == "ankles only"

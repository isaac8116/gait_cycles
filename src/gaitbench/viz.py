"""Diagnostic plot: the foot signals coordinate-based detectors use, with every detector's events.

For each side the plot shows how far the heel and the toe are in front of the
pelvis along the walking direction, in pixels. Zeni's rule puts heel strike
at the heel's forward peak and toe off at the toe's backward trough, so a
glance shows whether events sit where the signal says they should. Under each
side's signals a raster gives one row per detector (heel strikes as circles,
toe offs as triangles), so many detectors can be compared at once without a
colour per detector; ground truth is drawn as vertical lines through both.

The figure is built with :class:`matplotlib.figure.Figure` directly, not
pyplot, so it never opens a window or changes the global backend: it is safe
in headless runs and test suites. In a notebook, show it with
``IPython.display.display(fig)``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .formats import SIDES, Keypoints, frames_to_times

# Reference categorical slots 1 and 2 (blue, orange); heel strike shares the
# heel's colour and toe off the toe's, so a marker's colour names its signal.
HEEL_COLOR = "#2a78d6"
TOE_COLOR = "#eb6834"
TRUTH_COLOR = "#52514e"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
SURFACE = "#fcfcfb"

_SIDE_NAMES = {"L": "Left", "R": "Right"}
_CHAINS = {"heel": ("HEEL", "ANKLE"), "toe": ("FOOT_INDEX", "BIG_TOE", "ANKLE")}


def walking_direction(kp: Keypoints, min_conf: float = 0.3) -> tuple[int, str]:
    """Walking direction (+1 left-to-right) and how it was found.

    Uses :func:`gaitbench.detectors.base.estimate_direction` when the
    detectors package is importable, so the plot agrees with the detectors;
    otherwise returns ``(1, "raw")`` and the signals are not sign-corrected.
    """
    try:
        from .detectors.base import estimate_direction
    except ImportError:
        return 1, "raw"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = estimate_direction(kp, min_conf=min_conf)
    if isinstance(result, tuple):
        direction, info = result
        method = str(info.get("method", "estimated")) if isinstance(info, Mapping) else "estimated"
    else:
        direction, method = result, "estimated"
    return int(direction), method


def foot_minus_pelvis(
    kp: Keypoints, side: str, part: str, direction: int = 1, min_conf: float = 0.3
) -> tuple[np.ndarray, str | None]:
    """``direction * (foot_x - pelvis_x)`` for one side, NaN where a point is missing.

    Parameters
    ----------
    kp:
        Keypoints of one clip.
    side:
        ``"L"`` or ``"R"``.
    part:
        ``"heel"`` (HEEL, else ANKLE) or ``"toe"`` (FOOT_INDEX, BIG_TOE, else ANKLE).
    direction:
        +1 or -1; positive values then mean "in front of the pelvis".
    min_conf:
        Points below this confidence count as missing.

    Returns
    -------
    signal, point:
        ``(n_frames,)`` pixels, and the keypoint actually used (None if none).
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    point = _choose_point(kp, side, part, min_conf)
    pelvis = _pelvis_x(kp, min_conf)
    if point is None or pelvis is None:
        return np.full(kp.n_frames, np.nan), point
    return direction * (kp.masked_point(point, min_conf)[:, 0] - pelvis), point


def plot_signals(
    kp: Keypoints,
    events_by_detector: Mapping[str, pd.DataFrame],
    truth: pd.DataFrame | None = None,
    out_path: str | Path | None = None,
    *,
    min_conf: float = 0.3,
    direction: int | None = None,
    title: str | None = None,
) -> Figure:
    """Plot heel and toe signals per side with detector events and truth.

    Parameters
    ----------
    kp:
        Keypoints of one clip.
    events_by_detector:
        ``{label: events table}``; rows for other clips are ignored. Each
        label gets one raster row per side.
    truth:
        Optional labels table, drawn as vertical lines (solid HS, dashed TO).
    out_path:
        If given, the figure is also saved there (format from the suffix).
    min_conf:
        Points below this confidence are not drawn.
    direction:
        Walking direction; estimated if None (see :func:`walking_direction`).
    title:
        Figure title; defaults to clip, backend and direction.

    Returns
    -------
    matplotlib.figure.Figure
    """
    method = "given"
    if direction is None:
        direction, method = walking_direction(kp, min_conf)
    labels = list(events_by_detector)
    t = kp.time_s
    truth = _for_clip(truth, kp) if truth is not None else None

    n_rows = len(labels)
    raster_h = 0.3 + 0.24 * n_rows if n_rows else 0.0
    ratios: list[float] = []
    for _ in SIDES:
        ratios += [2.4, raster_h] if n_rows else [2.4]
    fig = Figure(figsize=(11, sum(ratios) + 1.0), facecolor=SURFACE, layout="constrained")
    grid = fig.add_gridspec(len(ratios), 1, height_ratios=ratios)

    axes = []
    row = 0
    for side in SIDES:
        ax = fig.add_subplot(grid[row], sharex=axes[0] if axes else None)
        axes.append(ax)
        row += 1
        _style(ax)
        points = {}
        for part, color in (("heel", HEEL_COLOR), ("toe", TOE_COLOR)):
            signal, point = foot_minus_pelvis(kp, side, part, direction, min_conf)
            points[part] = point
            ax.plot(t, signal, color=color, linewidth=1.5)
        ax.axhline(0.0, color=GRID_COLOR, linewidth=1.0, zorder=0)
        ax.set_ylabel("in front of pelvis (px)", color=TEXT_SECONDARY, fontsize=9)
        used = ", ".join(p for p in points.values() if p) or "no foot points"
        ax.set_title(f"{_SIDE_NAMES[side]} ({used})", loc="left", color=TEXT_PRIMARY, fontsize=10)
        side_axes = [ax]

        if n_rows:
            rax = fig.add_subplot(grid[row], sharex=axes[0])
            axes.append(rax)
            row += 1
            _style(rax)
            rax.grid(False, axis="y")
            for i, label in enumerate(labels):
                ev = _for_clip(events_by_detector[label], kp)
                ev = ev[ev["side"] == side]
                for event, marker, color in (("HS", "o", HEEL_COLOR), ("TO", "^", TOE_COLOR)):
                    times = _event_times(ev[ev["event"] == event], kp)
                    rax.plot(times, np.full(times.shape, i), linestyle="none", marker=marker,
                             markersize=6, color=color, markeredgecolor=SURFACE, markeredgewidth=0.8)
            rax.set_yticks(range(n_rows), labels, fontsize=8, color=TEXT_SECONDARY)
            rax.set_ylim(n_rows - 0.5, -0.5)
            side_axes.append(rax)

        if truth is not None:
            side_truth = truth[truth["side"] == side]
            for event, style in (("HS", "-"), ("TO", "--")):
                for x in _event_times(side_truth[side_truth["event"] == event], kp):
                    for a in side_axes:
                        a.axvline(x, color=TRUTH_COLOR, linestyle=style, linewidth=1.0, alpha=0.6, zorder=1)

    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("time (s)", color=TEXT_SECONDARY, fontsize=9)
    if len(t):
        axes[0].set_xlim(float(t[0]), float(t[-1]))

    handles = [
        Line2D([], [], color=HEEL_COLOR, linewidth=1.5, label="heel - pelvis"),
        Line2D([], [], color=TOE_COLOR, linewidth=1.5, label="toe - pelvis"),
    ]
    if n_rows:
        handles += [
            Line2D([], [], linestyle="none", marker="o", markersize=6, color=HEEL_COLOR, label="detected HS"),
            Line2D([], [], linestyle="none", marker="^", markersize=6, color=TOE_COLOR, label="detected TO"),
        ]
    if truth is not None:
        handles += [
            Line2D([], [], color=TRUTH_COLOR, linestyle="-", linewidth=1.0, label="truth HS"),
            Line2D([], [], color=TRUTH_COLOR, linestyle="--", linewidth=1.0, label="truth TO"),
        ]
    fig.legend(handles=handles, loc="outside upper right", ncol=len(handles), frameon=False, fontsize=8,
               labelcolor=TEXT_SECONDARY)
    walk = "left-to-right" if direction == 1 else "right-to-left"
    backend = kp.meta.get("backend", "")
    fig.suptitle(title or f"{kp.clip_id}  {backend}  walking {walk} ({method})", x=0.01, ha="left",
                 color=TEXT_PRIMARY, fontsize=11)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110, facecolor=SURFACE)
    return fig


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def _choose_point(kp: Keypoints, side: str, part: str, min_conf: float) -> str | None:
    try:
        from .detectors.base import choose_foot_point
    except ImportError:
        choose_foot_point = None
    if choose_foot_point is not None:
        return choose_foot_point(kp, side, part, min_conf)
    prefix = "LEFT_" if side == "L" else "RIGHT_"
    for p in _CHAINS[part]:
        name = prefix + p
        if kp.has(name) and np.any(kp.confidence(name) >= min_conf):
            return name
    return None


def _pelvis_x(kp: Keypoints, min_conf: float) -> np.ndarray | None:
    """The same pelvis the detectors use, so the plotted signal is the one they saw."""
    if not (kp.has("LEFT_HIP") or kp.has("RIGHT_HIP")):
        return None
    try:
        from .detectors.base import pelvis_x
    except ImportError:  # plotting must work without the detectors package
        hips = [kp.masked_point(n, min_conf)[:, 0] for n in ("LEFT_HIP", "RIGHT_HIP") if kp.has(n)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # frames with neither hip
            return np.nanmean(np.stack(hips), axis=0)
    return pelvis_x(kp, min_conf)


def _for_clip(events: pd.DataFrame, kp: Keypoints) -> pd.DataFrame:
    """Rows of ``events`` for this clip; a table with no clip ids is taken as all for this clip."""
    if "clip_id" not in events.columns:
        return events
    ids = events["clip_id"].fillna("").astype(str)
    if (ids == "").all():
        return events
    return events[ids == kp.clip_id]


def _event_times(events: pd.DataFrame, kp: Keypoints) -> np.ndarray:
    if events.empty:
        return np.array([], dtype=float)
    if "time_s" in events.columns:
        # copy=True: with pandas copy-on-write the array may be a read-only view.
        times = pd.to_numeric(events["time_s"], errors="coerce").to_numpy(dtype=float, copy=True)
    else:
        times = np.full(len(events), np.nan)
    missing = ~np.isfinite(times)
    if missing.any():
        times[missing] = frames_to_times(events["frame"].to_numpy()[missing], kp)
    return times

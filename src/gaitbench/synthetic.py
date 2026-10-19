"""A synthetic side-view walker with known heel strikes and toe offs.

It exists to test the harness itself: does a detector find every event, does
it get left and right the right way round, does it survive the camera panning
to follow the walker, a reversed walking direction, noise and dropped points.
It is a kinematic stick figure, not a biomechanical model, so **accuracy
numbers measured on it say nothing about accuracy on real people**.

Model, in world coordinates (metres, X along the walkway, Y up, ground at 0):

* The pelvis moves forward at constant speed with a small vertical bob. By
  default both hips sit on the pelvis point; ``hip_offset_m`` moves them
  apart the way perspective does in a real side view.
* Each foot is rigid. In stance it stays where it landed. In swing it moves
  one stride forward along a smooth profile whose speed rises and falls
  steeply at the ends, and lifts off the ground by a few centimetres. By
  default it stays flat through stance, so heel, ankle and toe signals have
  the same shape; ``heel_off_frac`` adds a heel rise (the foot rolls about
  the toe) before toe off, so they differ as on real feet.
* Heel strike is the instant a foot lands, toe off the instant it lifts.
  Left and right are half a cycle apart.
* Knees come from two-link inverse kinematics; trunk, head and arms are
  attached so that pose-based tools (myogait) get a full figure.

The camera looks at the walker side-on and may translate along the walkway
(``pan_speed_mps``). Points that fall outside the image are marked missing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import betainc

from . import __version__
from .formats import CANONICAL_KEYPOINTS, Keypoints, make_events

# Shape parameter of the swing profile (regularised incomplete beta). Values
# below 1 make the foot speed rise and fall steeply at toe off and heel strike.
_SWING_SHAPE = 1.3

# Segment proportions as fractions of body height (Winter-style approximations).
_HIP_HEIGHT = 0.53
_THIGH = 0.245
_SHANK = 0.246
_FOOT = 0.15
_ANKLE_HEIGHT = 0.045
_SHOULDER_ABOVE_HIP = 0.30
_NOSE_ABOVE_HIP = 0.40
_UPPER_ARM = 0.186
_FOREARM = 0.146
# Heel rise (WalkerParams.heel_off_frac): foot angle at toe off, and how the
# angle grows from heel off (angle ~ progress ** power, fastest at push-off).
_HEEL_RISE_DEG = 35.0
_HEEL_RISE_POWER = 2.0


@dataclass(frozen=True)
class WalkerParams:
    """Settings for :func:`simulate_walk`. Defaults resemble a young adult at self-selected pace."""

    clip_id: str = "synthetic"
    fps: float = 60.0
    duration_s: float = 5.0
    speed_mps: float = 1.3
    stride_time_s: float = 1.05
    stance_frac: float = 0.60
    height_m: float = 1.75
    swing_clearance_m: float = 0.06
    # +1: walks left-to-right in the image; -1: right-to-left.
    direction: int = 1
    # Camera translation along the walkway. Equal to speed_mps keeps the walker centred.
    pan_speed_mps: float = 0.0
    # Pelvis position relative to the camera centre at t = 0, in metres along the
    # walking direction. None centres the walker in the frame at mid-clip.
    start_offset_m: float | None = None
    px_per_m: float = 250.0
    width: int = 1920
    height: int = 1080
    # Phase of the right foot at t = 0 (0 = heel strike). The left foot is half a cycle later.
    right_phase0: float = 0.25
    near_side: str = "L"
    near_side_conf: float = 0.95
    far_side_conf: float = 0.85
    noise_px: float = 0.0
    dropout_prob: float = 0.0
    seed: int = 0
    # How far apart the two hips appear along the walkway, in metres, when the
    # walker is at the edge of the image. Perspective puts the near hip further
    # from the image centre than the far one, so the gap grows with distance
    # from the centre and changes sign as the walker crosses it. 0 puts both
    # hips on the pelvis point.
    hip_offset_m: float = 0.0
    # Heel off as a fraction of stance. From then until toe off the foot rolls
    # about the toe, raising the heel, and it flattens again in early swing.
    # None keeps the foot flat through stance (a rigid foot).
    heel_off_frac: float | None = None

    def __post_init__(self) -> None:
        if self.direction not in (1, -1):
            raise ValueError("direction must be +1 or -1")
        if self.near_side not in ("L", "R"):
            raise ValueError("near_side must be 'L' or 'R'")
        if not 0.0 < self.stance_frac < 1.0:
            raise ValueError("stance_frac must be between 0 and 1")
        if self.heel_off_frac is not None and not 0.0 <= self.heel_off_frac < 1.0:
            raise ValueError("heel_off_frac must be None or in [0, 1)")
        if self.fps <= 0 or self.duration_s <= 0 or self.stride_time_s <= 0:
            raise ValueError("fps, duration_s and stride_time_s must be positive")

    @property
    def stride_length_m(self) -> float:
        return self.speed_mps * self.stride_time_s


@dataclass
class SyntheticWalk:
    keypoints: Keypoints
    truth: pd.DataFrame
    params: WalkerParams
    # Noise-free world coordinates (metres) of every keypoint: (n_frames, n_keypoints, 2).
    world_xy: np.ndarray
    camera_x_m: np.ndarray


def simulate_walk(params: WalkerParams | None = None, **overrides: Any) -> SyntheticWalk:
    """Simulate one pass and return its keypoints and ground-truth events.

    ``simulate_walk(pan_speed_mps=1.3, fps=30)`` is shorthand for
    ``simulate_walk(replace(WalkerParams(), pan_speed_mps=1.3, fps=30))``.
    """
    p = replace(params or WalkerParams(), **overrides)
    n = int(round(p.duration_s * p.fps))
    t = np.arange(n) / p.fps
    T, s, v, h = p.stride_time_s, p.stance_frac, p.speed_mps, p.height_m

    start = p.start_offset_m
    if start is None:
        start = -(v - p.pan_speed_mps) * p.duration_s / 2.0
    pelvis_x = start + v * t
    pelvis_y = _HIP_HEIGHT * h + 0.015 * np.cos(4.0 * np.pi * t / T)
    camera_x = p.pan_speed_mps * t

    # Left minus right hip position along the walkway (see WalkerParams.hip_offset_m).
    half_view_m = p.width / 2.0 / p.px_per_m
    near_sign = 1.0 if p.near_side == "L" else -1.0
    hip_gap = near_sign * p.hip_offset_m * np.clip((pelvis_x - camera_x) / half_view_m, -1.0, 1.0)

    world: dict[str, np.ndarray] = {}
    for side, phase0 in (("R", p.right_phase0), ("L", p.right_phase0 + 0.5)):
        heel = _foot_heel(t, phase0, p, start)
        prefix = "RIGHT_" if side == "R" else "LEFT_"
        foot = _FOOT * h
        lift = heel[:, 1]
        ankle = np.column_stack([heel[:, 0] + 0.25 * foot, lift + _ANKLE_HEIGHT * h])
        toe = np.column_stack([heel[:, 0] + foot, 0.6 * lift])
        if p.heel_off_frac is not None:
            heel, ankle = _roll_about_toe(toe[:, 0], lift, _heel_roll(t, phase0, p), foot, _ANKLE_HEIGHT * h)
        world[prefix + "HEEL"] = heel
        world[prefix + "ANKLE"] = ankle
        world[prefix + "FOOT_INDEX"] = toe
        world[prefix + "BIG_TOE"] = toe.copy()
        world[prefix + "SMALL_TOE"] = toe - np.array([0.02, 0.0])
        hip_x = pelvis_x + (0.5 if side == "L" else -0.5) * hip_gap if p.hip_offset_m else pelvis_x
        hip = np.column_stack([hip_x, pelvis_y])
        world[prefix + "HIP"] = hip
        world[prefix + "KNEE"] = _knee(hip, world[prefix + "ANKLE"], _THIGH * h, _SHANK * h)

        # Each arm swings in phase with the opposite leg.
        opposite_phase0 = phase0 + 0.5
        theta = np.deg2rad(20.0) * np.cos(2.0 * np.pi * (t / T + opposite_phase0))
        shoulder = np.column_stack([pelvis_x + 0.02, pelvis_y + _SHOULDER_ABOVE_HIP * h])
        elbow = shoulder + _UPPER_ARM * h * np.column_stack([np.sin(theta), -np.cos(theta)])
        bend = theta + np.deg2rad(15.0)
        world[prefix + "SHOULDER"] = shoulder
        world[prefix + "ELBOW"] = elbow
        world[prefix + "WRIST"] = elbow + _FOREARM * h * np.column_stack([np.sin(bend), -np.cos(bend)])
    world["NOSE"] = np.column_stack([pelvis_x + 0.08, pelvis_y + _NOSE_ABOVE_HIP * h])

    names = tuple(n_ for n_ in CANONICAL_KEYPOINTS if n_ in world)
    world_xy = np.stack([world[n_] for n_ in names], axis=1)  # (n, k, 2)

    xy = _project(world_xy, camera_x, p)
    rng = np.random.default_rng(p.seed)
    if p.noise_px > 0:
        xy = xy + rng.normal(0.0, p.noise_px, size=xy.shape)

    conf = np.empty(xy.shape[:2])
    for j, name in enumerate(names):
        near = name.startswith("LEFT_" if p.near_side == "L" else "RIGHT_")
        far = name.startswith("RIGHT_" if p.near_side == "L" else "LEFT_")
        conf[:, j] = p.near_side_conf if near else (p.far_side_conf if far else p.near_side_conf)
    missing = ~_inside(xy, p)
    if p.dropout_prob > 0:
        missing |= rng.random(conf.shape) < p.dropout_prob
    xy[missing] = np.nan
    conf[missing] = 0.0

    meta = {
        "fps": p.fps,
        "width": p.width,
        "height": p.height,
        "source": "synthetic",
        "backend": "synthetic",
        "tool": "gaitbench.synthetic",
        "tool_version": __version__,
        "walk_direction": "left_to_right" if p.direction == 1 else "right_to_left",
        "camera": {"moving": p.pan_speed_mps != 0.0, "pan_speed_mps": p.pan_speed_mps},
        "params": asdict(p),
    }
    kp = Keypoints(
        clip_id=p.clip_id,
        names=names,
        xy=xy,
        conf=conf,
        time_s=t,
        frame_idx=np.arange(n),
        meta=meta,
    )
    truth = _truth_events(p, start)
    return SyntheticWalk(keypoints=kp, truth=truth, params=p, world_xy=world_xy, camera_x_m=camera_x)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _cycle(t: np.ndarray, phase0: float, T: float) -> tuple[np.ndarray, np.ndarray]:
    c = t / T + phase0
    k = np.floor(c)
    return k, c - k


def _landing_x(k: np.ndarray, phase0: float, p: WalkerParams, start: float) -> np.ndarray:
    """Where the heel lands at the heel strike that starts cycle ``k``."""
    t_hs = (k - phase0) * p.stride_time_s
    ahead = p.speed_mps * p.stance_frac * p.stride_time_s / 2.0
    return start + p.speed_mps * t_hs + ahead


def _foot_heel(t: np.ndarray, phase0: float, p: WalkerParams, start: float) -> np.ndarray:
    k, phi = _cycle(t, phase0, p.stride_time_s)
    s = p.stance_frac
    x = _landing_x(k, phase0, p, start)
    y = np.zeros_like(t)
    swing = phi >= s
    tau = (phi[swing] - s) / (1.0 - s)
    x = x.copy()
    x[swing] += p.stride_length_m * betainc(_SWING_SHAPE, _SWING_SHAPE, tau)
    y[swing] = p.swing_clearance_m * np.sin(np.pi * tau)
    return np.column_stack([x, y])


def _heel_roll(t: np.ndarray, phase0: float, p: WalkerParams) -> np.ndarray:
    """Foot angle in radians (0 = flat, positive = heel up) for a heel rise before toe off.

    The angle grows from heel off to :data:`_HEEL_RISE_DEG` at toe off,
    fastest at the end as in push-off. It leaves toe off at that same
    angular speed (so the heel and ankle do not change speed abruptly) and
    is back to 0 by mid-swing, long before the next heel strike.
    """
    _, phi = _cycle(t, phase0, p.stride_time_s)
    s = p.stance_frac
    heel_off = p.heel_off_frac * s
    top = np.deg2rad(_HEEL_RISE_DEG)
    roll = np.zeros_like(phi)
    rising = (phi >= heel_off) & (phi < s)
    roll[rising] = top * ((phi[rising] - heel_off) / (s - heel_off)) ** _HEEL_RISE_POWER
    # Cubic Hermite curve from (top, push-off speed) at toe off to (0, 0) at mid-swing.
    swing = phi >= s
    x = np.minimum((phi[swing] - s) / (0.5 * (1.0 - s)), 1.0)
    speed = top * _HEEL_RISE_POWER / (s - heel_off) * 0.5 * (1.0 - s)
    roll[swing] = top * (2 * x**3 - 3 * x**2 + 1) + speed * (x**3 - 2 * x**2 + x)
    return roll


def _roll_about_toe(
    toe_x: np.ndarray, lift: np.ndarray, roll: np.ndarray, foot: float, ankle_height: float
) -> tuple[np.ndarray, np.ndarray]:
    """Heel and ankle of a foot turned by ``roll`` about its toe (same as the rigid foot at 0)."""
    cos, sin = np.cos(roll), np.sin(roll)
    heel = np.column_stack([toe_x - foot * cos, lift + foot * sin])
    ankle = np.column_stack([toe_x - 0.75 * foot * cos + ankle_height * sin, lift + ankle_height * cos + 0.75 * foot * sin])
    return heel, ankle


def _knee(hip: np.ndarray, ankle: np.ndarray, thigh: float, shank: float) -> np.ndarray:
    """Two-link inverse kinematics with the knee bending forward (+X)."""
    d = ankle - hip
    dist = np.linalg.norm(d, axis=1)
    reach = np.clip(dist, 1e-6, thigh + shank - 1e-9)
    # Angle at the hip between the hip->ankle line and the thigh (law of cosines).
    cos_a = np.clip((thigh**2 + reach**2 - shank**2) / (2 * thigh * reach), -1.0, 1.0)
    a = np.arccos(cos_a)
    u = d / dist[:, None]
    # Rotate u towards +X (forward) by angle a. In (X, Y-up) coordinates the
    # leg points down, so a counter-clockwise rotation moves it forward.
    rot = np.column_stack([u[:, 0] * np.cos(a) - u[:, 1] * np.sin(a), u[:, 0] * np.sin(a) + u[:, 1] * np.cos(a)])
    return hip + thigh * rot


def _project(world_xy: np.ndarray, camera_x: np.ndarray, p: WalkerParams) -> np.ndarray:
    u = p.width / 2.0 + p.direction * p.px_per_m * (world_xy[..., 0] - camera_x[:, None])
    v = 0.9 * p.height - p.px_per_m * world_xy[..., 1]
    return np.stack([u, v], axis=-1)


def _inside(xy: np.ndarray, p: WalkerParams) -> np.ndarray:
    return (xy[..., 0] >= 0) & (xy[..., 0] < p.width) & (xy[..., 1] >= 0) & (xy[..., 1] < p.height)


def _truth_events(p: WalkerParams, start: float) -> pd.DataFrame:
    """Every heel strike and toe off inside the clip whose foot point is in the image."""
    T, s = p.stride_time_s, p.stance_frac
    records = []
    for side, phase0 in (("R", p.right_phase0), ("L", p.right_phase0 + 0.5)):
        k_first = int(np.floor(phase0)) - 1
        k_last = int(np.ceil(p.duration_s / T + phase0)) + 1
        for k in range(k_first, k_last + 1):
            t_hs = (k - phase0) * T
            for event, t_ev in (("HS", t_hs), ("TO", t_hs + s * T)):
                frame = int(round(t_ev * p.fps))
                if t_ev < 0 or frame >= int(round(p.duration_s * p.fps)):
                    continue
                # Foot point at the event: the planted heel (HS) or toe (TO) at the landing spot.
                heel_x = _landing_x(np.array([float(k)]), phase0, p, start)[0]
                point_x = heel_x if event == "HS" else heel_x + _FOOT * p.height_m
                u = p.width / 2.0 + p.direction * p.px_per_m * (point_x - p.pan_speed_mps * t_ev)
                if not 0 <= u < p.width:
                    continue
                records.append({"side": side, "event": event, "frame": frame, "time_s": t_ev})
    return make_events(
        records,
        defaults={
            "clip_id": p.clip_id,
            "tool": "synthetic",
            "tool_version": __version__,
            "backend": "synthetic",
            "detector": "truth",
            "confidence": np.nan,
        },
    )

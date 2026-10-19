# Upstream Issues Found While Testing

19 Oct 2026 · bugs in myogait 0.9.1 and gaitkit 1.4.9 found by the gaitbench review

Both were found on synthetic walks and confirmed against the source code. gaitbench works around the first; the second can only be avoided, by choosing other detectors. The texts below are ready to paste into GitHub issues ([myogait](https://github.com/IDMDataHub/myogait/issues), [gaitkit](https://github.com/IDMDataHub/gaitkit/issues)). Check that the issues still exist in the latest release before filing.

## myogait 0.9.1: mixed pixel and normalised coordinates sent to gaitkit on non-square frames

**Where.** `compute_angles` sets `needs_scale` when meta width ≠ height (`angles.py:936-938`) and builds every angle frame from `_pixelify_frame(frame, width, height)` (`angles.py:967`).

**What goes wrong.**
1. `_extract_landmark_positions` (`angles.py:464`, called at `:507`) therefore stores hip, knee and ankle in **pixels**.
2. `_convert_to_gaitkit_frames` treats them as normalised: `(x, 0, 1 - y)` at `events.py:869`.
3. It then adds heel and toe from `data["frames"]` in **normalised** units (`events.py:883-896`).
4. Finally it scales everything by a femur length measured in pixels (`events.py:921-923`).

**Result.** On a 1920×1080 frame the leg is about 2000 times larger than the foot and offset from it. At frame 100 of a synthetic walk, `left_hip = (4356, 0, -5172)` while `left_heel = (2.25, 0, 0.62)`.

**Effect.** gk_bike, gk_ensemble, gk_vancanneyt and gk_dgei fail on any non-square video, which means every real camera.

| Method | Recall at 1920×1080 | Recall at 1920×1920 |
|---|---|---|
| gk_bike | 0.00 | 1.00 |
| gk_ensemble | 0.32 | 1.00 |
| gk_vancanneyt | 0.00 | 0.82 |
| gk_dgei | 0 events found | 22 events |

**Workaround used in gaitbench.** Normalise both axes by max(W, H) and declare a square frame (`gaitbench/myogait_bridge.py`, "Square pivot").

**Suggested fix.** In `_convert_to_gaitkit_frames`, take all landmark positions from one coordinate space. For example, divide `landmark_positions` back by (W, H), or pixelify the raw-frame heel and toe as well, and flip y with the matching height.

**Related, same release:**
- `segment_cycles` computes `duration = (end_frame - start_frame) / fps` (`cycles.py:239`), and `filter_cycles_by_direction` looks frames up at `frame_idx - first_idx` (`pipeline.py:83-86`). Both assume contiguous `frame_idx`, so a subsampled pivot gets no cycles.
- `_remap_event_frames` computes time from `frame_idx` and fps (`events.py:76`).
- `event_consensus` returns array positions rather than `frame_idx` (`events.py:590-600`).

## gaitkit 1.4.9: walking direction is a coin flip on panning-camera clips, so HS and TO swap

**Code path.** `_core.py:288` maps `"zeni"` to `gaitkit.zeni_native.ZeniNativeGaitDetector`, which does not exist (the module defines `ZeniNativeDetector`). gk_zeni therefore falls back to `detectors/zeni_detector.py`, which at line 163 calls `detect_walking_direction(angle_frames, ap_axis)` without fps.

**Why it fails on a pan.** A camera that follows the walker leaves the ankle's net image displacement under the 500 mm threshold (measured −44 to −81 mm). `axis_utils.detect_walking_direction` (`axis_utils.py:206-236`) then falls back to an acceleration-asymmetry score. That score is near zero (about ±0.001 at 240 fps), so its sign is set by noise.

**Effect.**
- On a synthetic 1.3 m/s pan with 2 px noise, gk_zeni recall over three seeds is 0/0/0 at 240 fps, 0/1/0 at 60 fps and 1/1/0 at 30 fps.
- When it fails it swaps the events: median error is −417 ms for heel strike and +416 ms for toe off.
- Fixed-camera recall is 0.95–1.00.
- bayesian_bis (gk_bike) flips the same way, even though it passes fps (`bayesian_bis.py:210`).

**Suggested fix.**
- Don't rely on net image displacement. Use a direction cue that survives camera motion, such as toe ahead of heel, or the sign of knee flexion.
- At least, report the direction as low-confidence instead of defaulting to +1.
- Fix the `ZeniNativeGaitDetector` name.

**Second issue.** bayesian_bis reads `lp["left_knee"]`, `["right_knee"]`, `["left_ankle"]` and `["right_ankle"]` without checking they exist (`bayesian_bis_native.py:50-53`). It also calls `lp.get` on `None` for a frame with no landmarks (`bayesian_bis_native.py:310`). It raises `KeyError` or `AttributeError` instead of treating these as gaps; gaitbench fills leg gaps before calling it.

## What this means for our tests

- **Moving-camera results.** Treat gk_zeni and gk_bike on panning clips as direction-unreliable, and don't read their 240/60/30 fps differences as frame-rate effects.
- **Consensus.** gaitbench's default consensus is now `zeni`, `oconnor` and `gk_zeni`. It keeps 0.95 recall on both fixed and panning synthetic walks; the previous `zeni`, `gk_bike`, `gk_zeni`, `gk_oconnor` fell to 0.47 when panning.

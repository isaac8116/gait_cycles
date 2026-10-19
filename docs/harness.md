# Using the Test Harness

How to run `gaitbench` on your own clips. It covers installing on a Mac, the step-by-step route for iPhone clips, and what each output file means. Why the harness is built this way is in [the evaluation plan](open-source-evaluation-plan.md); what to film is in [the recording protocol](recording-protocol.md).

```
video ──► pose ──► keypoints ──► clean ──► detect ──► events ──► score ◄── your labels
```

Every arrow is a file on disk, so each step can be rerun on its own, and every detector sees exactly the same keypoints.

## Install (MacBook, CPU only)

Needs Python 3.11 or newer and ffmpeg. With Homebrew:

```
brew install python@3.12 ffmpeg
cd gait_cycles
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,video,myogait,pose]"
gaitbench env
```

`gaitbench env` prints the installed versions. **If it warns that OpenCV has no GUI**, the labelling window will not open. Run the `pip install --force-reinstall --no-deps opencv-python==…` command it prints. Three OpenCV packages share one `cv2`, and myogait installs the window-less one. Don't uninstall any of them: they share files.

Check the install with the synthetic walker, which needs no video:

```
python -m pytest -q -m "not slow"
gaitbench run configs/smoke_synthetic.yaml
```

The first time a pose model runs, it downloads its weights: MediaPipe from Google's servers, RTMW from openmmlab.com.

**RealSense D455 (optional, Apple Silicon):** `pip install -e ".[realsense]"`. This needs macOS 15 or newer. The comment in `pyproject.toml` gives the alternatives for older macOS. Check the camera is seen with `gaitbench realsense-modes`.

## Where files go

Use one data root per batch of clips, e.g. `data/phone/`. All of `data/` is git-ignored.

```
data/raw/phone/           your original videos (IMG_1234.MOV, or renamed clip ids)
data/phone/keypoints/     <clip>__<backend>.npz            one pose model's keypoints for one clip
data/phone/events/        <clip>__<backend>__<detector>.csv heel strikes and toe offs one detector found
data/phone/labels/        <clip>__<rater>.csv              your hand labels (the truth)
data/phone/results/       score and clean outputs
```

`<clip>` is the clip id: by default the video's file name without its extension. Keep it the same for a clip's video, keypoints and labels. That is how the scorer pairs them.

Backends name the keypoints source, and the steps below create several per clip:
- `mediapipe`: the pose model on the full 240 fps video
- `mediapipe-60fps`, `mediapipe-30fps`: decimated copies
- `mediapipe-clean`, `mediapipe-60fps-clean`, …: after leg-swap cleaning

Each is scored separately, so one table compares them all.

## Route for iPhone slow-motion clips

The example uses one clip, `p01_F240_LtoR_normal_01`. Steps 1–4 can run on a whole folder; step 5 is the slow, manual part.

**0. Name the clips from your session log.** This copies `IMG_1234.MOV` to `p01_F240_LtoR_normal_01.MOV`, using the rows of `session_log.csv`:

```
python scripts/rename_from_log.py data/raw/phone/session_log.csv data/raw/phone
```

**1. Check the file really is 240 fps.**

```
gaitbench probe data/raw/phone/p01_F240_LtoR_normal_01.MOV --no-timestamps
```

`fps_nominal` should be about 240. If it shows 30, the phone sent a slowed-down export; copy the original again with Image Capture.

**2. Run the pose model on the full-rate video.** About 2–4 minutes per 10 s clip on a MacBook CPU.

```
gaitbench pose --data data/phone --video data/raw/phone/p01_F240_LtoR_normal_01.MOV --backend mediapipe
```

**3. Make the 60 fps and 30 fps copies.**

```
gaitbench decimate --data data/phone --clip p01_F240_LtoR_normal_01 --backend mediapipe --step 4   # -> mediapipe-60fps
gaitbench decimate --data data/phone --clip p01_F240_LtoR_normal_01 --backend mediapipe --step 8   # -> mediapipe-30fps
```

This keeps every 4th or 8th frame of the keypoints. It is quick, but MediaPipe tracked those points at 240 fps. For the realistic version, make a real 60 fps video and run pose on it:

```
gaitbench cfr data/raw/phone/p01_F240_LtoR_normal_01.MOV data/raw/phone/p01_F240_LtoR_normal_01_60.mp4 --fps 60 --run
gaitbench pose --data data/phone --video data/raw/phone/p01_F240_LtoR_normal_01_60.mp4 \
    --clip-id p01_F240_LtoR_normal_01 --out-backend mediapipe-cfr60
```

Both copies keep the original's timeline, so both can be scored against the 240 fps labels. `--clip-id` ties the copy to the original's labels, and `--out-backend` keeps it from overwriting the 240 fps keypoints.

**4. Fix leg swaps.** Pose models sometimes swap left and right legs for a few frames when the legs cross. This writes `-clean` copies of every keypoints file and leaves the originals alone:

```
gaitbench clean --data data/phone
```

**5. Label the clip by hand.** Label the 240 fps original, not a copy:

```
gaitbench label --data data/phone --video data/raw/phone/p01_F240_LtoR_normal_01.MOV --rater ih --backend mediapipe
```

| Key | Action |
|---|---|
| `d` / `a` | next / previous frame (`D` / `A`: 10 frames) |
| `1` / `2` | **left** heel strike / left toe off |
| `3` / `4` | **right** heel strike / right toe off |
| `x` | delete the nearest event |
| `u` / `r` | undo / redo |
| `w` / `s` | jump to next / previous labelled event |
| `z` | 2× zoom on the feet (needs `--backend`) |
| `h` | key help |
| `Enter` | save |
| `q` | save and quit |

- **Left and right** are the walker's own sides. When they walk left to right on screen, their right side faces the camera.
- **What counts as an event.** Heel strike is the first frame the heel touches the floor. Toe off is the first frame the toe is off the floor.
- **Blind by design.** The window never shows what any detector found.
- **Saving.** Work is autosaved every 20 edits, and reopening the same clip resumes where you stopped.

**6. Run the detectors.**

```
gaitbench detect --data data/phone --detectors ours,myogait:zeni,myogait:oconnor,myogait:consensus
```

`--detectors all` runs every method, including the 11 gaitkit ones, which are slower; `gaitbench list-detectors` lists them. A detector that crashes on a clip is reported and counted as having missed that clip's events, never silently left out.

**7. Score against your labels.**

```
gaitbench score --data data/phone --rater ih
```

The table is printed and saved under `data/phone/results/score__ih/`:

| File | What it holds |
|---|---|
| `metrics.csv` | One row per backend × detector × event type. `recall` is the share of your events found within 100 ms; `precision` is the share of detections that are real; `bias_ms` is the mean signed error (negative = early); `mae_ms` is the mean absolute error; `p_within_1_frame` / `p_within_2_frames` are judged in the scored stream's own frames (a 60 fps copy in 60 fps frames, which is what the plan's "≤ 2 frames at 60 fps" bar means); `n_clips_missing` shows clips where the detector produced nothing |
| `metrics_by_side.csv` | The same, split into left and right foot (near and far side) |
| `params.csv` | Stride time, stance %, double support % and step time per detector, against the same values from your labels (mean difference, MAE, ICC, Bland–Altman limits) |
| `matches.csv` | Every event: matched (TP), missed (FN) or extra (FP), with its error |

**8. Look at the signals.**

```
gaitbench plot --data data/phone --rater ih --detectors ours,myogait:zeni
```

Each PNG in `data/phone/plots/` shows heel-minus-pelvis and toe-minus-pelvis for each foot, every detector's events, and your labels as vertical lines. It is the quickest way to see why a detector missed something.

**Checking two raters agree.** The protocol requires two raters to agree before their labels are used as truth. With a second rater's labels saved as `--rater jk`:

```
gaitbench compare-raters --data data/phone --a ih --b jk
```

It reports, per clip and event type, how many events both found and how far apart they are in 240 fps frames. The protocol's bar is heel strikes within one frame.

## Running a whole batch from a config

For repeatable runs, put the steps in a YAML file and run `gaitbench run my_run.yaml`. Each run writes `results/runs/<run_id>/` with:
- the config
- the exact package versions (`env.json`)
- the metrics and plots

`configs/smoke_synthetic.yaml` and `configs/l0_gpjatk_sample.yaml` are worked examples. The GPJATK one needs `python scripts/fetch_sample_data.py` first.

## Troubleshooting

| Problem | Fix |
|---|---|
| Label window does not open, or "OpenCV has no GUI" | Run the command `gaitbench env` prints (see Install) |
| `probe` says 30 fps for a slow-motion clip | The phone exported a slowed-down copy; transfer the original with Image Capture |
| OpenCV can't read an iPhone `.MOV` (HEVC) | Make an H.264 copy that keeps every frame: `gaitbench cfr in.MOV in_h264.mp4 --fps 240 --run` |
| Many false events at the same few moments | Leg swaps; score the `-clean` backend, and look at the plot |
| A detector shows `n_clips_missing` > 0 | It crashed on those clips; run `detect -v` to see why |
| `gaitbench pose` fails on Linux with `libEGL.so.1` | `sudo apt install libegl1 libgles2` (not needed on macOS) |
| RealSense camera not listed on macOS | Try `sudo gaitbench realsense-modes`; librealsense sometimes needs root for USB on macOS |

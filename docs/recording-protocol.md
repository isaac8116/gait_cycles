# Phone Recording Protocol

16 Oct 2026 · for the phone-only route in [the evaluation plan](open-source-evaluation-plan.md#phone-only-route-while-the-d455-is-away)

What to film with one iPhone so the harness can score gait event detectors. The clips feed experiments E2 (detectors), E3 (pose models), E4 (stress conditions) and E5 (moving camera).

## Before the session: a one-clip pilot (10 minutes)

Film one 240 fps pass, copy it to the Mac and check it before filming everyone.

1. **Copy the original file to the Mac.** Use **Image Capture** with a USB cable, or AirDrop with *Options → All Photos Data* switched on. A plain share can send a slowed-down 30 fps export instead of the original.
2. **Check the frame rate.** In the Mac terminal run:
   ```
   ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate,nb_frames,width,height -of compact data/raw/phone/IMG_1234.MOV
   ```
   `avg_frame_rate` should be about `240/1`. If it shows 30, the file is a slowed-down export; copy it again with Image Capture.
3. **Look for flicker.** Play the clip frame by frame. If the brightness pulses or dark bands roll through the picture, the lights flicker at mains frequency. Film in daylight instead.
4. **Check the feet.** Both feet should be sharp and fully in frame for at least three strides.

## Setting up the walkway

```
   start line                 camera's field of view (~4.5 m)                 stop line
       |<---- 3 m ---->|<----------------------------------------->|<---- 3 m ---->|
  ─────┼───────────────┼────────────[ 2.00 m scale marks ]───────────┼───────────────┼───── walking line
                                              ▲
                                              │ 3.5 m, at right angles to the line
                                              │
                                           [iPhone]  on tripod, ~1 m high, level, landscape
```

- **Walking line:** a straight line of floor tape at least 10 m long.
- **Start and stop lines:** 3 m outside each edge of the frame, so walkers are at a steady pace while in view.
- **Camera:** 3.5 m from the middle of the walking line, at right angles to it. Place it at hip height (about 1 m), in landscape. Turn on *Settings → Camera → Grid* and use the level so the phone is not tilted.
- **Scale marks:** two short tape marks on the walking line exactly **2.00 m** apart, both in view. They give a pixel-to-metre check.
- **Background:** a plain wall if possible, with **nobody else in frame**. Pose models can latch onto the wrong person.
- **Light:** bright, even light from the camera side. 240 fps needs a lot of light. Avoid a bright window behind the walker.

## Camera settings

| Setting | Value |
|---|---|
| Lens | **1×** (main camera). Not 0.5×: the ultra-wide bends straight lines |
| Normal video | *Settings → Camera → Record Video*: **1080p at 60 fps** |
| Slow motion | *Settings → Camera → Record Slo-mo*: **1080p at 240 fps** |
| Focus and exposure | Before each take, long-press on the walking line until **AE/AF LOCK** shows |
| Zoom | None |
| Stabilisation | Off if your app allows it (e.g. Blackmagic Camera). The built-in Camera app always stabilises; that is acceptable on the tripod. Note in the log which app you used for moving-camera takes |

## Walkers

- Fitted trousers, leggings or shorts, and everyday trainers with visible heels.
- Colours that contrast with the floor. No long coats or skirts covering the knees, except in the "baggy" takes below.
- Walk at your own normal pace unless the take says otherwise. Look ahead, not at the camera.
- Before filming, measure and log each person's **height**, **foot length** (back of the shoe heel to the shoe tip) and **thigh length** (the bony bump at the side of the hip to the knee joint line). If someone has an iPhone with LiDAR, a body scan is a bonus.
- Ask everyone's permission. The clips show identifiable people, so they stay off public sites and out of git.

## Shot list per person

A *pass* is one walk from start line to stop line. **LtoR** means the walker moves from the left to the right of the image.

| Code | Camera | Mode | Speed | Clothing | Takes | Used for |
|---|---|---|---|---|---|---|
| **F240** | Tripod | Slo-mo 240 fps | Normal | Fitted | 2 × LtoR, 2 × RtoL | E2, E3: the core set. 60 and 30 fps copies are made from these |
| **F240-S / F240-F** | Tripod | Slo-mo 240 fps | Slow / fast | Fitted | 1 slow LtoR, 1 fast RtoL | E4: speed |
| **F60** | Tripod | Video 60 fps | Normal | Fitted | 1 × LtoR, 1 × RtoL | E4: real 60 fps motion blur |
| **F240-B** | Tripod | Slo-mo 240 fps | Normal | Baggy trousers | 1 × LtoR, 1 × RtoL | E4: clothing (optional) |
| **M240** | Rolling base | Slo-mo 240 fps | Normal | Fitted | 2 × LtoR, 2 × RtoL | E5: moving camera |

That is 12–14 clips of about 10 seconds per person, roughly 20 minutes including changes.

**Moving-camera takes (M240).**
- Put the phone on a trolley, office chair or cart, at the same height and 3.5 m distance as the tripod.
- One person pushes it along a second line taped parallel to the walking line, keeping the walker in the middle of the frame at a steady speed. Practise once first.
- If a second phone is free, leave it on the tripod filming F240 at the same time. Clap once in view of both phones at the start, for sync.

**If time is short, the minimum useful set is:** 2 people × (4 F240 + 4 M240) = 16 clips.

## Logging every take

Keep the phone's own file names (`IMG_1234.MOV`) and write one row per take in a session log. Copy [`templates/session_log.csv`](templates/session_log.csv) and fill it in during the session.

| Column | Example |
|---|---|
| `file` | `IMG_1234.MOV` |
| `participant` | `p01` |
| `code` | `F240` |
| `direction` | `LtoR` |
| `speed` | `normal` |
| `clothing` | `fitted` |
| `take` | `1` |
| `app` | `Camera` |
| `notes` | `stumbled at the end` |

Participant measurements go in [`templates/participants.csv`](templates/participants.csv), one row per person.

Afterwards, copy all originals into `data/raw/phone/`. That folder is git-ignored, so videos never go to GitHub. A harness command will then rename each clip from the log, e.g. `p01_F240_LtoR_normal_01`, which becomes the clip ID in every later file.

## Labelling priority afterwards

Each 240 fps clip has about 12 events (3 strides × 2 feet × heel strike and toe off). Labelling takes roughly 3–5 minutes per clip. Do them in this order:

1. **F240 normal** for every person. Needed first, for E2.
2. **M240.** Needed for E5.
3. **Everything else.**

Have a second person label about 6 of the F240 clips independently. The two sets must agree within one 240 fps frame on heel strikes before they are trusted as ground truth (plan section 5).

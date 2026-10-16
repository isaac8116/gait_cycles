# Open-Source Gait Cycle Tools: Evaluation Plan

16 Oct 2026 · follows *Gait Cycles Deep Dive: Methods, Open Source and Next Steps* (2 Oct 2026)

## Summary

The deep dive chose myogait as the main cycle tool, with a Zeni-style method as a cross-check. This plan says how to test that choice, and the alternatives, before we build on any of them. The approach: run every candidate on the same clips through one test harness, score it against ground truth and against the acceptance criteria already set in the deep dive, and pick a tool at the Week 5 gate.

- **First, refresh the facts.** myogait has moved on since 2 Oct. It is now at v0.9.1 (7 Oct). It now bundles 15 event detectors through a companion library, gaitkit. Between v0.8.3 and v0.8.9 it fixed several bugs specific to panning cameras. Pinning v0.5.31, as the deep dive suggested, is out of date.
- **Keep pose and events separate.** Pose estimation is run once per clip and backend, and the keypoints are cached. Every event detector then runs on identical keypoints, so differences between detectors are not mixed up with differences between pose models.
- **Test on a ladder of data, from free to expensive:** sample keypoints shipped with Stenum's code, then public datasets with motion capture (BioCV, GPJATK), then our own hand-labelled phone clips, then the trolley moving-camera test, then the rig.
- **The moving-camera test is the one no published validation covers.** myogait claims support for following-camera video but was validated only on fixed cameras. Our trolley test is the real evidence.
- **Freeze a golden set.** myogait released 12 versions in 7 weeks. Every upgrade must rerun a fixed set of clips and show events moved by no more than one frame.

Total effort is about 2–2.5 weeks for one or two people. The schedule is in [Schedule](#schedule).

---

## 1. What changed since the deep dive

| Item | Deep dive (2 Oct) | Now (checked 16 Oct) | Effect on the plan |
|---|---|---|---|
| myogait version | Pin v0.5.31+ | **v0.9.1** (7 Oct 2026); 12 releases since 0.8.0 on 21 Aug | Pin `myogait==0.9.1`; add an upgrade regression test (E6) |
| Event detectors in myogait | "Several detectors with voting" | 15 methods: `zeni`, `crossing`, `velocity`, `oconnor` and 11 gaitkit methods (`gk_bike`, `gk_zeni`, `gk_oconnor`, `gk_hreljac`, `gk_mickelborough`, `gk_ghoussayni`, `gk_vancanneyt`, `gk_dgei`, `gk_intellevent`, `gk_deepevent`, `gk_ensemble`), plus `event_consensus` | The detector shoot-out (E2) can run almost entirely inside one package |
| Panning / following camera | Direction fix in v0.5.31 | v0.8.3: step length measured within one frame (pan-immune). v0.8.4: metrics made pelvis-relative. v0.8.6: walking direction by a vote of four pan-invariant cues. v0.8.9: scale taken over the whole clip (lengths were ~1.5× too long on panning video) | Good signs, but still not validated on moving cameras. E5 checks them |
| Validation | 3 healthy adults, peak timing ≤ 2% of cycle | v0.8.9 + Sapiens 2 on **BioCV**: 9 adults, 194 video–marker pairs, fixed 200 fps cameras analysed at 60 fps. Event timing −6 ± 26 ms (initial contact), −3 ± 29 ms (toe off), left lateral view. Stride length ICC 0.71 (bias −0.04 m) | Event errors are near our 33 ms bar, so we must measure on our own footage. Stride length is weak; keep our encoder-based method |
| How myogait's events were validated | — | Marker-side events were detected **by myogait's own algorithm on the marker data**, not by force plates. Only 62 of 83 left-lateral trials passed synchronisation | This tests the pose stage, not the event rule. We add force-plate or hand-labelled truth (E1, E2) |
| Pixel-to-metre | Pixel-based | Scale from anthropometry (`femur_mm`, `foot_mm`, `height_m`); no checkerboard or encoder input | Our rig conversion is still a gap to build; compare it against myogait's method |

Install check (done 16 Oct in a clean Python 3.13 venv): `pip install myogait==0.9.1` installs cleanly and pulls in gaitkit 1.4.9. `list_event_methods()` returns all 15 methods. Two small mismatches turned up, showing the docs and the code don't always agree:
- `available_models()` is documented at the top level but lives in `myogait.models`.
- `gaitkit.__version__` reports 1.4.7 while pip reports 1.4.9.

Record versions with `pip freeze`, not `__version__`.

---

## 2. Questions this plan answers

1. **Events:** which detector gives the most accurate heel strikes and toe offs on side-view keypoints from our cameras? Is it Zeni, gaitkit's Bayesian `gk_bike`, a consensus, or Stenum's ankle–pelvis rule?
2. **Pose:** which backend finds heel and toe reliably at our distance, at an acceptable speed on our hardware, and under a licence we can use?
3. **Moving camera:** do the event timings and walking direction stay correct when the camera tracks the walker?
4. **Pipeline:** does myogait's cycle segmentation, normalisation and quality gating do what we need, or do we keep only its pose and angle stages?
5. **Maintenance:** can we pin a version and detect it when an upgrade changes our numbers?

---

## 3. Step 0: a time-boxed discovery sweep (1 day)

The deep dive's tool list is a good start, but it was written before myogait's 0.8–0.9 releases. Spend one day checking that nothing better has appeared, then stop.

**Where to look**
- GitHub topics `gait-analysis`, `gait-event-detection`, `markerless-motion-capture`, sorted by recently updated
- PyPI search: "gait", "gait events", "pose gait"
- Papers with Code and JOSS for "gait event detection video"
- Recent reviews, e.g. the 2026 PRISMA scoping review of video-based markerless capture (arXiv 2609.18667); check the code links in its tables
- Code and data availability statements of the papers already cited in the deep dive

**Screening filter.** A tool must pass all of these to be tested:

| Filter | Pass if |
|---|---|
| Code available | Public repository with a licence file |
| Input | Works from a single side-view video, or from 2D keypoints we can supply |
| Output | Produces gait events (heel strike, toe off) or a cycle segmentation, not only joint angles |
| Language | Python, or simple enough to port in under a day |
| Activity | Commit in the last 12 months, **or** a peer-reviewed validation (stable research code is fine) |
| Validation | Some published comparison against motion capture, force plates or a walkway |

**Already screened out**, so we don't revisit them:

| Tool | Reason |
|---|---|
| GaitTransformer (Cotton) | Needs 3D lifting. Its README says it works poorly away from a frontal view, and no training data is available |
| gaitmap, mobgap, GaitPy | IMU-only |
| IntellEvent (standalone) | Built for Vicon markers; already inside gaitkit as `gk_intellevent` |
| OpenCap as a rig tool | Needs 2+ cameras; keep it only as ground truth for joint angles |
| VisionMD-Gait (Sci Rep 2025) | Code availability not confirmed. Recheck during the sweep; promote it if code is public |

---

## 4. Shortlist and what each is tested for

| Candidate | Role in the test | Version to pin | Licence | Notes |
|---|---|---|---|---|
| **myogait** | Main pipeline: pose → angles → events → cycles → parameters | 0.9.1 | MIT | Use `run_pipeline` with explicit settings, not `run_auto`, so every clip gets the same recipe |
| **gaitkit** (via myogait) | 11 event detectors incl. Bayesian `gk_bike` and `gk_ensemble`; event-matching metrics | 1.4.9 | MIT (IntellEvent models CC BY 4.0; DeepEvent see upstream) | Benchmarked on marker data only; its detectors adapt their windows below 100 fps since 1.4.6 |
| **Our Zeni** | Independent reference: heel/toe minus mid-hip, `scipy.signal.find_peaks` | ours | ours | ~40 lines; guards against bugs shared by myogait's two Zeni versions |
| **Stenum rule (ported)** | Ankle minus mid-hip peaks, validated on OpenPose at 25 fps | port of MATLAB | **GPL-3.0** | Re-implement from the paper; don't copy code into our module unless we accept GPL |
| **Pose2Sim `trc_gaitevents`** | Zeni (`forward_coordinates`) plus a **height-threshold** method (`height_coordinates`) | 0.10.49 | BSD-3 | Reads TRC; feed it with myogait's `export_trc`. Height-based events should also be immune to horizontal panning, so test them in E5 |
| **OpenCap `gait_analysis.py`** | Ground-truth joint angles and events on a subset (2 phones) | latest | Apache 2.0 | Its `segment_walking` is a 3D Zeni variant (calcaneus vs pelvis) |
| **Sports2D** | Fallback angle source if myogait's angles disappoint | 0.8.34 | BSD-3 | Angles only; events come from our harness |

**Pose backends to compare**, all through myogait's single API:

| Backend | Foot points | Hardware | Licence to check | Why include it |
|---|---|---|---|---|
| MediaPipe | Heel, foot index (33 points) | CPU | Apache 2.0 | Fast baseline; myogait's quick-start default |
| RTMW (rtmlib) | 6 foot points (133 whole-body) | CPU/GPU, ONNX | Apache 2.0 | Heel and big toe at a moderate speed |
| YOLOv8-Pose | None (COCO-17, ankles only) | CPU/GPU | **AGPL-3.0** (Ultralytics) | Tests whether ankle-only events are good enough; AGPL may rule it out for HTX |
| ViTPose | None (COCO-17) | GPU | Apache 2.0 | Strong body accuracy |
| Sapiens 2 | 308 Goliath points incl. feet | GPU | Check Meta's model licence | Backend used in myogait's own validation |

---

## 5. Test harness design

The harness is a set of stages that pass **files**, not Python objects. Then every tool can live in its own virtual environment: Pose2Sim pulls in OpenSim, OpenVINO and PySide6, and Sapiens 2 needs torch ≥ 2.7, so installing them together would clash.

```
video ──► [pose stage] ──► keypoints cache ──► [event stage] ──► events.csv ──► [scoring] ──► results/
            one run per        (per clip ×        one run per                     metrics, plots,
            clip × backend      backend)          clip × backend × detector       scorecard
```

**Common formats** (`docs/formats.md` once agreed):

| File | Contents |
|---|---|
| `keypoints/<clip>__<backend>.npz` | Array of frames × keypoints × (x, y, confidence), with metadata: fps, frame size, backend, tool versions |
| `events/<clip>__<backend>__<detector>.csv` | `clip_id, tool, tool_version, backend, detector, side (L/R), event (HS/TO), frame, time_s, confidence` |
| `labels/<clip>__<rater>.csv` | Same columns as events, `tool = human` |
| `cycles/<clip>__<config>.npz` | Cycles × 6 channels × 101 samples, with a per-cycle table: start/end frame, toe-off %, QC flags |
| `runs/<run_id>.yaml` | Config of the run: versions (from `pip freeze`), settings, git commit |

**Adapters**, one small script per tool, each converting that tool's output into the common format:
- `myogait`
- `pose2sim` (TRC in, events out)
- `stenum_port`
- `zeni_ref`
- `opencap` (for the ground-truth subset)

**Scoring.** Reuse gaitkit's MIT-licensed `evaluation.match_events` and `compute_event_metrics`. They do greedy matching within a tolerance and give precision, recall, F1 and MAE. On top of them we add signed bias ± SD, the share of events within 1 and 2 frames, ICC and Bland–Altman. The match tolerance is 100 ms. Errors are reported separately for heel strike and toe off, and for near-side and far-side limbs.

**Labelling tool.** Kinovea, or a ~100-line OpenCV frame stepper that writes the labels CSV. Two raters label the same clips. Check that they agree with each other (heel strike within 1 frame) before using the labels as truth.

---

## 6. Test data ladder

Each level is cheaper and faster than the next. Don't climb to the next level until the current one runs cleanly.

| Level | Data | Ground truth | Available | Use it for |
|---|---|---|---|---|
| 0 | Stenum repo, `Multiple-Perspectives` branch: OpenPose keypoint JSONs (135 frames) for GPJATK subject p01, left and right sagittal views plus frontal | None (visual check) | Now, in the public repo | Day-1 smoke test of event adapters without any video or GPU |
| 1a | **GPJATK** (Kwolek et al. 2019): 32 people, 166 sequences, 4 calibrated synchronised cameras + 10-camera Vicon; 6 people also recorded in different clothes | Vicon events | Zenodo; scientific use under a release agreement | Event and angle accuracy; the clothing subset also matters for identification |
| 1b | **BioCV** (Evans et al. 2024, doi:10.15125/BATH-01258): 15 adults, walking and other tasks, multi-camera 200 fps, Qualisys + **force plates** | Force-plate contact on plate steps; marker events elsewhere | Bath archive; all rights reserved, no redistribution | Reproduce myogait's published numbers; frame-rate degradation (200 → 60 → 30 fps) |
| 2 | Our side-on phone clips of teammates, 60 fps, plus a 240 fps slow-motion reference for a subset | Two-rater hand labels | Week of recording | Accuracy on our people, clothes and distances |
| 3 | **Trolley test**: fixed camera + camera pushed on a trolley alongside the walker, synced by clap or LED | Hand labels on the fixed camera; footprints taped for length | ~1 day to set up | Moving-camera invariance |
| 4 | Rig footage with encoder log | As level 3 | Weeks 8–9 | Final check (deep dive Phase 4) |

**Request access to GPJATK and BioCV now.** Do it together with the CASIA-B and OUMVLP-Pose requests, since agreements can take days.

---

## 7. Experiments

Run these in order. The full product of backends × detectors × conditions is too large, so each experiment fixes all but one factor.

| ID | Question | Hold fixed | Vary | Data | Pass if |
|---|---|---|---|---|---|
| **E0** Install and smoke | Does each candidate install and run here? | — | Each tool, in its own venv | L0, one L2 clip | Runs end to end; events look right overlaid on video; install problems logged |
| **E1** Reproduce | Do we get myogait's published numbers? | myogait 0.8.9 + their `validation/` code, then 0.9.1 | Version | L1b | Within the published SDs (e.g. initial contact −6 ± 26 ms). If not, stop and find out why before going on |
| **E2** Detector shoot-out | Which event rule is most accurate? | One strong backend (RTMW, or Sapiens 2 if a GPU is available) | All 15 myogait methods, `event_consensus`, our Zeni, Stenum port, Pose2Sim forward and height methods | L1a/L1b, L2 | Heel-strike MAE ≤ 2 frames at 60 fps; toe-off MAE ≤ 3 frames; ≥ 95% of events found. Keep the top 2–3 |
| **E3** Backend comparison | Which pose model is good enough on feet? | Top detector from E2 | MediaPipe, RTMW, YOLO, ViTPose, Sapiens 2 | L1, L2 | Foot-point dropout < 10% of frames; event bar as E2; joint-angle RMSE ≤ 6° hip/knee, ≤ 8° ankle; runtime and licence recorded |
| **E4** Stress | Where does it break? | Best backend + detector | fps 60/30; walking left→right vs right→left; near vs far limb; mid-stride start; resolution downscaled; clothing change (GPJATK) | L1, L2 | Degradation stays within the bar at 60 fps for near-side limb; record limits for the rest |
| **E5** Moving camera | Is a tracking camera safe? | Top 2–3 detectors | Fixed vs trolley camera, same walk | L3 | Mean event difference between cameras within 1 frame; velocity detectors expected to fail. Also checks myogait's walking direction and stride length against taped footprints |
| **E6** Golden set | Will an upgrade change our numbers? | The chosen configuration | Tool version | ~10 clips from L1–L3 with saved outputs | Events move ≤ 1 frame; cycle counts unchanged. Rerun on every upgrade |

Notes for running them:
- **Settings.** Use `run_pipeline` with the settings written into the run file (filter, detector, cycle gates). `run_auto` picks a recipe per recording, which would make clips non-comparable.
- **myogait-specific checks** taken from its changelog:
  - hip sign on single-direction clips (fixed in 0.9.1)
  - walking direction on panning clips (0.8.6)
  - metre scale on clips where the walker enters late (0.8.9)
  - number of cycles dropped by the direction filter on there-and-back walks
- **Learned detectors** (`gk_intellevent`, `gk_deepevent`) were trained on marker data at ≥ 100 Hz. Include them, since they cost nothing to run, but expect them to do badly on 60 fps keypoints.
- **Truth.** Report results against force-plate truth separately from marker-event truth. Marker events depend on the detector rule we are testing.

---

## 8. Scorecard and decision rule

**Hard gates.** A candidate is out if it fails any of these:
1. Licence fits how HTX will use the system (watch AGPL and non-commercial model weights)
2. Runs on the hardware we will have
3. Meets the heel-strike bar in E2 on our own footage (L2)

**Weighted score** for the candidates that pass the gates (weights are a proposal):

| Criterion | Weight | Evidence |
|---|---|---|
| Event accuracy on our footage | 30% | E2, E3 on L2 |
| Moving-camera robustness | 20% | E5 |
| Foot keypoints and joint angles | 15% | E3 |
| Robustness: failure rate, zero-cycle clips, stress results | 10% | E4, all runs |
| Runtime on our hardware | 10% | E3 timings |
| Maintenance and docs: release cadence, tests, response to issues | 10% | Discovery sweep, E0 |
| Install effort | 5% | E0 log |

**Decision rule at the Week 5 gate**
- **myogait's Zeni and our Zeni agree within 1 frame and both pass.** Adopt myogait for the pipeline and keep our Zeni as a permanent regression check.
- **`gk_bike` or `event_consensus` beats Zeni by more than 1 frame MAE on our footage.** Use it, and keep Zeni as the cross-check.
- **myogait fails E5 (moving camera) but its pose and angles are good.** Keep myogait for pose and angles only. Run our own Zeni events and segmentation on its keypoints.
- **All detectors fail on our footage.** The problem is upstream (pose, frame rate, distance). Go back to E3/E4 before blaming the detector.

Record the decision, versions and settings in the interface document. Save the run files behind it in `results/`.

---

## Schedule

| Step | Work | Experiments | Duration | Deliverable |
|---|---|---|---|---|
| 1 | Discovery sweep; set up one venv per tool; send dataset requests | Step 0, E0 | 2 days | Shortlist confirmed; install log; pinned `requirements-*.txt` |
| 2 | Build the harness: formats, adapters, scoring, labelling tool. Record and label 3–5 phone clips | — | 3 days | Harness runs end to end on L0 and one L2 clip |
| 3 | Run the shoot-outs on public data and phone clips | E1 (if BioCV access arrives), E2, E3 | 3 days | Event and backend results tables |
| 4 | Stress and trolley tests | E4, E5 | 2 days | Stress limits; moving-camera verdict |
| 5 | Score, decide, freeze the golden set | E6 | 1 day | Scorecard; decision record; golden set |

This covers Phases 1–2 of the deep dive (Weeks 3–5) and ends at the tool-selection gate. If the gate is closer than 2.5 weeks, the minimum path is:
1. E0
2. E2 on phone clips with hand labels
3. E5 trolley test
4. A decision based on those three

E1, E3 and E4 then move to Phase 3.

---

## Risks and watch-outs

| Risk | Mitigation |
|---|---|
| myogait changes fast (12 releases in 7 weeks), and a fix can change our numbers | Pin the version; golden-set regression (E6) before any upgrade; read the changelog |
| Published event errors (±26–29 ms SD) are close to our 33 ms bar | Decide on our own footage (L2), not on published numbers |
| myogait's event validation used its own detector on marker data | Use force plates (BioCV) and hand labels as truth |
| "Supports following cameras" is a claim, not a validation | E5 trolley test before the rig exists |
| Dependency clashes between tools | One venv per tool; stages talk through files |
| Licences: Stenum GPL-3.0, Ultralytics AGPL-3.0, BioCV all rights reserved, GPJATK scientific use only | Re-implement GPL algorithms; check weight licences in E3; never commit dataset files |
| Far-side limb and toe off are the weakest outputs | Report them separately; use the near-side limb for curves, as the deep dive proposes |
| Labelling takes longer than planned | Label heel strikes first; toe offs only on the clips needed for E2 |

---

## Proposed repo layout

```
gait_cycles/
├── docs/                  plan, formats, decision record
├── envs/                  requirements-myogait.txt, requirements-pose2sim.txt, ...
├── harness/
│   ├── pose/              run a backend on a clip → keypoints cache
│   ├── adapters/          myogait, pose2sim, stenum_port, zeni_ref, opencap
│   ├── scoring/           matching, metrics, Bland–Altman, plots
│   └── labelling/         frame-stepping labeller
├── configs/               one YAML per run
├── data/                  git-ignored: raw/, keypoints/, labels/
├── golden/                golden-set clip list + expected outputs (small files only)
└── results/               per-run CSVs and the scorecard
```

---

## Open questions

- [ ] **GPU.** Do we have one for Sapiens 2 / ViTPose, or is the decision limited to CPU backends (MediaPipe, RTMW)?
- [ ] **Licence constraints from HTX.** Are AGPL code or non-commercial model weights acceptable in a prototype? In a delivered system?
- [ ] **Dataset access.** Who signs the GPJATK and BioCV agreements, and where may the data be stored?
- [ ] **Labelling.** Who are the two raters, and how many clips can they label before the gate?
- [ ] **Gate date.** Is the Week 5 gate still the deadline, or is the minimum path needed?

---

## Sources

Checked 16 Oct 2026.

- myogait: [PyPI](https://pypi.org/project/myogait/) · [GitHub](https://github.com/IDMDataHub/myogait) · `CHANGELOG.md` and `validation/README.md` in the repo
- gaitkit: [PyPI](https://pypi.org/project/gaitkit/) · [GitHub](https://github.com/IDMDataHub/gaitkit) (README, `REPRODUCIBILITY.md`, `python/src/gaitkit/evaluation/`)
- Stenum et al.: [GaitAnalysis-PoseEstimation](https://github.com/janstenum/GaitAnalysis-PoseEstimation) (`master`: 2021 MATLAB code; `Multiple-Perspectives`: 2024 code and GPJATK example keypoints) · [PLOS Comp Bio 2021](https://doi.org/10.1371/journal.pcbi.1008935) · [PLOS Digital Health 2024](https://doi.org/10.1371/journal.pdig.0000467)
- Pose2Sim: [GitHub](https://github.com/perfanalytics/pose2sim) (`Pose2Sim/Utilities/trc_gaitevents.py`) · Sports2D: [GitHub](https://github.com/davidpagnon/Sports2D)
- OpenCap processing: [GitHub](https://github.com/stanfordnmbl/opencap-processing) (`ActivityAnalyses/gait_analysis.py`)
- GaitTransformer: [GitHub](https://github.com/peabody124/GaitTransformer)
- BioCV: Evans et al. 2024, *Scientific Data*, [doi:10.1038/s41597-024-04077-3](https://doi.org/10.1038/s41597-024-04077-3); data [doi:10.15125/BATH-01258](https://doi.org/10.15125/BATH-01258)
- GPJATK: Kwolek et al. 2019, *Multimedia Tools and Applications*, [doi:10.1007/s11042-019-07945-y](https://doi.org/10.1007/s11042-019-07945-y); data on [Zenodo](https://zenodo.org/records/15130800)
- VisionMD-Gait: [Scientific Reports 2025](https://www.nature.com/articles/s41598-025-34912-5)
- Scoping review of video-based markerless capture: [arXiv 2609.18667](https://arxiv.org/abs/2609.18667)

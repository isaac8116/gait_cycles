# gait_cycles

Gait cycle module for the rail-camera rig: finding, testing and adopting open-source gait event and cycle tools.

- [Open-source evaluation plan](docs/open-source-evaluation-plan.md): how candidate tools are tested and chosen before the Week 5 tool-selection gate.
- [Phone recording protocol](docs/recording-protocol.md): walkway setup, camera settings and the shot list for test clips.
- [Using the test harness](docs/harness.md): install on a Mac, then pose → clean → label → detect → score for iPhone clips.
- [Upstream issues](docs/upstream-issues.md): bugs found in myogait 0.9.1 and gaitkit 1.4.9, ready to file.

Quick start:

```
pip install -e ".[dev,video,myogait,pose]"
gaitbench run configs/smoke_synthetic.yaml
```

"""End-to-end tests of the gaitbench command line, run in a temporary data root."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gaitbench import cli
from gaitbench.formats import (
    Keypoints,
    empty_events,
    events_path,
    keypoints_path,
    labels_path,
    read_events,
    write_events,
)

detectors = pytest.importorskip("gaitbench.detectors")


def run(*argv: str) -> int:
    return cli.main([str(a) for a in argv])


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two clean synthetic clips (fixed and panning camera) with our detectors run on them."""
    root = tmp_path_factory.mktemp("data")
    assert run("simulate", "--data", root, "--clip-id", "fixed") == 0
    assert run("simulate", "--data", root, "--clip-id", "pan", "--pan", "follow", "--direction", "rtl") == 0
    assert run("detect", "--data", root, "--detectors", "ours") == 0
    return root


def test_simulate_writes_keypoints_and_truth(data: Path) -> None:
    kp = Keypoints.load(keypoints_path(data, "pan", "synthetic"))
    assert kp.meta["camera"]["pan_speed_mps"] == pytest.approx(1.3)
    assert kp.meta["walk_direction"] == "right_to_left"
    assert kp.time_s[-1] == pytest.approx(6.0 - 1 / 60)
    truth = read_events(labels_path(data, "pan", "truth"))
    assert set(truth["tool"]) == {"synthetic"} and len(truth) > 15


def test_detect_writes_one_file_per_detector(data: Path) -> None:
    for clip in ("fixed", "pan"):
        for det in ("zeni", "stenum"):
            events = read_events(events_path(data, clip, "synthetic", det))
            assert set(events["detector"]) == {det} and set(events["clip_id"]) == {clip}
            assert set(events["backend"]) == {"synthetic"}


def test_score_recall_is_one_for_zeni(data: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    pytest.importorskip("gaitbench.scoring")
    out = tmp_path / "res"
    assert run("score", "--data", data, "--rater", "truth", "--tolerance-ms", 50, "--out", out) == 0
    for name in ("matches.csv", "metrics.csv", "metrics_by_side.csv", "params.csv"):
        assert (out / name).is_file()
    metrics = pd.read_csv(out / "metrics.csv")
    zeni = metrics[metrics["detector"] == "zeni"]
    assert set(zeni["event"]) == {"HS", "TO"}
    assert (zeni["recall"] == 1.0).all() and (zeni["precision"] == 1.0).all()
    assert (zeni["p_within_1_frame"] == 1.0).all()
    matches = pd.read_csv(out / "matches.csv")
    assert set(matches["clip_id"]) == {"fixed", "pan"}
    params = pd.read_csv(out / "params.csv")
    stride = params[(params["detector"] == "zeni") & (params["parameter"] == "stride_time_s")]
    assert stride["mae"].iloc[0] < 1 / 60
    printed = capsys.readouterr().out
    assert "recall" in printed and "zeni" in printed


def test_empty_events_file_counts_as_missed(data: Path, tmp_path: Path) -> None:
    pytest.importorskip("gaitbench.scoring")
    root = tmp_path / "d"
    shutil.copytree(data, root)
    write_events(empty_events(), events_path(root, "fixed", "synthetic", "stenum"))
    assert run("score", "--data", root, "--out", tmp_path / "res") == 0
    metrics = pd.read_csv(tmp_path / "res" / "metrics.csv")
    stenum = metrics[metrics["detector"] == "stenum"].set_index("event")
    zeni = metrics[metrics["detector"] == "zeni"].set_index("event")
    assert (stenum["n_truth"] == zeni["n_truth"]).all()
    assert (stenum["recall"] < 0.6).all()


def test_agree(data: Path, tmp_path: Path) -> None:
    pytest.importorskip("gaitbench.scoring")
    out = tmp_path / "agree"
    assert run("agree", "--data", data, "--reference", "zeni", "--out", out) == 0
    table = pd.read_csv(out / "agreement.csv")
    assert set(table["detector"]) == {"stenum"} and set(table["ref_detector"]) == {"zeni"}
    assert (table["recall"] == 1.0).all()
    assert (out / "agreement_matches.csv").is_file()
    with pytest.warns(UserWarning, match="no events from reference"):
        assert run("agree", "--data", data, "--reference", "nobody", "--out", out) == 1


def test_detect_continues_past_a_failing_detector(data: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    root = tmp_path / "d"
    shutil.copytree(data, root)
    shutil.rmtree(root / "events")
    real = detectors.get_detector

    def flaky(name: str, **params):
        if name == "stenum":
            def broken(kp):
                raise RuntimeError("boom")
            return broken
        return real(name, **params)

    monkeypatch.setattr(detectors, "get_detector", flaky)
    assert run("detect", "--data", root, "--detectors", "stenum,zeni") == 0
    assert "boom" in capsys.readouterr().out
    assert events_path(root, "fixed", "synthetic", "zeni").is_file()
    assert not events_path(root, "fixed", "synthetic", "stenum").exists()
    assert run("detect", "--data", root, "--detectors", "stenum,zeni", "--strict") == 1


def test_detect_rejects_unknown_names_and_params(data: Path, capsys) -> None:
    assert run("detect", "--data", data, "--detectors", "zeen") == 1
    assert "zeen" in capsys.readouterr().err
    assert run("detect", "--data", data, "--detectors", "zeni", "--params", '{"zeni": {"nope": 1}}') == 1
    assert run("detect", "--data", data, "--detectors", "zeni", "--params", '{"stenum": {}}') == 1
    assert run("detect", "--data", data / "missing") == 1


def test_detect_selection_and_params(data: Path, tmp_path: Path) -> None:
    root = tmp_path / "d"
    shutil.copytree(data, root)
    shutil.rmtree(root / "events")
    assert run("detect", "--data", root, "--clip", "pan", "--detectors", "zeni",
               "--params", '{"zeni": {"cutoff_hz": 4.0}}') == 0
    assert [p.name for p in (root / "events").iterdir()] == ["pan__synthetic__zeni.csv"]
    mtime = events_path(root, "pan", "synthetic", "zeni").stat().st_mtime_ns
    assert run("detect", "--data", root, "--clip", "pan", "--detectors", "zeni", "--skip-existing") == 0
    assert events_path(root, "pan", "synthetic", "zeni").stat().st_mtime_ns == mtime


def test_expand_detectors_and_filter() -> None:
    assert cli.expand_detectors("ours") == ["zeni", "stenum"]
    assert cli.expand_detectors(["zeni", "ours"]) == ["zeni", "stenum"]
    assert cli.expand_detectors("all")[:2] == ["zeni", "stenum"]
    keep = cli.detector_filter("ours,myogait:gk_bike")
    assert keep("zeni") and keep("myogait-gk_bike") and not keep("myogait-zeni")
    assert cli.detector_filter("myogait")("myogait-zeni") and not cli.detector_filter("myogait")("zeni")
    assert cli.detector_filter("all") is None and cli.detector_filter(None) is None


def test_decimate_keeps_source_frames(tmp_path: Path) -> None:
    pytest.importorskip("gaitbench.scoring")
    root = tmp_path / "d"
    assert run("simulate", "--data", root, "--clip-id", "slowmo", "--fps", 240) == 0
    assert run("decimate", "--data", root, "--clip", "slowmo", "--backend", "synthetic", "--step", 4) == 0
    kp = Keypoints.load(keypoints_path(root, "slowmo", "synthetic-60fps"))
    assert kp.fps == pytest.approx(60.0)
    np.testing.assert_array_equal(kp.frame_idx[:3], [0, 4, 8])
    assert run("detect", "--data", root, "--backend", "synthetic-60fps", "--detectors", "zeni") == 0
    events = read_events(events_path(root, "slowmo", "synthetic-60fps", "zeni"))
    # Frames are numbered in the 240 fps source (sub-frame peaks may land between kept frames).
    assert events["frame"].max() > kp.n_frames and events["frame"].max() < 6 * 240
    assert run("score", "--data", root, "--out", tmp_path / "res") == 0
    metrics = pd.read_csv(tmp_path / "res" / "metrics.csv")
    assert (metrics["recall"] == 1.0).all()
    # Frame-based columns use the copy's own rate: 60 fps frames, not 240 fps ones.
    assert (metrics["mae_frames"] < 1.0).all()
    with pytest.raises(ValueError):
        cli.decimate_keypoints(kp, 2, offset=2)


def test_plot(data: Path, tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    out = tmp_path / "plots"
    assert run("plot", "--data", data, "--rater", "truth", "--clip", "pan", "--out", out) == 0
    assert [p.name for p in out.iterdir()] == ["pan__synthetic.png"]


def test_run_config(tmp_path: Path, capsys) -> None:
    pytest.importorskip("gaitbench.scoring")
    yaml = pytest.importorskip("yaml")
    config = {
        "run_id": "smoke",
        "data_root": str(tmp_path / "data"),
        "results_root": str(tmp_path / "runs"),
        "simulate": [
            {"clip_id": "fixed", "fps": 60, "pan": 0.0, "direction": "ltr", "duration": 6},
            {"clip_id": "pan", "fps": 60, "pan": "follow", "direction": "rtl", "duration": 6},
        ],
        "detectors": ["ours"],
        "truth_rater": "truth",
        "reference_detector": "zeni",
        "tolerance_ms": 100,
    }
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(config))
    assert run("run", path) == 0
    out = tmp_path / "runs" / "smoke"
    for name in ("config.yaml", "env.json", "metrics.csv", "matches.csv", "params.csv", "agreement.csv",
                 "detect.csv", "failures.csv"):
        assert (out / name).is_file(), name
    recorded = yaml.safe_load((out / "config.yaml").read_text())
    assert recorded["detectors_expanded"] == ["zeni", "stenum"]
    env = json.loads((out / "env.json").read_text())
    assert env["versions"]["numpy"]
    metrics = pd.read_csv(out / "metrics.csv")
    assert (metrics.loc[metrics["detector"] == "zeni", "recall"] == 1.0).all()
    assert "Results in" in capsys.readouterr().out
    assert run("run", path) == 1  # refuses to overwrite a run record
    assert run("run", path, "--overwrite") == 0
    assert run("run", path, "--run-id", "second") == 0
    assert (tmp_path / "runs" / "second" / "metrics.csv").is_file()


def test_run_config_bundled_examples_parse() -> None:
    yaml = pytest.importorskip("yaml")
    configs = Path(__file__).resolve().parents[1] / "configs"
    for path in sorted(configs.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        assert {"run_id", "data_root", "detectors"} <= set(cfg), path.name
        assert not set(cfg) - cli._CONFIG_KEYS, path.name
        assert cfg.get("truth_rater") or cfg.get("reference_detector"), path.name


def test_import_openpose(tmp_path: Path) -> None:
    jdir = tmp_path / "json"
    jdir.mkdir()
    for i in range(3):
        flat = []
        for j in range(25):
            flat += [100.0 + i + j, 200.0 + j, 0.9]
        (jdir / f"v_{i:012d}_keypoints.json").write_text(json.dumps({"people": [{"pose_keypoints_2d": flat}]}))
    root = tmp_path / "d"
    assert run("import-openpose", "--data", root, "--json-dir", jdir, "--clip-id", "op", "--fps", 25,
               "--width", 640, "--height", 480) == 0
    kp = Keypoints.load(keypoints_path(root, "op", "openpose"))
    assert kp.n_frames == 3 and kp.meta["fps"] == 25
    assert run("import-openpose", "--data", root, "--json-dir", jdir, "--clip-id", "op") == 1


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg")
def test_probe_and_cfr(tmp_path: Path, capsys) -> None:
    video = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30", "-t", "0.5",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    csv = tmp_path / "ts.csv"
    assert run("probe", video, "--timestamps-csv", csv) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["n_frames"] == 15 and info["is_vfr"] is False
    assert len(pd.read_csv(csv)) == 15
    assert run("cfr", video, tmp_path / "out.mp4", "--fps", 30) == 0
    assert "fps=30" in capsys.readouterr().out
    assert run("cfr", video, tmp_path / "out.mp4", "--fps", 15, "--run") == 0
    assert (tmp_path / "out.mp4").is_file()


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg")
def test_pose_passes_real_frame_times(tmp_path: Path, monkeypatch) -> None:
    from gaitbench import myogait_bridge
    from gaitbench.synthetic import simulate_walk

    video = tmp_path / "walk.mp4"
    # Variable frame rate: every third interval doubled.
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30", "-frames:v", "12",
                    "-vf", "setpts='(N+floor(N/2))/30/TB'", "-fps_mode", "passthrough", "-pix_fmt", "yuv420p",
                    str(video)], check=True)
    seen = {}

    def fake_extract(video_path, model="mediapipe", clip_id=None, *, timestamps=None, **kwargs):
        seen.update(model=model, clip_id=clip_id, timestamps=timestamps, kwargs=kwargs)
        kp = simulate_walk(fps=30, duration_s=12 / 30).keypoints
        times = kp.time_s if timestamps is None else timestamps[:12]
        return Keypoints(clip_id=clip_id, names=kp.names, xy=kp.xy, conf=kp.conf, time_s=times,
                         frame_idx=np.arange(12), meta={**kp.meta, "backend": model})

    monkeypatch.setattr(myogait_bridge, "extract_keypoints", fake_extract)
    assert run("pose", "--data", tmp_path / "d", "--video", video, "--backend", "rtmw", "--max-frames", 12) == 0
    assert seen["model"] == "rtmw" and seen["clip_id"] == "walk" and seen["kwargs"] == {"max_frames": 12}
    n = np.arange(12)
    np.testing.assert_allclose(seen["timestamps"][:12], (n + n // 2) / 30, atol=1e-3)
    assert len(seen["timestamps"]) > 12  # padded for decoders that see an extra frame
    kp = Keypoints.load(keypoints_path(tmp_path / "d", "walk", "rtmw"))
    assert kp.meta["video_probe"]["codec"] == "h264"
    assert run("pose", "--data", tmp_path / "d", "--video", video, "--nominal-time") == 0
    assert seen["timestamps"] is None


def test_list_detectors_and_env(capsys) -> None:
    assert run("list-detectors") == 0
    assert capsys.readouterr().out.split()[:2] == ["zeni", "stenum"]
    assert run("env") == 0
    assert "versions" in json.loads(capsys.readouterr().out)


def test_label_calls_the_labeller(tmp_path: Path, monkeypatch) -> None:
    calls = []
    fake = types.ModuleType("gaitbench.labelling.app")
    fake.run_labeller = lambda *a, **k: calls.append((a, k))
    monkeypatch.setitem(sys.modules, "gaitbench.labelling.app", fake)
    video = tmp_path / "walk01.mov"
    assert run("label", "--video", video, "--rater", "ab", "--data", tmp_path, "--backend", "mediapipe") == 0
    (args, kwargs), = calls
    assert args == (video, labels_path(tmp_path, "walk01", "ab"), "ab")
    assert kwargs == {"clip_id": "walk01", "keypoints_path": keypoints_path(tmp_path, "walk01", "mediapipe")}
    assert run("label", "--video", video, "--rater", "ab") == 1  # nowhere to write


def test_realsense_commands(tmp_path: Path, monkeypatch, capsys) -> None:
    recorded = {}
    fake = types.ModuleType("gaitbench.capture.realsense")
    fake.list_modes = lambda: [{"stream": "color", "width": 640, "height": 480, "fps": 60}]

    # Same signature and defaults as gaitbench.capture.realsense.record.
    def record(out_dir, seconds, color=(1280, 800, 30), depth=(848, 480, 30), clip_id=None, *, serial=None):
        recorded.update(out_dir=out_dir, seconds=seconds, color=color, depth=depth, clip_id=clip_id)
        return Path(out_dir) / "pass.db3"

    fake.DEFAULT_COLOR = (1280, 800, 30)
    fake.DEFAULT_DEPTH = (848, 480, 30)
    fake.record = record
    fake.bag_to_video = lambda bag, out_dir: [Path(out_dir) / "color.mp4"]
    monkeypatch.setitem(sys.modules, "gaitbench.capture.realsense", fake)
    assert run("realsense-modes") == 0
    assert "color" in capsys.readouterr().out
    assert run("realsense-record", "--out", tmp_path, "--duration", 3, "--fps", 60, "--clip-id", "pass") == 0
    assert recorded == {"out_dir": str(tmp_path), "seconds": 3.0, "color": (1280, 800, 60),
                        "depth": (848, 480, 60), "clip_id": "pass"}
    assert run("realsense-record", "--out", tmp_path, "--duration", 3, "--no-depth") == 0
    assert recorded["depth"] is None and recorded["color"] == (1280, 800, 30)
    assert run("realsense-export", tmp_path / "a.bag", tmp_path / "x") == 0
    assert "color.mp4" in capsys.readouterr().out


def test_missing_optional_module_is_a_clean_error(monkeypatch, capsys) -> None:
    monkeypatch.setitem(sys.modules, "gaitbench.capture.realsense", None)
    assert run("realsense-modes") == 1
    assert "RealSense capture is not available" in capsys.readouterr().err


def test_console_script_help() -> None:
    proc = subprocess.run([sys.executable, "-m", "gaitbench.cli", "--help"], capture_output=True, text=True)
    assert proc.returncode == 0 and "detect" in proc.stdout
    with pytest.raises(SystemExit):
        cli.main(["nonsense"])

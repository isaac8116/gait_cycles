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
    err = capsys.readouterr().err
    assert "RealSense capture is not available" in err
    # RS-2: pyrealsense2-macosx 2.56 is macOS 15+ only; older Macs are pointed at conda-forge.
    assert "macOS 15+" in err and "conda install -c conda-forge pyrealsense2" in err


def test_console_script_help() -> None:
    proc = subprocess.run([sys.executable, "-m", "gaitbench.cli", "--help"], capture_output=True, text=True)
    assert proc.returncode == 0 and "detect" in proc.stdout
    with pytest.raises(SystemExit):
        cli.main(["nonsense"])


# ---------------------------------------------------------------------------
# Regression tests for the review findings
# ---------------------------------------------------------------------------


def _crash_on(monkeypatch, detector: str, clip: str) -> None:
    """Make ``detector`` raise on ``clip`` (and work elsewhere)."""
    real = detectors.get_detector

    def flaky(name: str, **params):
        runner = real(name, **params)
        if name != detector:
            return runner

        def maybe(kp):
            if kp.clip_id == clip:
                raise RuntimeError("simulated crash")
            return runner(kp)
        return maybe

    monkeypatch.setattr(detectors, "get_detector", flaky)


def test_crashed_detector_counts_its_clips_as_missed(data: Path, tmp_path: Path, monkeypatch) -> None:
    """SCO-1 / FAIL-1: a crash removes the stale events file and the clip's labels count as FN."""
    pytest.importorskip("gaitbench.scoring")
    root = tmp_path / "d"
    shutil.copytree(data, root)
    stale = events_path(root, "pan", "synthetic", "stenum")
    assert stale.is_file()  # from the clean run in the fixture
    _crash_on(monkeypatch, "stenum", "pan")
    done, failures = cli.detect_all(root, ["zeni", "stenum"], echo=lambda s: None)
    assert list(failures["clip_id"]) == ["pan"] and not stale.exists()

    with pytest.warns(UserWarning, match=r"stenum: no events file for labelled clip\(s\) \['pan'\]"):
        result = cli.score_events(root, "truth", detectors=["zeni", "stenum"])
    metrics = result["metrics"].set_index(["detector", "event"])
    for event in ("HS", "TO"):
        assert metrics.loc[("stenum", event), "n_truth"] == metrics.loc[("zeni", event), "n_truth"]
        assert metrics.loc[("stenum", event), "recall"] < 0.6
    assert metrics.loc[("stenum", "HS"), ["n_clips", "n_clips_missing"]].tolist() == [2, 1]
    assert metrics.loc[("zeni", "HS"), ["n_clips", "n_clips_missing"]].tolist() == [2, 0]
    assert list(result["metrics"].columns[:5]) == ["backend", "detector", "event", "n_clips", "n_clips_missing"]
    assert set(result["metrics_by_side"]["n_clips_missing"]) == {0, 1}
    # Parameters are judged on the same clips as the metrics.
    params = result["params"].set_index(["detector", "parameter"])
    n_truth_cycles = params.loc[(slice(None), "stride_time_s"), "n_truth_cycles"]
    assert n_truth_cycles["stenum"].iloc[0] == n_truth_cycles["zeni"].iloc[0] > 0

    # A requested detector that crashed everywhere is still scored, with every label missed.
    _crash_on(monkeypatch, "stenum", "fixed")
    cli.detect_all(root, ["stenum"], clips=["fixed"], echo=lambda s: None)
    with pytest.warns(UserWarning):
        everywhere = cli.score_events(root, "truth", detectors=["zeni", "stenum"])["metrics"]
    stenum = everywhere[everywhere["detector"] == "stenum"]
    assert (stenum["recall"] == 0).all() and (stenum["n_clips_missing"] == 2).all()


def test_run_overwrite_empties_the_run_directory(tmp_path: Path) -> None:
    """FAIL-1: --overwrite leaves no result of the earlier run beside the new record."""
    pytest.importorskip("gaitbench.scoring")
    config = {"run_id": "r", "data_root": str(tmp_path / "data"), "results_root": str(tmp_path / "runs"),
              "simulate": [{"clip_id": "c", "fps": 60, "duration": 6}], "detectors": ["ours"],
              "truth_rater": "truth"}
    out = cli.run_config(config, echo=lambda s: None)
    (out / "plots").mkdir()
    (out / "plots" / "old.png").write_bytes(b"x")
    assert (out / "metrics.csv").is_file()
    again = cli.run_config({**config, "truth_rater": None, "reference_detector": "zeni"}, overwrite=True,
                           echo=lambda s: None)
    assert again == out
    assert not (out / "metrics.csv").exists() and not (out / "plots").exists()
    assert (out / "agreement.csv").is_file() and (out / "config.yaml").is_file()
    # Never empties a directory that holds the data.
    inside = {**config, "data_root": str(out / "data"), "simulate": []}
    (out / "data").mkdir()
    with pytest.raises(cli.CLIError, match="inside the run directory"):
        cli.run_config(inside, overwrite=True, echo=lambda s: None)
    with pytest.raises(cli.CLIError, match="not a usable directory name"):
        cli.run_config({**config, "run_id": "."}, overwrite=True, echo=lambda s: None)


def test_params_keep_empty_and_unnamed_sources(data: Path, tmp_path: Path) -> None:
    """SCO-6: an empty detector still has a params row; files with blank backends are not pooled."""
    pytest.importorskip("gaitbench.scoring")
    root = tmp_path / "d"
    shutil.copytree(data, root)
    for clip in ("fixed", "pan"):
        write_events(empty_events(), events_path(root, clip, "synthetic", "stenum"))
    params = cli.score_events(root, "truth")["params"]
    stride = params[params["parameter"] == "stride_time_s"].set_index("detector")
    assert set(stride.index) == {"zeni", "stenum"} and set(stride["backend"]) == {"synthetic"}
    assert stride.loc["stenum", "n_pred_cycles"] == 0
    assert stride.loc["stenum", "n_truth_cycles"] == stride.loc["zeni", "n_truth_cycles"] > 0

    blank = tmp_path / "blank"
    (blank / "labels").mkdir(parents=True)
    shutil.copy(labels_path(data, "fixed", "truth"), blank / "labels")
    events = read_events(events_path(data, "fixed", "synthetic", "zeni")).assign(backend="")
    for cam in ("camA", "camB"):
        write_events(events, events_path(blank, "fixed", cam, "zeni"))
    params = cli.score_events(blank, "truth")["params"]
    stride = params[params["parameter"] == "stride_time_s"].set_index("backend")
    assert set(stride.index) == {"camA", "camB"}
    assert (stride["n_pairs"] > 0).all() and (stride["n_pred_cycles"] == stride["n_truth_cycles"]).all()


def test_detect_names_events_by_the_file_name_backend(tmp_path: Path) -> None:
    """NAME-1: a renamed copy's events never overwrite the original's."""
    root = tmp_path / "d"
    cli.simulate_clip(root, "IMG_1234", fps=240)
    kp = Keypoints.load(keypoints_path(root, "IMG_1234", "synthetic"))
    copy = cli.decimate_keypoints(kp, 8, backend="synthetic")  # meta still says the model name
    copy.save(keypoints_path(root, "IMG_1234", "synthetic-cfr30"))
    with pytest.warns(UserWarning, match="file name says 'synthetic-cfr30'"):
        cli.detect_all(root, ["zeni"], echo=lambda s: None)
    original = read_events(events_path(root, "IMG_1234", "synthetic", "zeni"))
    renamed = read_events(events_path(root, "IMG_1234", "synthetic-cfr30", "zeni"))
    assert set(original["backend"]) == {"synthetic"} and set(renamed["backend"]) == {"synthetic-cfr30"}


def _fake_pose(monkeypatch, seen: dict) -> None:
    from gaitbench import myogait_bridge
    from gaitbench.synthetic import simulate_walk

    def fake_extract(video_path, model="mediapipe", clip_id=None, *, timestamps=None, **kwargs):
        seen.update(model=model, clip_id=clip_id, timestamps=timestamps)
        kp = simulate_walk(fps=30, duration_s=2.0).keypoints
        n = kp.n_frames
        times = kp.time_s if timestamps is None else timestamps[:n]
        return Keypoints(clip_id=clip_id, names=kp.names, xy=kp.xy, conf=kp.conf, time_s=times,
                         frame_idx=np.arange(n), meta={**kp.meta, "backend": model})

    monkeypatch.setattr(myogait_bridge, "extract_keypoints", fake_extract)


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg")
def test_pose_out_backend_and_realsense_times(tmp_path: Path, monkeypatch) -> None:
    """NAME-1: --out-backend names the keypoints; RS-1: pose reads the RealSense frames CSV like the labeller."""
    from gaitbench.capture.realsense import frame_times_path, frame_times_table, write_frame_times
    from gaitbench.labelling.app import load_timestamps as labeller_times

    video = tmp_path / "IMG_1234_30.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30", "-frames:v", "60",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    seen: dict = {}
    _fake_pose(monkeypatch, seen)
    root = tmp_path / "d"
    assert run("pose", "--data", root, "--video", video, "--clip-id", "IMG_1234", "--out-backend",
               "mediapipe-cfr30") == 0
    assert seen["model"] == "mediapipe"
    kp = Keypoints.load(keypoints_path(root, "IMG_1234", "mediapipe-cfr30"))
    assert kp.meta["backend"] == "mediapipe-cfr30" and kp.meta["time_source"] == "video"
    assert not keypoints_path(root, "IMG_1234", "mediapipe").exists()

    # Hardware times with 3 dropped frames after frame 30.
    hw = [1000.0 + 1000.0 / 30 * (i if i <= 30 else i + 3) for i in range(60)]
    rows = [{"frame": i, "rs_frame_number": i + (0 if i <= 30 else 3), "timestamp_ms": t} for i, t in enumerate(hw)]
    write_frame_times(frame_times_table(rows), frame_times_path(video))
    assert run("pose", "--data", root, "--video", video, "--clip-id", "rs") == 0
    label_times, source, _ = labeller_times(video)
    np.testing.assert_allclose(seen["timestamps"][:60], label_times)
    kp = Keypoints.load(keypoints_path(root, "rs", "mediapipe"))
    assert kp.meta["time_source"] == source == "realsense:IMG_1234_30.frames.csv"
    assert kp.time_s[50] == pytest.approx(53 / 30)


def _tick600(n: int, meta_fps: float | None = 240.0, backend: str = "phone240") -> Keypoints:
    """240 fps keypoints in a 1/600 s time base: intervals alternate 2 and 3 ticks."""
    from gaitbench.synthetic import simulate_walk

    kp = simulate_walk(fps=240, duration_s=n / 240).keypoints
    meta = {**kp.meta, "backend": backend}
    if meta_fps is None:
        meta.pop("fps", None)
    return Keypoints(kp.clip_id, kp.names, kp.xy[:n], kp.conf[:n], np.floor(np.arange(n) * 2.5) / 600,
                     kp.frame_idx[:n], meta)


@pytest.mark.parametrize("n", [1920, 1919, 480])
@pytest.mark.parametrize("meta_fps", [240.0, None])
def test_rate_of_tick_quantised_240fps(n: int, meta_fps) -> None:
    """RATE-1 / SCO-2(a): the rate is the mean one, not 200 or 300 fps from the median interval."""
    kp = _tick600(n, meta_fps)
    assert cli._sample_rate(kp) == pytest.approx(240.0, rel=1e-3)
    assert cli.decimate_keypoints(kp, 4).meta["backend"] == "phone240-60fps"
    assert cli.decimate_keypoints(kp, 8).meta["backend"] == "phone240-30fps"


def test_frame_metrics_use_the_mean_rate(tmp_path: Path) -> None:
    """RATE-1: a detector one 240 fps frame late on tick-quantised keypoints is judged in 240 fps frames."""
    pytest.importorskip("gaitbench.scoring")
    from gaitbench.formats import make_events, read_labels, write_labels

    root = tmp_path / "d"
    cli.simulate_clip(root, "slowmo", fps=240, duration=8)
    keypoints_path(root, "slowmo", "synthetic").unlink()
    kp = _tick600(1920)
    kp.clip_id = "slowmo"
    kp.save(keypoints_path(root, "slowmo", "phone240"))
    truth = read_labels(labels_path(root, "slowmo", "truth"))
    truth["time_s"] = kp.time_s[truth["frame"].to_numpy()]
    write_labels(truth, labels_path(root, "slowmo", "truth"))
    late = [{"side": r.side, "event": r.event, "frame": int(r.frame) + 1, "time_s": float(kp.time_s[int(r.frame) + 1])}
            for r in truth.itertuples()]
    write_events(make_events(late, defaults={"clip_id": "slowmo", "backend": "phone240", "detector": "late1"}),
                 events_path(root, "slowmo", "phone240", "late1"))
    metrics = cli.score_events(root, "truth")["metrics"].set_index("event")
    error_ms = metrics["mae_ms"]
    np.testing.assert_allclose(metrics["mae_frames"], error_ms / 1000 * 240)  # 240 fps frames, not 300 or 200
    assert (metrics["mae_frames"] < 1.25).all()
    # SCO-2(b): exactly one frame late is within one frame, whichever 2- or 3-tick interval it crossed.
    assert (metrics["p_within_1_frame"] == 1.0).all()


def test_decimate_selects_source_frames_shared_by_backends() -> None:
    """DEC-1: copies of different backends keep the same source frames."""
    def kp(first: int, backend: str, n: int = 200) -> Keypoints:
        frames = np.arange(first, first + n)
        return Keypoints(clip_id="c", names=["LEFT_HEEL"], xy=np.zeros((n, 1, 2)), conf=np.ones((n, 1)),
                         time_s=frames / 240, frame_idx=frames, meta={"backend": backend, "fps": 240.0})

    a = cli.decimate_keypoints(kp(37, "mediapipe"), 4)
    b = cli.decimate_keypoints(kp(40, "rtmw"), 4)
    assert (a.meta["backend"], b.meta["backend"]) == ("mediapipe-60fps", "rtmw-60fps")
    assert a.frame_idx[0] == 40 and (a.frame_idx % 4 == 0).all() and (b.frame_idx % 4 == 0).all()
    np.testing.assert_array_equal(a.frame_idx, b.frame_idx)  # 40, 44, ..., 236 for both
    shifted = cli.decimate_keypoints(kp(37, "mediapipe"), 4, offset=1)
    assert shifted.frame_idx[0] == 37 and (shifted.frame_idx % 4 == 1).all()
    # A gap in the source frames does not shift the phase of later frames.
    gappy = kp(0, "mediapipe")
    keep = np.r_[0:50, 53:200]
    gappy = Keypoints("c", gappy.names, gappy.xy[keep], gappy.conf[keep], gappy.time_s[keep], gappy.frame_idx[keep],
                      gappy.meta)
    assert (cli.decimate_keypoints(gappy, 4).frame_idx % 4 == 0).all()


def test_hidden_files_are_ignored(data: Path, tmp_path: Path) -> None:
    """MAC-1: macOS '._' AppleDouble files beside the data are not read."""
    pytest.importorskip("gaitbench.scoring")
    root = tmp_path / "d"
    shutil.copytree(data, root)
    junk = b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X        " + b"\xff\xfe" * 100
    for rel in ("keypoints/._pan__synthetic.npz", "events/._pan__synthetic__zeni.csv", "labels/._pan__truth.csv",
                "keypoints/.hidden__synthetic.npz"):
        (root / rel).write_bytes(junk)
    assert all(not f.path.name.startswith(".") for f in cli.keypoint_files(root))
    assert all(not f.path.name.startswith(".") for f in cli.event_files(root))
    assert [f.clip for f in cli.label_files(root, "truth")] == ["fixed", "pan"]
    assert run("detect", "--data", root, "--detectors", "zeni") == 0
    assert run("score", "--data", root, "--out", tmp_path / "res") == 0
    assert run("clean", "--data", root, "--clip", "pan") == 0


def _swapped(kp: Keypoints, frames) -> Keypoints:
    """LEFT_/RIGHT_ leg points exchanged on ``frames``, as a pose model does."""
    from gaitbench.clean import LEG_PARTS

    xy, conf = kp.xy.copy(), kp.conf.copy()
    rows = np.asarray(frames)
    for part in LEG_PARTS:
        if kp.has("LEFT_" + part) and kp.has("RIGHT_" + part):
            li, ri = kp.index("LEFT_" + part), kp.index("RIGHT_" + part)
            xy[rows, li], xy[rows, ri] = kp.xy[rows, ri], kp.xy[rows, li]
            conf[rows, li], conf[rows, ri] = kp.conf[rows, ri], kp.conf[rows, li]
    return Keypoints(kp.clip_id, kp.names, xy, conf, kp.time_s, kp.frame_idx, kp.meta)


def test_clean_command_writes_clean_copies(tmp_path: Path, capsys) -> None:
    root = tmp_path / "d"
    cli.simulate_clip(root, "walk", fps=60)
    path = keypoints_path(root, "walk", "synthetic")
    _swapped(Keypoints.load(path), [100, 101, 102]).save(path)
    assert run("clean", "--data", root, "--steps", "leg_swaps") == 0
    assert "3 frame(s) swapped" in capsys.readouterr().out
    cleaned = Keypoints.load(keypoints_path(root, "walk", "synthetic-clean"))
    assert cleaned.meta["backend"] == "synthetic-clean"
    assert cleaned.meta["cleaned"]["leg_swaps"]["n_swapped"] == 3
    report = pd.read_csv(root / "results" / "clean" / "walk__synthetic.csv")
    assert set(report.loc[report["action"] == "swap", "frame"]) == {100, 101, 102}
    # Cleaned files are not cleaned again; typos fail before anything is written.
    assert run("clean", "--data", root, "--backend", "synthetic-clean") == 1
    assert "already ending in '-clean'" in capsys.readouterr().err
    assert run("clean", "--data", root, "--steps", "leg_swap") == 1
    assert run("clean", "--data", root, "--params", '{"leg_swaps": {"nope": 1}}') == 1
    assert run("clean", "--data", root, "--backend", "all", "--clip", "all", "--out", tmp_path / "rep") == 0
    assert (tmp_path / "rep" / "walk__synthetic.csv").is_file()


def test_run_config_cleans_and_scores_raw_and_clean(tmp_path: Path) -> None:
    pytest.importorskip("gaitbench.scoring")
    config = {"run_id": "clean", "data_root": str(tmp_path / "data"), "results_root": str(tmp_path / "runs"),
              "simulate": [{"clip_id": "walk", "fps": 60, "duration": 6}], "backends": ["synthetic"],
              "clean": {"steps": ["leg_swaps"], "params": {"leg_swaps": {"min_conf": 0.1}}},
              "detectors": ["zeni"], "truth_rater": "truth"}
    out = cli.run_config(config, echo=lambda s: None)
    metrics = pd.read_csv(out / "metrics.csv")
    assert set(metrics["backend"]) == {"synthetic", "synthetic-clean"}
    assert (metrics["recall"] == 1.0).all()
    assert pd.read_csv(out / "clean.csv")["cleaned_backend"].tolist() == ["synthetic-clean"]
    assert (out / "clean" / "walk__synthetic.csv").is_file()
    yaml = pytest.importorskip("yaml")
    assert yaml.safe_load((out / "config.yaml").read_text())["backends_expanded"] == ["synthetic", "synthetic-clean"]
    with pytest.raises(cli.CLIError, match="Unknown keys in clean"):
        cli.run_config({**config, "run_id": "x", "clean": {"step": ["leg_swaps"]}}, echo=lambda s: None)


def test_bundled_gpjatk_config_cleans_openpose() -> None:
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "l0_gpjatk_sample.yaml").read_text())
    assert cfg["clean"]["steps"] == ["leg_swaps"] and cfg["backends"] == ["openpose"]
    assert cli._clean_config(cfg["clean"])[0] == ["leg_swaps"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg")
def test_cfr_command_moves_the_audio_with_a_late_video(tmp_path: Path, capsys) -> None:
    """CFR-1: when the video starts after the audio, gaitbench cfr shifts the audio by as much."""
    video, src = tmp_path / "v.mp4", tmp_path / "late.mov"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=240:duration=0.5",
                    "-pix_fmt", "yuv420p", str(video)], check=True, timeout=60)
    subprocess.run(["ffmpeg", "-v", "error", "-itsoffset", "0.1", "-i", str(video), "-f", "lavfi", "-i",
                    "sine=duration=0.6", "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "pcm_s16le", str(src)],
                   check=True, timeout=60)
    assert run("cfr", src, tmp_path / "out.mp4", "--fps", 60) == 0
    printed = capsys.readouterr().out
    assert "setpts=PTS-STARTPTS,fps=60:round=up" in printed and "atrim=start=0.1" in printed
    assert run("cfr", src, tmp_path / "out.mp4", "--fps", 60, "--no-audio") == 0
    assert "atrim" not in capsys.readouterr().out


def test_env_prints_the_opencv_repair(monkeypatch, capsys) -> None:
    """DEP-1: `gaitbench env` notices a headless cv2 and prints the repair, not an uninstall."""
    from gaitbench import env

    fake = types.ModuleType("cv2")
    fake.__version__ = "5.0.0"
    fake.getBuildInformation = lambda: "General configuration\n  GUI:                           NONE\n"
    monkeypatch.setitem(sys.modules, "cv2", fake)
    monkeypatch.setattr(env, "package_version", lambda name: "5.0.0.93" if name.startswith("opencv") else None)
    assert run("env") == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["opencv"]["has_gui"] is False
    assert "pip install --force-reinstall --no-deps opencv-python==5.0.0.93" in captured.err
    assert "uninstall" not in captured.err.replace("without uninstalling", "")

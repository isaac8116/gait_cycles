"""Tests for gaitbench.env."""

from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

import pytest

import gaitbench
from gaitbench import env
from gaitbench.env import PACKAGES, git_info, installed_distributions, opencv_gui, package_version, snapshot


def test_snapshot_is_json_and_complete() -> None:
    snap = snapshot()
    json.dumps(snap)
    assert set(snap["versions"]) == set(PACKAGES)
    assert snap["versions"]["numpy"]
    assert snap["versions"]["gaitbench"] in (gaitbench.__version__, None)
    assert snap["python"]["version"].count(".") == 2
    assert snap["platform"]["machine"]
    assert snap["platform"]["rosetta"] in (True, False, None)
    assert any(line.lower().startswith("numpy==") for line in snap["installed"])
    assert "installed" not in snapshot(all_packages=False)


def test_missing_package_is_none() -> None:
    assert package_version("surely-not-installed-gaitbench-test") is None
    names = [line.split("==")[0].lower() for line in installed_distributions()]
    assert names == sorted(names) and len(names) == len(set(names))


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_info(tmp_path: Path) -> None:
    assert git_info(tmp_path) is None
    info = git_info(Path(gaitbench.__file__))
    if info is None:
        pytest.skip("gaitbench is not installed from a git checkout")
    assert set(info) == {"commit", "branch", "dirty", "root"}
    assert info["commit"] is None or len(info["commit"]) == 40
    assert isinstance(info["dirty"], bool)


def _fake_cv2(monkeypatch, gui: str | None) -> None:
    fake = types.ModuleType("cv2")
    fake.__version__ = "5.0.0"
    lines = ["General configuration for OpenCV 5.0.0", "  Version control: 5.0.0"]
    if gui is not None:
        lines.append(f"  GUI:                           {gui}")
    fake.getBuildInformation = lambda: "\n".join(lines)
    monkeypatch.setitem(sys.modules, "cv2", fake)


def test_opencv_gui_reports_a_headless_build_and_its_repair(monkeypatch) -> None:
    """DEP-1: the repair reinstalls opencv-python at the installed version; it never uninstalls."""
    versions = {"opencv-python": "5.0.0.93", "opencv-python-headless": "5.0.0.93"}
    monkeypatch.setattr(env, "package_version", versions.get)
    _fake_cv2(monkeypatch, "NONE")
    info = opencv_gui()
    assert info == {"version": "5.0.0", "gui": "NONE", "has_gui": False,
                    "fix": "pip install --force-reinstall --no-deps opencv-python==5.0.0.93"}
    _fake_cv2(monkeypatch, "QT5")
    assert opencv_gui()["has_gui"] is True and opencv_gui()["fix"] is None
    _fake_cv2(monkeypatch, "COCOA")
    assert opencv_gui()["gui"] == "COCOA" and opencv_gui()["has_gui"]
    _fake_cv2(monkeypatch, None)  # a build that does not say
    assert opencv_gui()["has_gui"] is False
    # Only the headless package installed: pin opencv-python to its version.
    monkeypatch.setattr(env, "package_version", {"opencv-python-headless": "4.12.0.88"}.get)
    assert env.opencv_repair_command().endswith("opencv-python==4.12.0.88")
    monkeypatch.setitem(sys.modules, "cv2", None)  # not importable
    assert opencv_gui() is None


def test_install_notes_in_pyproject() -> None:
    """RS-2 / DEP-1: the extras document the macOS 15 floor of pyrealsense2-macosx and the OpenCV repair."""
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    realsense = text[text.index("# Intel RealSense"):text.index("realsense = [")]
    assert "macOS 15" in realsense and "conda-forge pyrealsense2" in realsense
    video = text[text.index("# Video reading"):text.index("video = [")]
    assert "pip install --force-reinstall --no-deps opencv-python==" in video and "uninstall" in video

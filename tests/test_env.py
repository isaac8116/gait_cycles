"""Tests for gaitbench.env."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import gaitbench
from gaitbench.env import PACKAGES, git_info, installed_distributions, package_version, snapshot


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

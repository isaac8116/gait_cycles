"""scripts/rename_from_log.py: clip ids from the session log."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "rename_from_log", Path(__file__).resolve().parents[1] / "scripts" / "rename_from_log.py")
rename = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rename)

HEADER = "file,participant,code,direction,speed,clothing,take,app,notes\n"


def _log(tmp_path: Path, rows: str) -> Path:
    path = tmp_path / "session_log.csv"
    path.write_text(HEADER + rows)
    return path


def test_clip_id_pads_the_take_and_keeps_codes_readable() -> None:
    row = {"participant": "p01", "code": "F240-S", "direction": "LtoR", "speed": "slow", "take": "1"}
    assert rename.clip_id(row) == "p01_F240-S_LtoR_slow_01"
    assert rename.clip_id({**row, "speed": "very slow"}) == "p01_F240-S_LtoR_very-slow_01"


def test_copies_skips_missing_and_never_overwrites(tmp_path: Path) -> None:
    (tmp_path / "IMG_1234.MOV").write_bytes(b"a")
    log = _log(tmp_path, "IMG_1234.MOV,p01,F240,LtoR,normal,fitted,1,Camera,\n"
                         "IMG_9999.MOV,p02,M240,RtoL,normal,fitted,2,Camera,\n")
    assert rename.main([str(log), str(tmp_path)]) == 0
    target = tmp_path / "p01_F240_LtoR_normal_01.MOV"
    assert target.read_bytes() == b"a" and (tmp_path / "IMG_1234.MOV").exists()
    target.write_bytes(b"edited")
    assert rename.main([str(log), str(tmp_path)]) == 0
    assert target.read_bytes() == b"edited"


def test_dry_run_and_move(tmp_path: Path) -> None:
    (tmp_path / "IMG_1.MOV").write_bytes(b"a")
    log = _log(tmp_path, "IMG_1.MOV,p01,M240,RtoL,normal,fitted,3,Camera,\n")
    rename.main([str(log), str(tmp_path), "--dry-run"])
    assert not (tmp_path / "p01_M240_RtoL_normal_03.MOV").exists()
    rename.main([str(log), str(tmp_path), "--move"])
    assert (tmp_path / "p01_M240_RtoL_normal_03.MOV").exists() and not (tmp_path / "IMG_1.MOV").exists()


def test_duplicate_clip_ids_and_missing_columns_are_refused(tmp_path: Path) -> None:
    log = _log(tmp_path, "IMG_1.MOV,p01,F240,LtoR,normal,fitted,1,Camera,\n"
                         "IMG_2.MOV,p01,F240,LtoR,normal,fitted,1,Camera,\n")
    with pytest.raises(SystemExit, match="same clip id"):
        rename.plan(log, tmp_path, tmp_path)
    bad = tmp_path / "bad.csv"
    bad.write_text("file,participant\nIMG_1.MOV,p01\n")
    with pytest.raises(SystemExit, match="missing columns"):
        rename.plan(bad, tmp_path, tmp_path)

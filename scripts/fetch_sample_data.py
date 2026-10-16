#!/usr/bin/env python3
"""Download the GPJATK level-0 sample (OpenPose JSON + rendered video) into data/raw/gpjatk_sample/.

The two zips come from the ``Multiple-Perspectives`` branch of Jan Stenum's
GaitAnalysis-PoseEstimation repository: subject p01 of the GPJATK dataset
(Kwolek et al. 2019) seen by the left and right sagittal cameras, 135 frames
at 25 fps, 960x540, already run through OpenPose BODY_25.

GPJATK is released for scientific use under a release agreement: cite the
paper and do not redistribute the files. They must stay out of git; ``data/``
is git-ignored for this reason.

Usage::

    python scripts/fetch_sample_data.py [--out data/raw/gpjatk_sample] [--force]

then ``gaitbench run configs/l0_gpjatk_sample.yaml``. Uses only the standard
library; honours ``HTTPS_PROXY`` and, for a custom CA bundle, ``SSL_CERT_FILE``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/janstenum/GaitAnalysis-PoseEstimation/Multiple-Perspectives"
NAMES = ("C_Sag_Left", "C_Sag_Right")
DEFAULT_OUT = Path("data") / "raw" / "gpjatk_sample"

NOTICE = """\
These files come from the GPJATK dataset:
  Kwolek B, Michalczuk A, Krzeszowski T, Switonski A, Josinski H, Wojciechowski K (2019).
  Calibrated and synchronized multi-view video and motion capture dataset for evaluation
  of gait recognition. Multimedia Tools and Applications 78.
  doi:10.1007/s11042-019-07945-y
They are provided for scientific use under the dataset's release agreement:
cite the paper, and do not redistribute them. Keep them out of git (data/ is git-ignored).
OpenPose output prepared by J. Stenum (github.com/janstenum/GaitAnalysis-PoseEstimation)."""


def download(url: str, dest: Path, timeout: float = 60.0) -> Path:
    """Download ``url`` to ``dest`` via a temporary ``.part`` file, so an interrupted run leaves no bad zip."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as response, open(part, "wb") as out:
        shutil.copyfileobj(response, out)
    part.replace(dest)
    return dest


def safe_extract(archive: Path, out_dir: Path) -> list[Path]:
    """Extract ``archive`` into ``out_dir``, refusing members that would land outside it."""
    root = out_dir.resolve()
    written = []
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if member.filename.startswith("__MACOSX/") or Path(member.filename).name == ".DS_Store":
                continue
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(f"{archive.name}: member {member.filename!r} would extract outside {root}")
            zf.extract(member, root)
            if not member.is_dir():
                written.append(target)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"target directory (default {DEFAULT_OUT})")
    parser.add_argument("--force", action="store_true", help="download again even if the zip is already there")
    args = parser.parse_args(argv)

    print(NOTICE + "\n")
    status = 0
    for name in NAMES:
        archive = args.out / f"{name}.zip"
        try:
            if archive.exists() and not args.force:
                print(f"{archive} exists; not downloading again (--force to refresh)")
            else:
                url = f"{BASE_URL}/{name}.zip"
                print(f"downloading {url}")
                download(url, archive)
            files = safe_extract(archive, args.out)
        except (urllib.error.URLError, OSError, zipfile.BadZipFile, RuntimeError) as exc:
            print(f"error: {name}: {exc}", file=sys.stderr)
            status = 1
            continue
        n_json = sum(f.name.endswith("_keypoints.json") for f in files)
        print(f"  {name}: {len(files)} files ({n_json} OpenPose JSON) in {args.out / name}")
    if status == 0:
        print("\nNext: gaitbench run configs/l0_gpjatk_sample.yaml")
    return status


if __name__ == "__main__":
    sys.exit(main())

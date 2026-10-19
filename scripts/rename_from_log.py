#!/usr/bin/env python
"""Give phone clips their clip ids from the session log.

The recording protocol (docs/recording-protocol.md) keeps the phone's own
file names (``IMG_1234.MOV``) during the session and logs one row per take in
``session_log.csv``. This script copies each logged file to
``<participant>_<code>_<direction>_<speed>_<take>.<ext>``, e.g.
``p01_F240_LtoR_normal_01.MOV``, which then becomes the clip id of every file
the harness writes for it.

    python scripts/rename_from_log.py data/raw/phone/session_log.csv data/raw/phone
    python scripts/rename_from_log.py LOG VIDEO_DIR --dry-run     # show what would happen
    python scripts/rename_from_log.py LOG VIDEO_DIR --move        # rename instead of copying

Existing targets are never overwritten, and a log row whose file is missing is
reported and skipped.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

REQUIRED = ("file", "participant", "code", "direction", "speed", "take")
_UNSAFE = re.compile(r"[^A-Za-z0-9.-]+")


def clip_id(row: dict[str, str]) -> str:
    """``p01_F240_LtoR_normal_01`` from one log row (fields made filename-safe)."""
    take = row["take"].strip()
    take = f"{int(take):02d}" if take.isdigit() else take
    parts = [row["participant"], row["code"], row["direction"], row["speed"], take]
    return "_".join(_UNSAFE.sub("-", p.strip()).strip("-") for p in parts)


def plan(log_csv: Path, video_dir: Path, out_dir: Path) -> list[tuple[Path, Path]]:
    """``(source, target)`` pairs for every log row; raises on a bad log."""
    with open(log_csv, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{log_csv}: missing columns {missing} (see docs/templates/session_log.csv)")
        rows = [r for r in reader if (r.get("file") or "").strip()]
    pairs, seen = [], {}
    for row in rows:
        src = video_dir / row["file"].strip()
        target = out_dir / (clip_id(row) + src.suffix)
        if target.name in seen:
            raise SystemExit(f"Two log rows give the same clip id {target.stem!r}: "
                             f"{seen[target.name]} and {src.name}. Give them different take numbers.")
        seen[target.name] = src.name
        pairs.append((src, target))
    return pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log_csv", type=Path)
    ap.add_argument("video_dir", type=Path, help="folder holding the phone's original files")
    ap.add_argument("--out", type=Path, help="where the renamed clips go (default: video_dir)")
    ap.add_argument("--move", action="store_true", help="rename instead of copying")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    out_dir = args.out or args.video_dir
    pairs = plan(args.log_csv, args.video_dir, out_dir)
    done = skipped = 0
    for src, target in pairs:
        if not src.exists():
            print(f"missing   {src.name} (logged as {target.stem})", file=sys.stderr)
            skipped += 1
            continue
        if target.exists():
            print(f"exists    {target.name} (left alone)")
            skipped += 1
            continue
        print(f"{'move' if args.move else 'copy'}      {src.name} -> {target.name}")
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            (shutil.move if args.move else shutil.copy2)(src, target)
        done += 1
    print(f"{done} clip(s) {'would be ' if args.dry_run else ''}named, {skipped} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

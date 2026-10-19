"""Record the software environment of a run, so results can be traced to exact versions.

:func:`snapshot` is written next to every run's results (``env.json``). Event
timing can shift between releases of a pose model or detector (plan E6,
golden set), so a number without the versions behind it cannot be compared
with a later one.

Apple Silicon note: an x86_64 Python running under Rosetta reports
``machine = "x86_64"`` on an arm64 Mac. It works but is several times slower
and pulls different wheels (e.g. ``pyrealsense2-macosx`` exists only for
arm64), so ``rosetta`` is reported separately.

OpenCV note: ``opencv-python`` (the ``video`` extra, with windows),
``opencv-python-headless`` (required by myogait) and
``opencv-contrib-python`` (required by mediapipe) all install the same
``cv2`` package, and whichever was installed last wins. If that is the
headless one, the labeller cannot open a window. :func:`opencv_gui` checks
the ``GUI`` line of ``cv2.getBuildInformation()``, and ``gaitbench env``
prints the repair, ``pip install --force-reinstall --no-deps
opencv-python==<installed version>``. Uninstalling one of the three is not
a fix: it deletes files the others share and leaves a broken ``cv2``.
"""

from __future__ import annotations

import datetime as _dt
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Distribution names (as pip knows them) whose versions are recorded.
PACKAGES: tuple[str, ...] = (
    "gaitbench",
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "pyyaml",
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "mediapipe",
    "rtmlib",
    "onnxruntime",
    "myogait",
    "gaitkit",
    "pyrealsense2",
    "pyrealsense2-macosx",
)
# The distributions that install ``cv2``; see the module docstring.
OPENCV_DISTRIBUTIONS: tuple[str, ...] = ("opencv-python", "opencv-python-headless", "opencv-contrib-python")


def package_version(name: str) -> str | None:
    """Installed version of a distribution, or None if it is not installed."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_info(path: str | Path | None = None) -> dict[str, Any] | None:
    """Commit and dirty flag of the git repository containing ``path``.

    Parameters
    ----------
    path:
        Any path inside the repository; defaults to this package's source
        directory, which is the gaitbench repository for an editable install.

    Returns
    -------
    dict or None
        ``{"commit", "branch", "dirty", "root"}``; ``dirty`` counts untracked
        files too, since a new, uncommitted module changes results as much as
        an edited one. None if git is missing or ``path`` is not in a repository.
    """
    git = shutil.which("git")
    if git is None:
        return None
    where = Path(path) if path is not None else Path(__file__).resolve().parent
    if where.is_file():
        where = where.parent

    def run(*args: str) -> str | None:
        try:
            proc = subprocess.run([git, "-C", str(where), *args], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    root = run("rev-parse", "--show-toplevel")
    if root is None:
        return None
    status = run("status", "--porcelain")
    return {
        # None in a repository with no commits yet.
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "root": root,
    }


def opencv_repair_command() -> str:
    """pip command that puts OpenCV's GUI build back over the headless one, uninstalling nothing.

    Pinned to the installed OpenCV version (``opencv-python``'s, else the
    headless or contrib package's), so no other package's requirement moves.
    """
    version = next((v for v in map(package_version, OPENCV_DISTRIBUTIONS) if v), None)
    return "pip install --force-reinstall --no-deps opencv-python" + (f"=={version}" if version else "")


def opencv_gui() -> dict[str, Any] | None:
    """Whether the installed ``cv2`` can open windows, as the labeller needs.

    Returns
    -------
    dict or None
        ``version`` (``cv2.__version__``), ``gui`` (the ``GUI`` line of
        ``cv2.getBuildInformation()``: e.g. ``QT5``, ``COCOA``, or ``NONE``
        for a headless build), ``has_gui`` and ``fix`` (the command from
        :func:`opencv_repair_command` when there is no GUI, else None).
        None if ``cv2`` cannot be imported.
    """
    try:
        import cv2
    except Exception:  # noqa: BLE001 - a half-uninstalled cv2 can fail in many ways
        return None
    try:
        build = str(cv2.getBuildInformation())
    except Exception:  # noqa: BLE001 - broken installs lack the function
        build = ""
    gui = next((line.split(":", 1)[1].strip() for line in build.splitlines() if line.strip().startswith("GUI:")), None)
    has_gui = bool(gui) and gui.upper() != "NONE"
    return {
        "version": getattr(cv2, "__version__", None),
        "gui": gui,
        "has_gui": has_gui,
        "fix": None if has_gui else opencv_repair_command(),
    }


def installed_distributions() -> list[str]:
    """Every installed distribution as ``name==version``, sorted: the equivalent of ``pip freeze``."""
    seen = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            seen.setdefault(name.lower(), f"{name}=={dist.version}")
    return [seen[k] for k in sorted(seen)]


def snapshot(repo: str | Path | None = None, *, all_packages: bool = True) -> dict[str, Any]:
    """Versions and platform facts for a run record.

    Parameters
    ----------
    repo:
        Path inside the git repository to describe; see :func:`git_info`.
    all_packages:
        Also list every installed distribution (``installed``), as the plan's
        run record asks for ``pip freeze`` output.

    Returns
    -------
    dict
        JSON-serialisable: ``created_utc``, ``python`` (version, implementation,
        executable), ``platform`` (system, release, machine, processor,
        ``rosetta``, ``cpu_count``), ``versions`` (``PACKAGES`` -> version or
        None), ``tools`` (ffmpeg/ffprobe version line or None), ``git`` and,
        with ``all_packages``, ``installed``.
    """
    out = {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "mac_ver": platform.mac_ver()[0] or None,
            "rosetta": _running_under_rosetta(),
            "cpu_count": os.cpu_count(),
        },
        "versions": {name: package_version(name) for name in PACKAGES},
        "tools": {name: _tool_version(name) for name in ("ffmpeg", "ffprobe")},
        "git": git_info(repo),
    }
    if all_packages:
        out["installed"] = installed_distributions()
    return out


def _running_under_rosetta() -> bool | None:
    """True for an x86_64 process translated by Rosetta 2; None off macOS."""
    if sys.platform != "darwin":
        return None
    try:
        proc = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    # The key does not exist on Intel Macs, which is "not translated".
    return proc.stdout.strip() == "1"


def _tool_version(name: str) -> str | None:
    exe = shutil.which(name)
    if exe is None:
        return None
    try:
        proc = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    first = (proc.stdout or proc.stderr).splitlines()
    return first[0].strip() if first else None

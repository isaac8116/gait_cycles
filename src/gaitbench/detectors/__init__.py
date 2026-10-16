"""Gait event detectors, looked up by name.

Every detector takes a :class:`~gaitbench.formats.Keypoints` and returns an
events table (``formats.EVENT_COLUMNS``). Names:

* ``"zeni"``: our Zeni et al. (2008) rule (heel and toe relative to the pelvis).
* ``"stenum"``: our Stenum et al. (2021) rule (ankle relative to the pelvis).
* ``"myogait:<method>"``: one of myogait's event methods, run through
  :mod:`gaitbench.myogait_bridge`. Listed only when myogait is installed;
  myogait is imported the first time one of these is requested.

Example
-------
>>> from gaitbench.detectors import list_detectors, run_detector
>>> events = run_detector("zeni", kp, cutoff_hz=5.0)  # doctest: +SKIP
"""

from __future__ import annotations

import functools
import importlib
import importlib.metadata
import importlib.util
import inspect
import warnings
from types import ModuleType
from typing import Any, Callable

import pandas as pd

from .. import __version__
from ..formats import Keypoints, validate_events
from .base import TOOL_NAME, estimate_direction
from .stenum import detect_stenum
from .zeni import detect_zeni

__all__ = [
    "MYOGAIT_PREFIX",
    "detect_stenum",
    "detect_zeni",
    "estimate_direction",
    "get_detector",
    "list_detectors",
    "run_detector",
]

MYOGAIT_PREFIX = "myogait:"

_BUILTIN: dict[str, Callable[..., pd.DataFrame]] = {
    "zeni": detect_zeni,
    "stenum": detect_stenum,
}


def list_detectors() -> list[str]:
    """Names accepted by :func:`get_detector`.

    Returns
    -------
    list of str
        ``"zeni"`` and ``"stenum"``, then ``"myogait:<method>"`` for each
        method of :func:`gaitbench.myogait_bridge.list_methods` if myogait
        can be imported.
    """
    return list(_BUILTIN) + [MYOGAIT_PREFIX + method for method in _myogait_methods()]


def get_detector(name: str, **params: Any) -> Callable[[Keypoints], pd.DataFrame]:
    """Look up a detector and bind its parameters.

    Parameters
    ----------
    name:
        One of :func:`list_detectors`.
    **params:
        Keyword parameters for the detector, e.g. ``cutoff_hz=5.0``. For our
        own detectors they are checked here, so a typo fails immediately
        rather than on the first clip.

    Returns
    -------
    callable
        ``detect(kp) -> events``, giving the same table as :func:`run_detector`.

    Raises
    ------
    KeyError
        If ``name`` is not a known detector; the message lists the valid names.
    ImportError
        If ``name`` is a ``myogait:`` detector and myogait is not installed.
    TypeError
        If ``params`` contains a parameter the detector does not take.
    """
    if name in _BUILTIN:
        fn = _BUILTIN[name]
        _check_params(name, fn, params)
        detect: Callable[[Keypoints], pd.DataFrame] = functools.partial(fn, **params)
        tool, tool_version = TOOL_NAME, __version__
    elif name.startswith(MYOGAIT_PREFIX):
        bridge = _myogait_bridge()
        method = name[len(MYOGAIT_PREFIX):]
        if method not in bridge.list_methods():
            raise KeyError(_unknown_message(name))
        detect = functools.partial(bridge.detect_events_myogait, method=method, **params)
        tool, tool_version = "myogait", _package_version("myogait")
    else:
        raise KeyError(_unknown_message(name))

    def run(kp: Keypoints) -> pd.DataFrame:
        return _finish(detect(kp), kp, name, tool, tool_version)

    run.__name__ = f"detect_{name}"
    run.__doc__ = f"Run detector {name!r} with {params or 'default parameters'}."
    return run


def run_detector(name: str, kp: Keypoints, **params: Any) -> pd.DataFrame:
    """Run one detector on one clip.

    Parameters
    ----------
    name:
        One of :func:`list_detectors`.
    kp:
        Keypoints of one clip.
    **params:
        Detector parameters, see :func:`get_detector`.

    Returns
    -------
    pandas.DataFrame
        Validated events table with ``clip_id = kp.clip_id``,
        ``backend = kp.meta.get("backend", "")``, ``detector = name`` and
        ``tool``/``tool_version`` filled in.
    """
    return get_detector(name, **params)(kp)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _finish(events: pd.DataFrame, kp: Keypoints, name: str, tool: str, tool_version: str) -> pd.DataFrame:
    """Stamp the identifying columns so every detector's output looks the same."""
    out = events.copy()
    out["clip_id"] = kp.clip_id
    out["backend"] = str(kp.meta.get("backend", ""))
    out["detector"] = name
    # Keep a tool and version the bridge already set; fill them only where blank.
    for col, value in (("tool", tool), ("tool_version", tool_version)):
        current = out[col].fillna("").astype(str) if col in out.columns else pd.Series("", index=out.index)
        out[col] = current.where(current != "", value)
    return validate_events(out)


def _check_params(name: str, fn: Callable[..., Any], params: dict[str, Any]) -> None:
    accepted = [p for p in inspect.signature(fn).parameters if p != "kp"]
    unknown = sorted(set(params) - set(accepted))
    if unknown:
        raise TypeError(f"Detector {name!r} does not take {unknown}; it takes {accepted}")


def _unknown_message(name: str) -> str:
    return f"Unknown detector {name!r}. Valid names: {', '.join(list_detectors())}"


def _myogait_available() -> bool:
    """True if the myogait package can be found (it is not imported here)."""
    return importlib.util.find_spec("myogait") is not None


def _myogait_bridge() -> ModuleType:
    """Import :mod:`gaitbench.myogait_bridge`, with a clear error if myogait is missing."""
    if not _myogait_available():
        raise ImportError("myogait detectors need myogait and gaitkit: pip install 'gaitbench[myogait]'")
    return importlib.import_module("gaitbench.myogait_bridge")


def _myogait_methods() -> list[str]:
    """myogait's event methods, or [] if myogait is missing or fails to load.

    Listing detectors must never crash (the CLI calls it to print help), so
    an installed-but-broken myogait only produces a warning.
    """
    if not _myogait_available():
        return []
    try:
        return list(_myogait_bridge().list_methods())
    except Exception as exc:
        warnings.warn(f"myogait is installed but its detectors are unavailable: {exc}", stacklevel=3)
        return []


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return ""

"""Hand labelling of heel strikes and toe offs: the ground truth for our own clips.

:class:`LabelSession` is the pure labelling logic (no GUI). :func:`run_labeller`
opens the OpenCV frame stepper; OpenCV is imported only when it is called.
Labelling is blind: no detector output is ever shown to the rater.
"""

from .app import HELP_LINES, KEY_BINDINGS, run_labeller
from .state import REPLACE_WINDOW_FRAMES, Edit, LabelledEvent, LabelSession

__all__ = [
    "HELP_LINES",
    "KEY_BINDINGS",
    "REPLACE_WINDOW_FRAMES",
    "Edit",
    "LabelSession",
    "LabelledEvent",
    "run_labeller",
]

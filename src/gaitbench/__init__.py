"""gaitbench: run every candidate gait tool on the same clips and score it the same way.

Stages pass files, not Python objects (see docs/harness.md):

    video -> pose stage -> keypoints .npz -> event stage -> events .csv -> scoring -> results
"""

__version__ = "0.1.0"

"""Readers that turn other tools' keypoint files into harness :class:`~gaitbench.formats.Keypoints`.

Only OpenPose JSON is supported so far (:func:`load_openpose_dir`). It is the
format of the GPJATK sample in Stenum's ``Multiple-Perspectives`` branch (data
ladder level 0) and what Pose2Sim reads, so other OpenPose-style exporters can
reuse it.
"""

from __future__ import annotations

from .openpose import BODY_25, COCO_18, load_openpose_dir, read_openpose_frame

__all__ = ["BODY_25", "COCO_18", "load_openpose_dir", "read_openpose_frame"]

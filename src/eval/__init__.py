"""Evaluation: detection mAP, ball center error, optional MOT metrics via TrackEval."""

from src.eval.detection import DetectionEvalResult, evaluate_detection
from src.eval.loaders import (
    align_frames_by_number,
    frames_to_boxes_by_frame,
    ground_truth_to_frames,
    load_ground_truth_json,
    load_prediction_json,
    predictions_to_frames,
)
from src.eval.runner import build_report
from src.eval.schema import CLASS_NAMES, Box, FrameAnnotations
from src.eval.tracking import TrackingEvalResult, evaluate_tracking_trackeval, has_gt_track_ids, has_pred_track_ids

__all__ = [
    "CLASS_NAMES",
    "Box",
    "DetectionEvalResult",
    "FrameAnnotations",
    "TrackingEvalResult",
    "build_report",
    "align_frames_by_number",
    "evaluate_detection",
    "evaluate_tracking_trackeval",
    "frames_to_boxes_by_frame",
    "ground_truth_to_frames",
    "has_gt_track_ids",
    "has_pred_track_ids",
    "load_ground_truth_json",
    "load_prediction_json",
    "predictions_to_frames",
]

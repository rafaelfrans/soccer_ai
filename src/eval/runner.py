"""Orchestrate loading GT/preds and building the eval report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.eval.detection import evaluate_detection
from src.eval.loaders import (
    align_frames_by_number,
    frames_to_boxes_by_frame,
    ground_truth_to_frames,
    load_ground_truth_json,
    load_prediction_json,
    predictions_to_frames,
)
from src.eval.schema import CLASS_NAMES
from src.eval.tracking import evaluate_tracking_trackeval, tracking_result_to_dict


def build_report(
    *,
    gt_path: str,
    pred_path: str,
    mode: str,
    run_tracking: bool,
) -> dict[str, Any]:
    gt = load_ground_truth_json(gt_path)
    pred = load_prediction_json(pred_path)

    pred_frames = predictions_to_frames(pred, mode=mode)
    gt_frames = ground_truth_to_frames(gt)

    pred_aligned, gt_aligned = align_frames_by_number(pred_frames, gt_frames)
    pred_boxes = frames_to_boxes_by_frame(pred_aligned)
    gt_boxes = frames_to_boxes_by_frame(gt_aligned)

    det = evaluate_detection(pred_boxes, gt_boxes)

    report: dict[str, Any] = {
        "ground_truth": str(gt_path),
        "predictions": str(pred_path),
        "eval_mode": mode,
        "frames_evaluated": len(pred_boxes),
        "detection": {
            "mAP_0.50": det.map_50,
            "mAP_0.50:0.95": det.map_50_95,
            "ap_per_class": {CLASS_NAMES[i]: det.ap_per_class[CLASS_NAMES[i]] for i in range(len(CLASS_NAMES))},
            "ap50_per_class": {CLASS_NAMES[i]: det.ap50_per_class[CLASS_NAMES[i]] for i in range(len(CLASS_NAMES))},
            "ball_center_distance_mean_px": det.ball_center_distance_mean,
            "ball_center_distance_count": len(det.ball_center_distances),
        },
    }

    if run_tracking:
        vi = pred.get("video_info") or gt.get("video_info") or {}
        nf = int(vi.get("total_frames", max(pred_boxes.keys(), default=-1) + 1))
        if nf <= 0:
            nf = max(pred_boxes.keys(), default=-1) + 1
        w = int(vi.get("width", 1920))
        h = int(vi.get("height", 1080))
        fps = float(vi.get("fps", 30.0))
        tr = evaluate_tracking_trackeval(
            pred_boxes,
            gt_boxes,
            num_frames=nf,
            width=w,
            height=h,
            fps=fps,
            seq_name=Path(gt_path).stem,
        )
        report["tracking"] = tracking_result_to_dict(tr)

    return report

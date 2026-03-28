"""Smoke tests for eval metrics."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.eval.detection import _ap_at_threshold, evaluate_detection
from src.eval.schema import Box, boxes_from_pred_frame
from src.eval.tracking import evaluate_tracking_trackeval


def test_model_only_empty_objects_branch():
    """Empty model_only objects must not fall through to tracked_objects."""
    fr = {
        "frame_number": 0,
        "objects": [],
        "tracked_objects": [{"bbox": [0, 0, 10, 10], "class_id": 0, "confidence": 0.9}],
    }
    boxes = boxes_from_pred_frame(fr, mode="model_only", video_width=100, video_height=100)
    assert len(boxes) == 0


def test_full_json_null_lists_coerced():
    fr = {"frame_number": 0, "tracked_objects": None, "ball": None}
    boxes = boxes_from_pred_frame(fr, mode="full", video_width=100, video_height=100)
    assert boxes == []


def test_ap_at_threshold_pred_frame_without_gt_entry():
    pred = {0: [Box(0, 0, 10, 10, 0, confidence=0.9)]}
    gt = {1: [Box(0, 0, 10, 10, 0)]}
    assert _ap_at_threshold(pred, gt, class_id=0, iou_thr=0.5) == 0.0


def test_detection_perfect_match():
    pred = {
        0: [
            Box(10, 10, 20, 20, 0, confidence=0.99),
            Box(100, 100, 150, 150, 1, confidence=0.9, track_id=1),
        ]
    }
    gt = {
        0: [
            Box(10, 10, 20, 20, 0),
            Box(100, 100, 150, 150, 1, track_id=1),
        ]
    }
    r = evaluate_detection(pred, gt)
    assert r.ap50_per_class["ball"] == pytest.approx(1.0)
    assert r.ap50_per_class["goalkeeper"] == pytest.approx(1.0)
    assert r.ball_center_distance_mean == pytest.approx(0.0)


def test_eval_cli_json(tmp_path: Path):
    gt = {
        "video_info": {"width": 640, "height": 480, "total_frames": 1},
        "detections": [
            {
                "frame_number": 0,
                "objects": [{"bbox": [0, 0, 10, 10], "class_id": 0}],
            }
        ],
    }
    pred = {
        "video_info": {"width": 640, "height": 480, "total_frames": 1},
        "detections": [
            {
                "frame_number": 0,
                "objects": [{"bbox": [0, 0, 10, 10], "class_id": 0, "confidence": 0.99}],
            }
        ],
    }
    gt_path = tmp_path / "gt.json"
    pr_path = tmp_path / "pred.json"
    gt_path.write_text(json.dumps(gt), encoding="utf-8")
    pr_path.write_text(json.dumps(pred), encoding="utf-8")

    from src.eval.runner import build_report

    rep = build_report(gt_path=str(gt_path), pred_path=str(pr_path), mode="model_only", run_tracking=False)
    assert rep["detection"]["ap50_per_class"]["ball"] == pytest.approx(1.0)
    assert rep["detection"]["mAP_0.50"] == pytest.approx(0.25)  # mean over 4 classes; only ball has GT


@pytest.mark.skipif(importlib.util.find_spec("trackeval") is None, reason="trackeval not installed")
def test_tracking_trackeval_smoke():
    pred = {0: [Box(0, 0, 10, 10, 1, track_id=1, confidence=0.9)]}
    gt = {0: [Box(0, 0, 10, 10, 1, track_id=1)]}
    r = evaluate_tracking_trackeval(pred, gt, num_frames=1, width=100, height=100, fps=30)
    assert r is not None
    assert "MOTA" in r.clear

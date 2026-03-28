"""Load ground-truth and prediction JSON for evaluation."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from src.eval.schema import Box, FrameAnnotations, boxes_from_gt_frame, boxes_from_pred_frame


def _warn_duplicate_frame_numbers(detections: list, source: str) -> None:
    seen: set[int] = set()
    dups: set[int] = set()
    for fr in detections:
        fn = int(fr["frame_number"])
        if fn in seen:
            dups.add(fn)
        seen.add(fn)
    if dups:
        sample = sorted(dups)[:15]
        more = "..." if len(dups) > len(sample) else ""
        warnings.warn(
            f"{source}: duplicate frame_number values {sample}{more}; last occurrence wins.",
            stacklevel=2,
        )


def load_prediction_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_ground_truth_json(path: str | Path) -> dict[str, Any]:
    """Load GT file. Expected shape:

    {
      "video_info": { "width", "height", "total_frames" (optional) },
      "detections": [
        { "frame_number": int, "objects": [ { "bbox": [x1,y1,x2,y2], "class_id": 0-3, "track_id": optional } ] }
      ]
    }
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def predictions_to_frames(pred: dict[str, Any], mode: str) -> list[FrameAnnotations]:
    vi = pred.get("video_info", {})
    w, h = vi.get("width"), vi.get("height")
    dets = pred.get("detections") or []
    if isinstance(dets, list):
        _warn_duplicate_frame_numbers(dets, "predictions JSON")
    frames: list[FrameAnnotations] = []
    for fr in dets:
        fn = int(fr["frame_number"])
        boxes = boxes_from_pred_frame(fr, mode=mode, video_width=w, video_height=h)
        frames.append(FrameAnnotations(frame_number=fn, boxes=boxes))
    return frames


def ground_truth_to_frames(gt: dict[str, Any]) -> list[FrameAnnotations]:
    dets = gt.get("detections") or []
    if isinstance(dets, list):
        _warn_duplicate_frame_numbers(dets, "ground-truth JSON")
    frames: list[FrameAnnotations] = []
    for fr in dets:
        fn = int(fr["frame_number"])
        boxes = boxes_from_gt_frame(fr)
        frames.append(FrameAnnotations(frame_number=fn, boxes=boxes))
    return frames


def align_frames_by_number(
    pred_frames: list[FrameAnnotations] | dict[int, FrameAnnotations],
    gt_frames: list[FrameAnnotations] | dict[int, FrameAnnotations],
) -> tuple[dict[int, FrameAnnotations], dict[int, FrameAnnotations]]:
    """Convert lists to dicts keyed by frame_number and keep intersection of frames."""
    pred_d = {f.frame_number: f for f in pred_frames} if isinstance(pred_frames, list) else pred_frames
    gt_d = {f.frame_number: f for f in gt_frames} if isinstance(gt_frames, list) else gt_frames
    common = sorted(set(pred_d) & set(gt_d))
    return {i: pred_d[i] for i in common}, {i: gt_d[i] for i in common}


def frames_to_boxes_by_frame(frames: dict[int, FrameAnnotations]) -> dict[int, list[Box]]:
    return {fn: f.boxes for fn, f in frames.items()}

"""Per-frame IoU matching and mAP / ball center error."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.eval.schema import CLASS_NAMES, Box


def bbox_iou_xyxy(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.x1, a.y1, a.x2, a.y2
    bx1, by1, bx2, by2 = b.x1, b.y1, b.x2, b.y2
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def ball_center_distance(a: Box, b: Box) -> float:
    acx = (a.x1 + a.x2) / 2.0
    acy = (a.y1 + a.y2) / 2.0
    bcx = (b.x1 + b.x2) / 2.0
    bcy = (b.y1 + b.y2) / 2.0
    return float(np.hypot(acx - bcx, acy - bcy))


def _iou_matrix(pred: list[Box], gt: list[Box]) -> np.ndarray:
    if not pred or not gt:
        return np.zeros((len(pred), len(gt)))
    out = np.zeros((len(pred), len(gt)))
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            out[i, j] = bbox_iou_xyxy(p, g)
    return out


def match_frame(
    pred: list[Box],
    gt: list[Box],
    class_id: int,
    iou_thr: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy IoU matching for one class in one frame. Returns matched (pred_idx, gt_idx), unmatched pred, unmatched gt."""
    p_idx = [i for i, b in enumerate(pred) if b.class_id == class_id]
    g_idx = [j for j, b in enumerate(gt) if b.class_id == class_id]
    if not p_idx or not g_idx:
        return [], [i for i in p_idx], [j for j in g_idx]
    sub_pred = [pred[i] for i in p_idx]
    sub_gt = [gt[j] for j in g_idx]
    ious = _iou_matrix(sub_pred, sub_gt)
    matched: list[tuple[int, int]] = []
    used_p = set()
    used_g = set()
    # Sort candidate pairs by IoU descending
    pairs = [(ious[i, j], i, j) for i in range(len(p_idx)) for j in range(len(g_idx))]
    pairs.sort(reverse=True)
    for iou, i, j in pairs:
        if iou < iou_thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matched.append((p_idx[i], g_idx[j]))
    unmatched_p = [p_idx[i] for i in range(len(p_idx)) if i not in used_p]
    unmatched_g = [g_idx[j] for j in range(len(g_idx)) if j not in used_g]
    return matched, unmatched_p, unmatched_g


def voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """VOC-style AP: area under precision-recall curve (trapezoid)."""
    if len(recall) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


@dataclass
class DetectionEvalResult:
    map_50: float
    map_50_95: float
    ap_per_class: dict[str, float]
    ap50_per_class: dict[str, float]
    ball_center_distance_mean: float | None
    ball_center_distances: list[float] = field(default_factory=list)
    per_frame_counts: dict[str, Any] = field(default_factory=dict)


def evaluate_detection(
    pred_by_frame: dict[int, list[Box]],
    gt_by_frame: dict[int, list[Box]],
    iou_thresholds: list[float] | None = None,
) -> DetectionEvalResult:
    """Compute mAP at IoU 0.5 and mean mAP over thresholds (COCO-style for 0.5:0.95 step 0.05).

    Assumes pred_by_frame and gt_by_frame have the same keys (aligned frames).
    """
    if iou_thresholds is None:
        iou_thresholds = [x / 100.0 for x in range(50, 100, 5)]  # 0.5 to 0.95
    iou50 = 0.5

    ap50_per_class: dict[str, float] = {}
    ap_per_class: dict[str, float] = {}

    ball_dists: list[float] = []

    for class_id, name in enumerate(CLASS_NAMES):
        # AP at 0.5
        ap50 = _ap_at_threshold(pred_by_frame, gt_by_frame, class_id, iou50)
        ap50_per_class[name] = ap50

        # mAP over thresholds
        aps = [_ap_at_threshold(pred_by_frame, gt_by_frame, class_id, t) for t in iou_thresholds]
        ap_per_class[name] = float(np.mean(aps)) if aps else 0.0

        # Ball center distances on matched pairs at IoU 0.5
        if class_id == 0:
            for fn in sorted(pred_by_frame.keys()):
                p = pred_by_frame[fn]
                g = gt_by_frame.get(fn, [])
                matched, _, _ = match_frame(p, g, class_id=0, iou_thr=iou50)
                for pi, gi in matched:
                    ball_dists.append(ball_center_distance(p[pi], g[gi]))

    map50 = float(np.mean(list(ap50_per_class.values()))) if ap50_per_class else 0.0
    map_mean = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0
    ball_mean = float(np.mean(ball_dists)) if ball_dists else None

    return DetectionEvalResult(
        map_50=map50,
        map_50_95=map_mean,
        ap_per_class=ap_per_class,
        ap50_per_class=ap50_per_class,
        ball_center_distance_mean=ball_mean,
        ball_center_distances=ball_dists,
    )


def _ap_at_threshold(
    pred_by_frame: dict[int, list[Box]],
    gt_by_frame: dict[int, list[Box]],
    class_id: int,
    iou_thr: float,
) -> float:
    """Single-class AP: pool predictions globally, sort by score, greedy match to GT."""
    # Collect all GT count
    total_gt = sum(1 for fn in gt_by_frame for b in gt_by_frame[fn] if b.class_id == class_id)
    if total_gt == 0:
        # No GT: AP undefined; any prediction would be FP — report 0
        return 0.0

    # List of predictions (score, frame, index_in_frame)
    preds: list[tuple[float, int, int]] = []
    for fn in pred_by_frame:
        for bi, b in enumerate(pred_by_frame[fn]):
            if b.class_id == class_id:
                sc = b.confidence if b.confidence is not None else 0.0
                preds.append((float(sc), fn, bi))
    preds.sort(key=lambda x: -x[0])

    if not preds:
        return 0.0

    # Track which GT boxes are matched (frame -> set of gt indices)
    all_frames = set(pred_by_frame) | set(gt_by_frame)
    gt_matched: dict[int, set[int]] = {fn: set() for fn in all_frames}

    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))

    for i, (_score, fn, pidx) in enumerate(preds):
        pbox = pred_by_frame[fn][pidx]
        gt_list = gt_by_frame.get(fn, [])
        best_j = -1
        best_iou = iou_thr
        for j, g in enumerate(gt_list):
            if g.class_id != class_id:
                continue
            if j in gt_matched[fn]:
                continue
            iou = bbox_iou_xyxy(pbox, g)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            tp[i] = 1
            gt_matched[fn].add(best_j)
        else:
            fp[i] = 1

    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    recall = tp_c / total_gt
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    return voc_ap(recall, precision)


def hungarian_match_iou(pred: list[Box], gt: list[Box], class_id: int, iou_thr: float) -> list[tuple[int, int]]:
    """Optional: optimal matching for one frame/class (for diagnostics)."""
    p_idx = [i for i, b in enumerate(pred) if b.class_id == class_id]
    g_idx = [j for j, b in enumerate(gt) if b.class_id == class_id]
    if not p_idx or not g_idx:
        return []
    ious = _iou_matrix([pred[i] for i in p_idx], [gt[j] for j in g_idx])
    cost = 1.0 - ious
    row_ind, col_ind = linear_sum_assignment(cost)
    out = []
    for r, c in zip(row_ind, col_ind):
        if ious[r, c] >= iou_thr:
            out.append((p_idx[r], g_idx[c]))
    return out

"""MOT export and TrackEval integration for person-only tracking (classes 1–3)."""

from __future__ import annotations

import configparser
import contextlib
import io
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.schema import Box

# MOTChallenge uses 1-based frame indices in files; class 1 = pedestrian.
MOT_PERSON_CLASS = 1


def _xyxy_to_xywh(box: Box) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return (x1, y1, w, h)


def _person_boxes(boxes: list[Box]) -> list[Box]:
    return [b for b in boxes if b.class_id in (1, 2, 3)]


def has_gt_track_ids(gt_by_frame: dict[int, list[Box]]) -> bool:
    for boxes in gt_by_frame.values():
        for b in _person_boxes(boxes):
            if b.track_id is not None:
                return True
    return False


def has_pred_track_ids(pred_by_frame: dict[int, list[Box]]) -> bool:
    for boxes in pred_by_frame.values():
        for b in _person_boxes(boxes):
            if b.track_id is not None:
                return True
    return False


def write_mot_gt_txt(path: Path, gt_by_frame: dict[int, list[Box]], num_frames: int) -> None:
    """Write MOT gt.txt (1-based frames). Only persons with track_id are written."""
    lines: list[str] = []
    for fn in range(num_frames):
        frame_1 = fn + 1
        boxes = gt_by_frame.get(fn, [])
        for b in _person_boxes(boxes):
            if b.track_id is None:
                continue
            left, top, w, h = _xyxy_to_xywh(b)
            # frame, id, bb_left, bb_top, bb_w, bb_h, conf, class, visibility
            lines.append(
                f"{frame_1},{int(b.track_id)},{left:.2f},{top:.2f},{w:.2f},{h:.2f},1,{MOT_PERSON_CLASS},1\n"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def write_mot_pred_txt(path: Path, pred_by_frame: dict[int, list[Box]], num_frames: int) -> None:
    """Write tracker prediction file (1-based frames)."""
    lines: list[str] = []
    for fn in range(num_frames):
        frame_1 = fn + 1
        boxes = pred_by_frame.get(fn, [])
        for b in _person_boxes(boxes):
            tid = b.track_id if b.track_id is not None else -1
            left, top, w, h = _xyxy_to_xywh(b)
            conf = b.confidence if b.confidence is not None else 1.0
            lines.append(
                f"{frame_1},{int(tid)},{left:.2f},{top:.2f},{w:.2f},{h:.2f},{conf:.4f},{MOT_PERSON_CLASS},-1\n"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def write_seqinfo_ini(path: Path, seq_name: str, num_frames: int, width: int, height: int, fps: float) -> None:
    cfg = configparser.ConfigParser()
    cfg["Sequence"] = {
        "name": seq_name,
        "imDir": "img1",
        "frameRate": str(int(round(fps))),
        "seqLength": str(num_frames),
        "imWidth": str(width),
        "imHeight": str(height),
        "imExt": "jpg",
    }
    with open(path, "w", encoding="utf-8") as f:
        cfg.write(f)


@dataclass
class TrackingEvalResult:
    hota: dict[str, float] = field(default_factory=dict)
    clear: dict[str, float] = field(default_factory=dict)
    identity: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def evaluate_tracking_trackeval(
    pred_by_frame: dict[int, list[Box]],
    gt_by_frame: dict[int, list[Box]],
    *,
    num_frames: int,
    width: int,
    height: int,
    fps: float,
    seq_name: str = "eval_seq",
    keep_temp: bool = False,
) -> TrackingEvalResult | None:
    """Run HOTA, CLEAR (MOTA), Identity (IDF1) via trackeval MOTChallenge adapter.

    Returns None if trackeval is not installed, GT/pred lack person track IDs, or TrackEval raises.
    """
    if not has_gt_track_ids(gt_by_frame) or not has_pred_track_ids(pred_by_frame):
        return None

    try:
        from trackeval.datasets import MotChallenge2DBox
        from trackeval.eval import Evaluator
        from trackeval.metrics import CLEAR, HOTA, Identity
    except ImportError:
        return None

    tmp = tempfile.mkdtemp(prefix="soccer_ai_mot_")
    try:
        gt_root = Path(tmp) / "gt"
        trk_root = Path(tmp) / "trackers"
        seq_dir = gt_root / seq_name
        (seq_dir / "gt").mkdir(parents=True)
        write_seqinfo_ini(seq_dir / "seqinfo.ini", seq_name, num_frames, width, height, fps)
        write_mot_gt_txt(seq_dir / "gt" / "gt.txt", gt_by_frame, num_frames)

        tracker_name = "pred"
        pred_dir = trk_root / tracker_name / "data"
        pred_dir.mkdir(parents=True)
        write_mot_pred_txt(pred_dir / f"{seq_name}.txt", pred_by_frame, num_frames)

        dataset_config = {
            "GT_FOLDER": str(gt_root),
            "TRACKERS_FOLDER": str(trk_root),
            "TRACKERS_TO_EVAL": [tracker_name],
            "BENCHMARK": "MOT17",
            "SPLIT_TO_EVAL": "train",
            "SKIP_SPLIT_FOL": True,
            "SEQ_INFO": {seq_name: num_frames},
            "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
            "OUTPUT_FOLDER": str(Path(tmp) / "output"),
            "DO_PREPROC": True,
            "TRACKER_SUB_FOLDER": "data",
            "PRINT_CONFIG": False,
            "CLASSES_TO_EVAL": ["pedestrian"],
        }
        dataset = MotChallenge2DBox(dataset_config)

        eval_config = {
            "USE_PARALLEL": False,
            "PRINT_RESULTS": False,
            "PRINT_CONFIG": False,
            "PLOT_CURVES": False,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
        }
        evaluator = Evaluator(eval_config)
        metrics_list = [HOTA(), CLEAR(), Identity()]
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                output_res, _msg = evaluator.evaluate([dataset], metrics_list)
        except Exception:
            return None

        res = TrackingEvalResult()
        res.raw = output_res
        ds = output_res.get("MotChallenge2DBox") or next(iter(output_res.values()), {})
        trk_block = ds.get(tracker_name, {})
        combined = trk_block.get("COMBINED_SEQ", {})
        ped = combined.get("pedestrian", {})

        def _flat_metric(m: dict[str, Any]) -> dict[str, float]:
            flat: dict[str, float] = {}
            for k, v in m.items():
                if isinstance(v, np.ndarray):
                    flat[k] = float(np.mean(v)) if v.size else 0.0
                elif isinstance(v, (int, float)):
                    flat[k] = float(v)
            return flat

        for key, dest in (("HOTA", res.hota), ("CLEAR", res.clear), ("Identity", res.identity)):
            if key in ped and isinstance(ped[key], dict):
                dest.update(_flat_metric(ped[key]))
        return res
    finally:
        if not keep_temp:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)


def tracking_result_to_dict(r: TrackingEvalResult | None) -> dict[str, Any]:
    if r is None:
        return {"skipped": True, "reason": "missing_track_ids_or_trackeval"}
    return {
        "hota": r.hota,
        "clear": r.clear,
        "identity": r.identity,
    }

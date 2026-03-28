"""Canonical class IDs and prediction normalization for evaluation.

Class IDs (AGENTS.md):
  0 — ball
  1 — goalkeeper
  2 — player
  3 — referee

Predictions from *_detections.json use shifted IDs for tracked_objects (0,1,2 → gk, player, ref)
and ball in a separate list with YOLO class_id 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Ball boxes are padded by this many pixels before export in the full pipeline (see video_processor.py).
BALL_PAD_PX = 10

CLASS_NAMES: tuple[str, ...] = ("ball", "goalkeeper", "player", "referee")


@dataclass
class Box:
    """Single detection in pixel xyxy (inclusive)."""

    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    confidence: float | None = None
    track_id: int | None = None

    def to_xyxy_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class FrameAnnotations:
    """All boxes for one frame (canonical class_ids 0–3)."""

    frame_number: int
    boxes: list[Box] = field(default_factory=list)


def pred_tracked_class_to_canonical(shifted_id: int) -> int:
    """Map JSON tracked_objects class_id (0=gk, 1=player, 2=ref) to canonical 1–3."""
    s = int(shifted_id)
    s = max(0, min(2, s))
    return s + 1


def unpad_ball_xyxy(xyxy: list[float] | tuple[float, ...], pad_px: int = BALL_PAD_PX, width: int | None = None, height: int | None = None) -> tuple[float, float, float, float]:
    """Remove symmetric padding from ball box; clip to image bounds if width/height given."""
    x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
    x1 += pad_px
    y1 += pad_px
    x2 -= pad_px
    y2 -= pad_px
    if width is not None and height is not None and width > 0 and height > 0:
        x1 = max(0.0, min(x1, float(width - 1)))
        y1 = max(0.0, min(y1, float(height - 1)))
        x2 = max(0.0, min(x2, float(width - 1)))
        y2 = max(0.0, min(y2, float(height - 1)))
    # Enforce valid ordering after unpad (tiny padded boxes can invert)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return (x1, y1, x2, y2)


def _json_list(frame: dict, key: str) -> list:
    """Coerce missing or JSON-null list fields to []."""
    v = frame.get(key)
    if v is None:
        return []
    return v if isinstance(v, list) else []


def boxes_from_pred_frame(
    frame: dict,
    *,
    mode: str,
    video_width: int | None,
    video_height: int | None,
) -> list[Box]:
    """Convert one prediction JSON frame dict to canonical Box list.

    If the frame contains \"objects\" (canonical class_id 0–3), use that (model_only export).
    Otherwise use tracked_objects + ball (full pipeline export).

    mode:
      - \"full\": unpad ball boxes for fair IoU vs tight GT
      - \"model_only\": no ball unpad (objects are already unpadded)
    """
    # Use key presence so model_only frames with objects: [] do not fall through to full layout.
    if "objects" in frame:
        out: list[Box] = []
        for o in _json_list(frame, "objects"):
            bbox = o["bbox"]
            conf = o.get("confidence")
            out.append(
                Box(
                    x1=float(bbox[0]),
                    y1=float(bbox[1]),
                    x2=float(bbox[2]),
                    y2=float(bbox[3]),
                    class_id=int(o["class_id"]),
                    confidence=float(conf) if conf is not None else None,
                    track_id=int(o["track_id"]) if o.get("track_id") is not None else None,
                )
            )
        return out

    out: list[Box] = []
    for o in _json_list(frame, "tracked_objects"):
        bbox = o["bbox"]
        cid = pred_tracked_class_to_canonical(int(o["class_id"]))
        conf = o.get("confidence")
        tid = o.get("tracker_id")
        out.append(
            Box(
                x1=float(bbox[0]),
                y1=float(bbox[1]),
                x2=float(bbox[2]),
                y2=float(bbox[3]),
                class_id=cid,
                confidence=float(conf) if conf is not None else None,
                track_id=int(tid) if tid is not None else None,
            )
        )
    for o in _json_list(frame, "ball"):
        bbox = o["bbox"]
        if mode == "full":
            x1, y1, x2, y2 = unpad_ball_xyxy(bbox, pad_px=BALL_PAD_PX, width=video_width, height=video_height)
        else:
            x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        conf = o.get("confidence")
        out.append(
            Box(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                class_id=0,
                confidence=float(conf) if conf is not None else None,
                track_id=None,
            )
        )
    return out


def boxes_from_gt_frame(frame: dict) -> list[Box]:
    """Load canonical boxes from ground-truth JSON frame (see loaders)."""
    out: list[Box] = []
    for o in _json_list(frame, "objects"):
        bbox = o["bbox"]
        tid = o.get("track_id")
        out.append(
            Box(
                x1=float(bbox[0]),
                y1=float(bbox[1]),
                x2=float(bbox[2]),
                y2=float(bbox[3]),
                class_id=int(o["class_id"]),
                confidence=None,
                track_id=int(tid) if tid is not None else None,
            )
        )
    return out

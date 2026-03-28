"""Convert SoccerNet-v3 Labels-v3.json to soccer_ai eval ground-truth JSON.

SoccerNet-v3 schema follows the official dataloader (see SoccerNet-v3 repo).
Class strings map to soccer_ai 0–3: ball, goalkeeper, player, referee.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

# SoccerNet FRAME_CLASS_DICTIONARY indices (subset we care about)
_SOCCER_AI_BY_SN_CLASS: dict[str, int] = {
    "Ball": 0,
    "Goalkeeper team left": 1,
    "Goalkeeper team right": 1,
    "Goalkeeper team unknown": 1,
    "Player team left": 2,
    "Player team right": 2,
    "Player team unknown 1": 2,
    "Player team unknown 2": 2,
    "Main referee": 3,
    "Side referee": 3,
}


def sn_class_to_soccer_ai(class_name: str | None) -> int | None:
    if class_name is None:
        return None
    return _SOCCER_AI_BY_SN_CLASS.get(class_name)


def _bbox_to_xyxy(points: dict[str, Any]) -> tuple[float, float, float, float]:
    x1 = float(points["x1"])
    y1 = float(points["y1"])
    x2 = float(points["x2"])
    y2 = float(points["y2"])
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def iter_labels_v3_frames(annotations: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (image_filename, entry) in evaluation order (same as SN-v3 dataloader).

    For each action: action frame first, then linked replays in order.
    """
    meta = annotations.get("GameMetadata") or {}
    actions_root = annotations.get("actions") or {}
    replays_root = annotations.get("replays") or {}

    for action_name in meta.get("list_actions") or []:
        if action_name not in actions_root:
            continue
        block = actions_root[action_name]
        linked = list(block.get("linked_replays") or [])
        img_list = [action_name] + linked
        for i, img in enumerate(img_list):
            entry = actions_root.get(img) if i == 0 else replays_root.get(img)
            if entry is None:
                continue
            yield img, entry


def labels_v3_to_eval_gt(
    annotations: dict[str, Any],
    *,
    source_note: str = "SoccerNet-v3",
) -> dict[str, Any]:
    """Build eval GT dict with video_info + detections[].objects per frame."""
    frames: list[dict[str, Any]] = []
    width: int | None = None
    height: int | None = None
    frame_number = 0

    for _img_name, entry in iter_labels_v3_frames(annotations):
        im = entry.get("imageMetadata") or {}
        if width is None and im.get("width"):
            width = int(im["width"])
        if height is None and im.get("height"):
            height = int(im["height"])

        objects: list[dict[str, Any]] = []
        for bbox in entry.get("bboxes") or []:
            cls_name = bbox.get("class")
            cid = sn_class_to_soccer_ai(cls_name)
            if cid is None:
                continue
            pts = bbox.get("points") or {}
            if not all(k in pts for k in ("x1", "y1", "x2", "y2")):
                continue
            x1, y1, x2, y2 = _bbox_to_xyxy(pts)
            obj: dict[str, Any] = {
                "bbox": [x1, y1, x2, y2],
                "class_id": cid,
            }
            bid = bbox.get("ID")
            if bid is not None and str(bid).isnumeric():
                obj["track_id"] = int(bid)
            objects.append(obj)

        frames.append({"frame_number": frame_number, "objects": objects})
        frame_number += 1

    return {
        "source": source_note,
        "video_info": {
            "width": width or 0,
            "height": height or 0,
            "total_frames": len(frames),
        },
        "detections": frames,
    }

"""SoccerNet-v3 label conversion tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.soccer_net_v3 import (
    iter_labels_v3_frames,
    labels_v3_to_eval_gt,
    sn_class_to_soccer_ai,
)


def test_sn_class_mapping():
    assert sn_class_to_soccer_ai("Ball") == 0
    assert sn_class_to_soccer_ai("Goalkeeper team left") == 1
    assert sn_class_to_soccer_ai("Player team right") == 2
    assert sn_class_to_soccer_ai("Main referee") == 3
    assert sn_class_to_soccer_ai("Goal left post left ") is None


def test_sample_fixture_roundtrip():
    p = Path(__file__).resolve().parent / "fixtures" / "soccer_net" / "Labels-v3-sample.json"
    annotations = json.loads(p.read_text(encoding="utf-8"))
    names = [n for n, _ in iter_labels_v3_frames(annotations)]
    assert names == ["0.png", "0_1.png"]
    gt = labels_v3_to_eval_gt(annotations)
    assert gt["video_info"]["total_frames"] == 2
    assert gt["detections"][0]["objects"][0]["class_id"] == 0
    assert gt["detections"][1]["objects"][0]["class_id"] == 2
    assert gt["detections"][1]["objects"][0]["track_id"] == 10


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"GameMetadata": {}, "actions": {}, "replays": {}},
    ],
)
def test_empty_labels(bad: dict):
    gt = labels_v3_to_eval_gt(bad)
    assert gt["detections"] == []
    assert gt["video_info"]["total_frames"] == 0

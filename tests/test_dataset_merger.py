"""Tests for dataset merger utilities."""

import pytest

from src.data.dataset_merger import _extract_ball_annotations


@pytest.fixture
def label_file(tmp_path):
    """Create a temporary label file with mixed annotations."""
    path = tmp_path / "label.txt"
    path.write_text("0 0.5 0.5 0.1 0.1\n1 0.3 0.3 0.2 0.2\n2 0.7 0.7 0.05 0.05\n")
    return path


class TestExtractBallAnnotations:
    def test_maps_all_annotations_to_ball_class(self, label_file):
        annotations = _extract_ball_annotations(label_file, ball_class_id=0)

        assert len(annotations) == 3
        for line in annotations:
            assert line.startswith("0 ")

    def test_custom_ball_class_id(self, label_file):
        annotations = _extract_ball_annotations(label_file, ball_class_id=5)

        assert len(annotations) == 3
        for line in annotations:
            assert line.startswith("5 ")

    def test_preserves_coordinates(self, label_file):
        annotations = _extract_ball_annotations(label_file, ball_class_id=0)

        parts = annotations[0].strip().split()
        assert parts[1:] == ["0.5", "0.5", "0.1", "0.1"]

    def test_nonexistent_file(self, tmp_path):
        result = _extract_ball_annotations(tmp_path / "missing.txt", ball_class_id=0)
        assert result == []

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.txt"
        path.write_text("")
        result = _extract_ball_annotations(path, ball_class_id=0)
        assert result == []

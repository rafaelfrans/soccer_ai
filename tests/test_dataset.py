"""Tests for dataset utilities."""

import pytest
import yaml

from src.data.dataset import fix_data_yaml, remap_labels


@pytest.fixture
def tmp_yaml(tmp_path):
    """Create a temporary data.yaml file."""
    yaml_path = tmp_path / "data.yaml"
    data = {
        "path": "/some/path",
        "train": "train/images",
        "val": "val/images",
        "nc": 20,
        "names": [f"class_{i}" for i in range(20)],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)
    return yaml_path


@pytest.fixture
def tmp_labels(tmp_path):
    """Create temporary label files with various class IDs."""
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # File with classes that exist in the map
    (label_dir / "img1.txt").write_text("4 0.5 0.5 0.1 0.1\n11 0.3 0.3 0.2 0.2\n")
    (images_dir / "img1.jpg").touch()

    # File with a class NOT in the map (should be removed)
    (label_dir / "img2.txt").write_text("99 0.5 0.5 0.1 0.1\n")
    (images_dir / "img2.jpg").touch()

    # File with mixed: one mapped, one not
    (label_dir / "img3.txt").write_text("6 0.4 0.4 0.15 0.15\n99 0.1 0.1 0.05 0.05\n")
    (images_dir / "img3.jpg").touch()

    return tmp_path


class TestFixDataYaml:
    def test_sets_default_classes(self, tmp_yaml):
        fix_data_yaml(str(tmp_yaml))

        with open(tmp_yaml) as f:
            data = yaml.safe_load(f)

        assert data["nc"] == 4
        assert data["names"] == ["ball", "goalkeeper", "player", "referee"]

    def test_custom_classes(self, tmp_yaml):
        fix_data_yaml(str(tmp_yaml), num_classes=2, class_names=["a", "b"])

        with open(tmp_yaml) as f:
            data = yaml.safe_load(f)

        assert data["nc"] == 2
        assert data["names"] == ["a", "b"]

    def test_preserves_other_fields(self, tmp_yaml):
        fix_data_yaml(str(tmp_yaml))

        with open(tmp_yaml) as f:
            data = yaml.safe_load(f)

        assert data["path"] == "/some/path"
        assert data["train"] == "train/images"


class TestRemapLabels:
    def test_remaps_known_classes(self, tmp_labels):
        class_map = {4: 0, 6: 1, 11: 2, 16: 3}
        remapped, removed = remap_labels(str(tmp_labels / "labels"), class_map)

        # img1 and img3 have mappable classes, img2 does not
        assert remapped == 2
        assert removed == 1

        # Check img1 was remapped correctly
        content = (tmp_labels / "labels" / "img1.txt").read_text()
        lines = content.strip().split("\n")
        assert lines[0].startswith("0 ")  # class 4 -> 0
        assert lines[1].startswith("2 ")  # class 11 -> 2

    def test_removes_unmapped_files(self, tmp_labels):
        class_map = {4: 0, 6: 1, 11: 2, 16: 3}
        remap_labels(str(tmp_labels / "labels"), class_map)

        # img2 had only unmapped classes -> removed
        assert not (tmp_labels / "labels" / "img2.txt").exists()
        assert not (tmp_labels / "images" / "img2.jpg").exists()

    def test_filters_unmapped_lines(self, tmp_labels):
        class_map = {4: 0, 6: 1, 11: 2, 16: 3}
        remap_labels(str(tmp_labels / "labels"), class_map)

        # img3 had one mapped line (class 6) and one unmapped (class 99)
        content = (tmp_labels / "labels" / "img3.txt").read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 1
        assert lines[0].startswith("1 ")  # class 6 -> 1

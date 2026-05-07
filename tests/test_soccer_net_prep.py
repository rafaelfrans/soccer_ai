"""Tests for SoccerNet-v3 asset preparation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.soccer_net_prep import extract_frames_from_directory


def test_extract_frames_from_directory_missing_file(tmp_path: Path) -> None:
    (tmp_path / "0.png").write_bytes(b"fake")
    dest = tmp_path / "out"
    with pytest.raises(FileNotFoundError, match="0_1"):
        extract_frames_from_directory(tmp_path, ["0.png", "0_1.png"], dest)

"""Build eval assets from SoccerNet-v3 Labels-v3.json + frame images (zip or folder).

SoccerNet-v3 ships **per-game** ``Labels-v3.json`` (bounding boxes + metadata) and
``Frames-v3.zip`` (still images in evaluation order — not a single pre-cut MP4 with a sidecar JSON).
This module merges labels with extracted frames into ``clip.mp4`` + ``gt.json`` for soccer_ai eval.
"""

from __future__ import annotations

import json
import os
import shutil
import ssl
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from src.data.soccer_net_v3 import iter_labels_v3_frames, labels_v3_to_eval_gt


def _zip_open_failure_is_io_timeout(err: zipfile.BadZipFile) -> bool:
    """True if BadZipFile was raised while handling a read timeout (cloud/network FS)."""
    chain: BaseException | None = err.__cause__ or err.__context__
    while chain is not None:
        if isinstance(chain, TimeoutError):
            return True
        if isinstance(chain, OSError) and getattr(chain, "errno", None) in (60, 110):  # ETIMEDOUT common values
            return True
        chain = chain.__cause__ or chain.__context__
    return False


def apply_insecure_ssl() -> None:
    """Call before SoccerNet urllib downloads if SSL verification fails (e.g. corporate proxy)."""
    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[assignment]


def ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return False


def ordered_image_names_from_labels(annotations: dict[str, Any]) -> list[str]:
    return [img for img, _ in iter_labels_v3_frames(annotations)]


def build_video_from_ordered_images(image_paths: list[Path], out_mp4: Path, fps: float) -> None:
    if not image_paths:
        raise ValueError("no images for video")
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg not found on PATH; install ffmpeg to build clip.mp4")

    tmp = Path(tempfile.mkdtemp(prefix="sn_frames_"))
    try:
        for i, p in enumerate(image_paths):
            dst = tmp / f"{i:06d}.png"
            shutil.copy2(p, dst)
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(tmp / "%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_mp4),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {r.stderr[:800]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def extract_zip_frames_ordered(zip_path: Path, ordered_names: list[str], dest_dir: Path) -> list[Path]:
    zp = Path(zip_path).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    try:
        with zipfile.ZipFile(zp, "r") as zf:
            names = set(zf.namelist())
            for name in ordered_names:
                if name not in names:
                    base = os.path.basename(name)
                    alt = next((n for n in names if n.endswith(base) or n.endswith("/" + base)), None)
                    if alt is None:
                        raise FileNotFoundError(f"Image {name!r} not found in {zp}")
                    name = alt
                data = zf.read(name)
                local = dest_dir / os.path.basename(name)
                local.write_bytes(data)
                out.append(local)
    except zipfile.BadZipFile as e:
        if _zip_open_failure_is_io_timeout(e):
            raise RuntimeError(
                "Reading Frames-v3.zip timed out while opening or listing the archive (Errno 60 / ETIMEDOUT). "
                "This often happens when the dataset folder is on iCloud Drive, Dropbox, Google Drive, "
                "or a slow/network disk — reads from the end of large zip files may exceed the OS timeout.\n"
                "Try: move `data/SoccerNet` (or your --soccer-net-root) to a local SSD folder, "
                "ensure cloud sync finished, or copy Frames-v3.zip locally and pass "
                "`--from-files --labels ... --frames /path/to/local/Frames-v3.zip`."
            ) from e
        raise RuntimeError(
            "Frames-v3.zip could not be read as a zip file (corrupt, incomplete download, or truncated). "
            "Delete the zip for this game and re-download, or verify with `unzip -t path/to/Frames-v3.zip`."
        ) from e
    return out


def extract_frames_from_directory(frames_dir: Path, ordered_names: list[str], dest_dir: Path) -> list[Path]:
    """Resolve each label image filename by basename under ``frames_dir`` (recursive search)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, Path] = {}
    for p in frames_dir.rglob("*"):
        if p.is_file() and p.name not in index:
            index[p.name] = p
    out: list[Path] = []
    for name in ordered_names:
        base = os.path.basename(name)
        src = index.get(base)
        if src is None:
            raise FileNotFoundError(
                f"Frame image {base!r} not found under {frames_dir}. "
                "Use the extracted Frames-v3.zip contents (or pass the .zip path instead)."
            )
        dst = dest_dir / base
        if dst.resolve() != src.resolve():
            shutil.copy2(src, dst)
        out.append(dst)
    return out


def prepare_sn_v3_eval_assets(
    *,
    labels_path: Path,
    frames_source: Path,
    output_dir: Path,
    fps: float = 25.0,
    max_frames: int | None = None,
    source_note: str | None = None,
) -> tuple[Path, Path]:
    """Build ``clip.mp4`` and ``gt.json`` under ``output_dir``.

    ``frames_source`` may be ``Frames-v3.zip`` or a directory of extracted images.

    Returns:
        ``(clip_mp4_path, gt_json_path)``
    """
    annotations = json.loads(labels_path.read_text(encoding="utf-8"))
    names = ordered_image_names_from_labels(annotations)
    if max_frames is not None:
        names = names[:max_frames]

    note = source_note or f"SoccerNet-v3:{labels_path.parent.name}"
    gt = labels_v3_to_eval_gt(annotations, source_note=note)
    if max_frames is not None:
        gt["detections"] = gt["detections"][: len(names)]
        gt["video_info"]["total_frames"] = len(gt["detections"])

    ext = output_dir / "extracted_frames"
    ext.mkdir(parents=True, exist_ok=True)

    if frames_source.is_file() and frames_source.suffix.lower() == ".zip":
        paths = extract_zip_frames_ordered(frames_source, names, ext)
    elif frames_source.is_dir():
        paths = extract_frames_from_directory(frames_source, names, ext)
    else:
        raise ValueError(f"frames_source must be a .zip file or a directory, got: {frames_source}")

    out_dir = output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_path = out_dir / "gt.json"
    gt_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")
    mp4 = out_dir / "clip.mp4"
    build_video_from_ordered_images(paths, mp4, fps=fps)
    return mp4, gt_path


def download_sn_v3_game_files(
    *,
    soccer_net_root: Path,
    split: str,
    game_index: int,
    insecure_ssl: bool = False,
) -> tuple[Path, Path, str]:
    """Download Labels-v3.json + Frames-v3.zip for one game. Returns ``(labels_path, zip_path, game_id)``."""
    if insecure_ssl:
        apply_insecure_ssl()
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
        from SoccerNet.utils import getListGames
    except ImportError as e:
        py = sys.executable
        raise SystemExit(
            "SoccerNet is not installed for the Python you are using to run this script.\n"
            f"  Interpreter: {py}\n"
            f"  Install into that environment with:\n"
            f"    {py} -m pip install -r requirements-optional.txt\n"
            "  or:\n"
            f"    {py} -m pip install 'SoccerNet>=0.1.62' Pillow\n"
            f"  (using `pip install` without `python -m` can target a different Python and cause this error.)\n"
            f"  Original import error: {e}"
        ) from e

    games = getListGames([split], task="frames")
    if game_index < 0 or game_index >= len(games):
        raise SystemExit(f"game_index must be in [0, {len(games) - 1}] for split {split!r}")
    game = games[game_index]

    root = soccer_net_root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    dl = SoccerNetDownloader(str(root))
    dl.password = "SoccerNet"
    dl.downloadGame(game, files=["Labels-v3.json", "Frames-v3.zip"], spl=split, verbose=True)

    labels_path = root / game / "Labels-v3.json"
    zip_path = root / game / "Frames-v3.zip"
    if not labels_path.is_file():
        raise SystemExit(f"Missing {labels_path} after download")
    if not zip_path.is_file():
        raise SystemExit(f"Missing {zip_path} after download")
    return labels_path, zip_path, game


def write_sample_assets(output_dir: Path, repo_root: Path, fps: float) -> tuple[Path, Path]:
    """Offline smoke test: fixture labels + synthetic PNGs → clip + gt (see tests/fixtures)."""
    try:
        from PIL import Image
    except ImportError as e:
        raise SystemExit("Pillow is required for sample mode. pip install Pillow") from e

    fixture = repo_root / "tests" / "fixtures" / "soccer_net" / "Labels-v3-sample.json"
    annotations = json.loads(fixture.read_text(encoding="utf-8"))
    gt = labels_v3_to_eval_gt(annotations, source_note="SoccerNet-v3-sample-fixture")
    w, h = int(gt["video_info"]["width"]), int(gt["video_info"]["height"])
    names = ordered_image_names_from_labels(annotations)

    out_dir = output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "extracted_frames"
    frames_dir.mkdir(exist_ok=True)
    paths: list[Path] = []
    for i, _name in enumerate(names):
        p = frames_dir / f"frame_{i:04d}.png"
        Image.new("RGB", (w, h), (30 + i * 20, 40, 50)).save(p)
        paths.append(p)

    gt["detections"] = gt["detections"][: len(paths)]
    gt["video_info"]["total_frames"] = len(paths)
    gt_path = out_dir / "gt.json"
    gt_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")
    mp4 = out_dir / "clip.mp4"
    build_video_from_ordered_images(paths, mp4, fps=fps)
    return mp4, gt_path

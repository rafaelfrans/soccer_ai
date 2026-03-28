#!/usr/bin/env python3
"""
Download a single SoccerNet-v3 game (labels + frames zip), build a clip MP4 and eval GT JSON.

Requires: pip install SoccerNet (see requirements-dev.txt), ffmpeg on PATH, network for download.

Example:
  python scripts/prepare_soccer_net_eval.py --soccer-net-root data/SoccerNet --output-dir data/sn_eval_clip

Then:
  python inference.py --model-path best.pt --source data/sn_eval_clip/clip.mp4 --output data/sn_eval_clip/out.mp4 --json-only
  python eval.py -g data/sn_eval_clip/gt.json -p data/sn_eval_clip/out_detections.json --mode full

Offline smoke test (no download):
  python scripts/prepare_soccer_net_eval.py --sample --output-dir data/sn_eval_sample
"""

from __future__ import annotations

import argparse
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

# Allow running without installing package as module
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.soccer_net_v3 import labels_v3_to_eval_gt  # noqa: E402


def _apply_insecure_ssl() -> None:
    """Use before SoccerNet urllib downloads if SSL verification fails (e.g. corporate proxy)."""
    ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001


def _require_soccer_net():
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
        from SoccerNet.utils import getListGames
    except ImportError as e:
        raise SystemExit(f"SoccerNet is not installed. Run: pip install SoccerNet\nOriginal error: {e}") from e
    return SoccerNetDownloader, getListGames


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return False


def _write_png(path: Path, width: int, height: int, rgb: tuple[int, int, int]) -> None:
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("Pillow is required for --sample. pip install Pillow") from None
    Image.new("RGB", (width, height), rgb).save(path)


def build_video_from_ordered_images(image_paths: list[Path], out_mp4: Path, fps: float) -> None:
    if not image_paths:
        raise ValueError("no images for video")
    if not _ffmpeg_available():
        raise SystemExit("ffmpeg not found on PATH; install ffmpeg to build clip.mp4")

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
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        for name in ordered_names:
            # Some zips prefix paths; try basename match
            if name not in names:
                base = os.path.basename(name)
                alt = next((n for n in names if n.endswith(base) or n.endswith("/" + base)), None)
                if alt is None:
                    raise FileNotFoundError(f"Image {name!r} not found in {zip_path}")
                name = alt
            data = zf.read(name)
            local = dest_dir / os.path.basename(name)
            local.write_bytes(data)
            out.append(local)
    return out


def ordered_image_names_from_labels(annotations: dict[str, Any]) -> list[str]:
    from src.data.soccer_net_v3 import iter_labels_v3_frames

    return [img for img, _ in iter_labels_v3_frames(annotations)]


def run_sample(output_dir: Path, fps: float) -> None:
    fixture = _REPO_ROOT / "tests" / "fixtures" / "soccer_net" / "Labels-v3-sample.json"
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
        _write_png(p, w, h, (30 + i * 20, 40, 50))
        paths.append(p)
    # Truncate GT if we ever shorten names list
    gt["detections"] = gt["detections"][: len(paths)]
    gt["video_info"]["total_frames"] = len(paths)
    gt_path = out_dir / "gt.json"
    gt_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")
    mp4 = out_dir / "clip.mp4"
    build_video_from_ordered_images(paths, mp4, fps=fps)
    print(f"Sample ready:\n  video: {mp4}\n  gt:    {gt_path}")


def run_download(args: argparse.Namespace) -> None:
    if getattr(args, "insecure_ssl", False):
        _apply_insecure_ssl()
        print("Warning: SSL certificate verification disabled for this download (--insecure-ssl).")
    SoccerNetDownloader, getListGames = _require_soccer_net()
    games = getListGames([args.split], task="frames")
    if args.game_index < 0 or args.game_index >= len(games):
        raise SystemExit(f"game-index must be in [0, {len(games) - 1}] for split {args.split!r}")
    game = games[args.game_index]
    print(f"Selected game: {game}")

    root = Path(args.soccer_net_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dl = SoccerNetDownloader(str(root))
    dl.password = "SoccerNet"
    dl.downloadGame(game, files=["Labels-v3.json", "Frames-v3.zip"], spl=args.split, verbose=True)

    labels_path = root / game / "Labels-v3.json"
    zip_path = root / game / "Frames-v3.zip"
    if not labels_path.is_file():
        raise SystemExit(f"Missing {labels_path} after download")
    if not zip_path.is_file():
        raise SystemExit(f"Missing {zip_path} after download")

    annotations = json.loads(labels_path.read_text(encoding="utf-8"))
    names = ordered_image_names_from_labels(annotations)
    if args.max_frames is not None:
        names = names[: args.max_frames]

    gt = labels_v3_to_eval_gt(annotations, source_note=f"SoccerNet-v3:{game}")
    if args.max_frames is not None:
        gt["detections"] = gt["detections"][: args.max_frames]
        gt["video_info"]["total_frames"] = len(gt["detections"])

    ext = out_dir / "extracted_frames"
    ext.mkdir(parents=True, exist_ok=True)
    try:
        paths = extract_zip_frames_ordered(zip_path, names, ext)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

    gt_path = out_dir / "gt.json"
    gt_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")
    mp4 = out_dir / "clip.mp4"
    build_video_from_ordered_images(paths, mp4, fps=args.fps)
    print(f"Ready:\n  video: {mp4}\n  gt:    {gt_path}\n  (from {len(paths)} frames)")


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare SoccerNet-v3 clip + GT JSON for soccer_ai eval")
    p.add_argument("--soccer-net-root", type=str, default=None, help="Root dir for SoccerNet data (download target)")
    p.add_argument("--output-dir", type=str, required=True, help="Where to write clip.mp4 and gt.json")
    p.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--game-index", type=int, default=0, help="Index into split game list (frames task)")
    p.add_argument("--fps", type=float, default=25.0, help="FPS for synthesized clip.mp4")
    p.add_argument("--max-frames", type=int, default=None, help="Use only first N frames (smaller MP4)")
    p.add_argument(
        "--sample",
        action="store_true",
        help="Offline: use bundled Labels-v3-sample.json + synthetic PNGs (no SoccerNet download)",
    )
    p.add_argument(
        "--insecure-ssl",
        action="store_true",
        help="Disable HTTPS certificate verification (use if download fails with SSL errors)",
    )
    args = p.parse_args()

    out = Path(args.output_dir)
    if args.sample:
        run_sample(out, fps=args.fps)
        return

    if not args.soccer_net_root:
        raise SystemExit("--soccer-net-root is required unless --sample is set")
    run_download(args)


if __name__ == "__main__":
    main()

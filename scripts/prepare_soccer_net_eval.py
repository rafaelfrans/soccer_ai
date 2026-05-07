#!/usr/bin/env python3
"""
Download a single SoccerNet-v3 game (labels + frames zip), build a clip MP4 and eval GT JSON.

Requires: pip install -r requirements-optional.txt (SoccerNet, Pillow), ffmpeg on PATH, network for download.

Example (download one game):
  python scripts/prepare_soccer_net_eval.py --soccer-net-root data/SoccerNet --output-dir data/sn_eval_clip

Example (you already have Labels-v3.json + Frames-v3.zip or extracted frames folder):
  python scripts/prepare_soccer_net_eval.py --labels path/to/Labels-v3.json --frames path/to/Frames-v3.zip --output-dir data/sn_eval_clip

Then:
  python scripts/soccer_net_benchmark.py --model-path best.pt --work-dir data/sn_eval_clip --skip-prepare

Or use ``soccer_net_benchmark.py`` with ``--from-files`` / ``--download`` to prepare + run eval in one step.

Offline smoke test (no download):
  python scripts/prepare_soccer_net_eval.py --sample --output-dir data/sn_eval_sample
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.soccer_net_prep import (  # noqa: E402
    download_sn_v3_game_files,
    prepare_sn_v3_eval_assets,
    write_sample_assets,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare SoccerNet-v3 clip + GT JSON for soccer_ai eval")
    p.add_argument("--output-dir", type=str, required=True, help="Where to write clip.mp4 and gt.json")
    p.add_argument("--fps", type=float, default=25.0, help="FPS for synthesized clip.mp4")
    p.add_argument("--max-frames", type=int, default=None, help="Use only first N labeled frames (smaller MP4)")
    p.add_argument(
        "--sample",
        action="store_true",
        help="Offline: use bundled Labels-v3-sample.json + synthetic PNGs (no SoccerNet download)",
    )
    p.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Path to Labels-v3.json (use with --frames; skips SoccerNet download)",
    )
    p.add_argument(
        "--frames",
        type=str,
        default=None,
        help="Path to Frames-v3.zip or a directory of extracted frame images (use with --labels)",
    )
    p.add_argument("--soccer-net-root", type=str, default=None, help="Root dir for SoccerNet data (download target)")
    p.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--game-index", type=int, default=0, help="Index into split game list (frames task)")
    p.add_argument(
        "--insecure-ssl",
        action="store_true",
        help="Disable HTTPS certificate verification (use if download fails with SSL errors)",
    )
    args = p.parse_args()

    out = Path(args.output_dir).resolve()

    if args.sample:
        mp4, gt = write_sample_assets(out, _REPO_ROOT, fps=args.fps)
        print(f"Sample ready:\n  video: {mp4}\n  gt:    {gt}")
        return

    if args.labels and args.frames:
        mp4, gt = prepare_sn_v3_eval_assets(
            labels_path=Path(args.labels).resolve(),
            frames_source=Path(args.frames).resolve(),
            output_dir=out,
            fps=args.fps,
            max_frames=args.max_frames,
        )
        print(f"Ready:\n  video: {mp4}\n  gt:    {gt}\n  (local Labels + frames)")
        return

    if args.labels or args.frames:
        raise SystemExit("Provide both --labels and --frames, or neither (for --download mode).")

    if not args.soccer_net_root:
        raise SystemExit("--soccer-net-root is required unless --sample or (--labels and --frames) is used")

    if args.insecure_ssl:
        print("Warning: SSL certificate verification disabled for this download (--insecure-ssl).")

    labels_path, zip_path, game = download_sn_v3_game_files(
        soccer_net_root=Path(args.soccer_net_root).resolve(),
        split=args.split,
        game_index=args.game_index,
        insecure_ssl=args.insecure_ssl,
    )
    print(f"Selected game: {game}")

    mp4, gt = prepare_sn_v3_eval_assets(
        labels_path=labels_path,
        frames_source=zip_path,
        output_dir=out,
        fps=args.fps,
        max_frames=args.max_frames,
        source_note=f"SoccerNet-v3:{game}",
    )
    print(f"Ready:\n  video: {mp4}\n  gt:    {gt}\n  (from {game})")


if __name__ == "__main__":
    main()

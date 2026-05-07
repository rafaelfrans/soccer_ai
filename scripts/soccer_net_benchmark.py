#!/usr/bin/env python3
"""
End-to-end SoccerNet-v3 benchmark: prepare clip + GT (optional), run inference, run eval.

SoccerNet-v3 provides **Labels-v3.json** + **Frames-v3.zip** per game (still images + boxes), not a
single host video file with a matching JSON. This script builds ``clip.mp4`` + ``gt.json``, runs your
model, and writes ``eval_report.json``.

Examples::

  # One command: download game 0 from test split, evaluate
  python scripts/soccer_net_benchmark.py \\
    --model-path best.pt --work-dir data/sn_bench \\
    --download --soccer-net-root data/SoccerNet --split test --game-index 0

  # Local files (after manual download or existing cache)
  python scripts/soccer_net_benchmark.py \\
    --model-path best.pt --work-dir data/sn_local \\
    --from-files --labels path/to/Labels-v3.json --frames path/to/Frames-v3.zip

  # Offline sample (no network)
  python scripts/soccer_net_benchmark.py \\
    --model-path best.pt --work-dir data/sn_sample --sample

  # Inference + eval only (clip.mp4 + gt.json already in work-dir)
  python scripts/soccer_net_benchmark.py \\
    --model-path best.pt --work-dir data/sn_eval_clip --skip-prepare --tracking

Requires: base project deps + ffmpeg for clip synthesis; optional ``pip install -r requirements-optional.txt``
for SoccerNet download mode.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _run_inference(repo_root: Path, model_path: str, clip: Path, predictions_stub: Path) -> Path:
    cmd = [
        sys.executable,
        str(repo_root / "inference.py"),
        "--model-path",
        model_path,
        "--source",
        str(clip),
        "--output",
        str(predictions_stub),
        "--json-only",
    ]
    r = subprocess.run(cmd, cwd=str(repo_root))
    if r.returncode != 0:
        raise SystemExit(r.returncode)
    pred_json = predictions_stub.with_name(predictions_stub.stem + "_detections.json")
    if not pred_json.is_file():
        raise SystemExit(f"Expected predictions JSON at {pred_json}")
    return pred_json


def _run_eval_report(gt: Path, pred: Path, tracking: bool) -> dict:
    from src.eval.runner import build_report

    return build_report(
        gt_path=str(gt),
        pred_path=str(pred),
        mode="full",
        run_tracking=tracking,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="SoccerNet-v3: prepare assets, run inference, evaluate against GT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model-path", required=True, help="Path to trained YOLO weights (.pt)")
    p.add_argument("--work-dir", required=True, help="Directory for clip, gt, predictions, report")
    p.add_argument(
        "--tracking",
        action="store_true",
        help="Include HOTA/MOTA/IDF1 (needs track_id in GT — SoccerNet jersey IDs when numeric)",
    )
    p.add_argument("--fps", type=float, default=25.0, help="FPS when synthesizing clip.mp4")
    p.add_argument("--max-frames", type=int, default=None, help="Cap labeled frames / clip length")

    prep = p.add_mutually_exclusive_group(required=True)
    prep.add_argument(
        "--download",
        action="store_true",
        help="Download one SoccerNet-v3 game (requires --soccer-net-root and optional --split / --game-index)",
    )
    prep.add_argument(
        "--from-files",
        action="store_true",
        help="Use local Labels-v3.json + Frames-v3.zip or extracted frames dir (requires --labels and --frames)",
    )
    prep.add_argument("--sample", action="store_true", help="Use bundled offline fixture + synthetic frames")
    prep.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Use existing clip.mp4 + gt.json inside --work-dir (already prepared)",
    )

    p.add_argument("--labels", type=str, default=None, help="Path to Labels-v3.json (with --from-files)")
    p.add_argument(
        "--frames",
        type=str,
        default=None,
        help="Path to Frames-v3.zip or directory of extracted images (with --from-files)",
    )
    p.add_argument("--soccer-net-root", type=str, default=None, help="SoccerNet download directory")
    p.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--game-index", type=int, default=0)
    p.add_argument("--insecure-ssl", action="store_true", help="For --download only: skip SSL verify")

    args = p.parse_args()

    if args.from_files:
        if not args.labels or not args.frames:
            raise SystemExit("--from-files requires both --labels and --frames")
    elif args.labels or args.frames:
        raise SystemExit("Use --from-files together with --labels and --frames")
    if args.download and not args.soccer_net_root:
        raise SystemExit("--download requires --soccer-net-root")
    if args.skip_prepare and (
        args.soccer_net_root or args.labels or args.frames or args.sample or args.download or args.from_files
    ):
        raise SystemExit("Do not combine --skip-prepare with other prepare options.")

    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    clip = work / "clip.mp4"
    gt = work / "gt.json"
    predictions_video = work / "predictions.mp4"
    report_path = work / "eval_report.json"

    if not args.skip_prepare:
        from src.data.soccer_net_prep import (
            download_sn_v3_game_files,
            prepare_sn_v3_eval_assets,
            write_sample_assets,
        )

        if args.sample:
            c, g = write_sample_assets(work, _REPO_ROOT, fps=args.fps)
            clip, gt = c, g
        elif args.download:
            labels_path, zip_path, game = download_sn_v3_game_files(
                soccer_net_root=Path(args.soccer_net_root).resolve(),
                split=args.split,
                game_index=args.game_index,
                insecure_ssl=args.insecure_ssl,
            )
            if args.insecure_ssl:
                print("Warning: SSL certificate verification was disabled for download.")
            print(f"Downloaded game: {game}")
            c, g = prepare_sn_v3_eval_assets(
                labels_path=labels_path,
                frames_source=zip_path,
                output_dir=work,
                fps=args.fps,
                max_frames=args.max_frames,
                source_note=f"SoccerNet-v3:{game}",
            )
            clip, gt = c, g
        else:
            c, g = prepare_sn_v3_eval_assets(
                labels_path=Path(args.labels).resolve(),
                frames_source=Path(args.frames).resolve(),
                output_dir=work,
                fps=args.fps,
                max_frames=args.max_frames,
            )
            clip, gt = c, g
    else:
        if not clip.is_file() or not gt.is_file():
            raise SystemExit(f"--skip-prepare requires {clip} and {gt}")

    print(f"Using video: {clip}\nUsing GT:    {gt}")

    pred_json = _run_inference(_REPO_ROOT, args.model_path, clip, predictions_video)
    print(f"Predictions: {pred_json}")

    report = _run_eval_report(gt, pred_json, args.tracking)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved to {report_path}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

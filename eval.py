#!/usr/bin/env python3
"""
Evaluate pipeline predictions against ground-truth JSON (see eval_sets/README.md for GT schema).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.eval.runner import build_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate *_detections.json against ground-truth JSON")
    parser.add_argument("--ground-truth", "-g", required=True, help="Ground-truth JSON path")
    parser.add_argument("--predictions", "-p", required=True, help="Predictions JSON (full or model_only export)")
    parser.add_argument(
        "--mode",
        choices=["full", "model_only"],
        default="full",
        help="Must match how predictions were exported (ball unpadding applies only for full)",
    )
    parser.add_argument(
        "--tracking",
        action="store_true",
        help="Run HOTA/MOTA/IDF1 via TrackEval if GT and preds have person track IDs",
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Write report JSON to this path")
    args = parser.parse_args()

    report = build_report(
        gt_path=args.ground_truth,
        pred_path=args.predictions,
        mode=args.mode,
        run_tracking=args.tracking,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        outp = Path(args.output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(text, encoding="utf-8")
        print(f"Report saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Download SoccerNet-v3 **Labels-v3.json** + **Frames-v3.zip** for one game only (no ffmpeg, no clip).

Use this when you want the raw SoccerNet files first — same artifacts as the SoccerNet pip downloader
used elsewhere in this repo. Store them on a **local disk** (not iCloud-only paths) to avoid zip read
timeouts when you later run `prepare_soccer_net_eval.py` or `soccer_net_benchmark.py`.

Requires: ``python3 -m pip install -r requirements-optional.txt`` (same Python you use to run this script).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.soccer_net_prep import download_sn_v3_game_files  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Download SoccerNet-v3 labels + frames zip for one game (frames split)")
    p.add_argument(
        "--soccer-net-root",
        type=str,
        required=True,
        help="Directory where SoccerNet data is stored (e.g. data/SoccerNet)",
    )
    p.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--game-index", type=int, default=0, help="Index into the split game list")
    p.add_argument("--insecure-ssl", action="store_true", help="Disable TLS verification if downloads fail")
    args = p.parse_args()

    labels, frames_zip, game = download_sn_v3_game_files(
        soccer_net_root=Path(args.soccer_net_root).resolve(),
        split=args.split,
        game_index=args.game_index,
        insecure_ssl=args.insecure_ssl,
    )

    print(f"Game: {game}")
    print(f"Labels-v3.json: {labels}")
    print(f"Frames-v3.zip:  {frames_zip}")
    print()
    print("Build clip + gt.json for eval:")
    print("  python3 scripts/prepare_soccer_net_eval.py \\")
    print(f"    --labels {labels} \\")
    print(f"    --frames {frames_zip} \\")
    print("    --output-dir <your_output_dir>")
    print()
    print("Or prep + inference + eval_report.json:")
    print("  python3 scripts/soccer_net_benchmark.py --model-path YOUR.pt --work-dir <your_output_dir> \\")
    print(f"    --from-files --labels {labels} --frames {frames_zip}")


if __name__ == "__main__":
    main()

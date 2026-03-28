#!/usr/bin/env python3
"""
Inference entry point for soccer AI video processing.
"""

import argparse
import contextlib
import os
import subprocess
import tempfile

from src.inference import AnnotatorConfig, VideoProcessor


def main():
    parser = argparse.ArgumentParser(description="Process soccer videos with object detection and tracking")

    # Model arguments
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained YOLO model")

    # Video arguments
    parser.add_argument("--source", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=True, help="Path to output video")
    parser.add_argument(
        "--fix-rotation",
        action="store_true",
        help="Re-encode video with ffmpeg to bake in rotation metadata (fixes sideways iPhone/MOV videos)",
    )

    # Detection arguments
    parser.add_argument(
        "--conf-min",
        type=float,
        default=0.1,
        help="Minimum confidence for YOLO raw output (keep low to retain candidates for per-class filtering)",
    )
    parser.add_argument(
        "--ball-conf",
        type=float,
        default=0.1,
        help="Per-class confidence threshold for ball detections",
    )
    parser.add_argument(
        "--player-conf",
        type=float,
        default=0.4,
        help="Per-class confidence threshold for non-ball detections (players, goalkeepers, referees)",
    )
    parser.add_argument("--nms-threshold", type=float, default=0.5, help="NMS threshold for non-ball detections")
    parser.add_argument(
        "--ball-nms-threshold",
        type=float,
        default=0.3,
        help="NMS threshold for ball detections (removes duplicate ball detections)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for YOLO internal NMS (lower = more aggressive suppression)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["low-quality", "high-quality"],
        default=None,
        help="Preset for video quality: low-quality (more permissive) or high-quality (stricter)",
    )
    parser.add_argument("--ball-class-id", type=int, default=0, help="Class ID for ball")
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["full", "model_only"],
        default="full",
        help="full: pipeline JSON (tracked_objects + ball); model_only: raw YOLO objects only (ablation eval)",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Do not write output video; only write detections JSON (faster for eval)",
    )

    # Annotator arguments (optional customization)
    parser.add_argument(
        "--ellipse-colors",
        type=str,
        nargs="+",
        default=["#00BFFF", "#FF1493", "#FFD700"],
        help="Colors for ellipse annotations",
    )
    parser.add_argument("--ellipse-thickness", type=int, default=2, help="Ellipse annotation thickness")
    parser.add_argument("--triangle-color", type=str, default="#FFD700", help="Color for triangle (ball) annotations")

    args = parser.parse_args()

    # Apply preset overrides for ball_conf and player_conf
    ball_conf = args.ball_conf
    player_conf = args.player_conf
    if args.preset == "low-quality":
        ball_conf = 0.15
        player_conf = 0.35
        print("📐 Using low-quality preset: ball_conf=0.15, player_conf=0.35")
    elif args.preset == "high-quality":
        ball_conf = 0.25
        player_conf = 0.5
        print("📐 Using high-quality preset: ball_conf=0.25, player_conf=0.5")

    # Create annotator config
    config = AnnotatorConfig(
        ellipse_colors=args.ellipse_colors, ellipse_thickness=args.ellipse_thickness, triangle_color=args.triangle_color
    )

    # Initialize processor
    print(f"🤖 Loading model: {args.model_path}")
    processor = VideoProcessor(
        model_path=args.model_path,
        ball_class_id=args.ball_class_id,
        conf_min=args.conf_min,
        ball_conf=ball_conf,
        player_conf=player_conf,
        nms_threshold=args.nms_threshold,
        ball_nms_threshold=args.ball_nms_threshold,
        iou_threshold=args.iou,
        config=config,
    )

    # Process video
    source_path = args.source
    rotation_tmp: str | None = None
    if args.fix_rotation:
        print("📐 Fixing rotation: re-encoding with ffmpeg (bakes in orientation metadata)...")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            rotation_tmp = tmp.name
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    args.source,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                    "-an",
                    "-y",
                    rotation_tmp,
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                if rotation_tmp and os.path.isfile(rotation_tmp):
                    os.unlink(rotation_tmp)
                raise RuntimeError(f"ffmpeg failed: {result.stderr[:500]}")
            source_path = rotation_tmp
        except FileNotFoundError:
            if rotation_tmp and os.path.isfile(rotation_tmp):
                with contextlib.suppress(OSError):
                    os.unlink(rotation_tmp)
            raise SystemExit("ffmpeg not found. Install it (e.g. brew install ffmpeg) to use --fix-rotation.") from None
        except subprocess.TimeoutExpired:
            if rotation_tmp and os.path.isfile(rotation_tmp):
                with contextlib.suppress(OSError):
                    os.unlink(rotation_tmp)
            raise SystemExit("ffmpeg timed out.") from None

    print(f"📹 Processing video: {source_path}")
    if args.json_only:
        print("📋 JSON-only mode: skipping annotated video output")
    else:
        print(f"💾 Output will be saved to: {args.output}")

    try:
        json_path = processor.process_video(
            source_path=source_path,
            target_path=args.output,
            reset_tracker=True,
            eval_mode=args.eval_mode,
            write_video=not args.json_only,
        )
    finally:
        if rotation_tmp and os.path.isfile(rotation_tmp) and source_path == rotation_tmp:
            with contextlib.suppress(OSError):
                os.unlink(rotation_tmp)

    print("\n✅ Processing complete!")
    if args.json_only:
        print(f"📁 Detections JSON: {json_path}")
    else:
        print(f"📁 Annotated video: {args.output}")
        print(f"📁 Detections JSON: {json_path}")


if __name__ == "__main__":
    main()

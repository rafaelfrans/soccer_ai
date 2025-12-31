#!/usr/bin/env python3
"""
Inference entry point for soccer AI video processing.
"""

import argparse
from src.inference import VideoProcessor, AnnotatorConfig


def main():
    parser = argparse.ArgumentParser(description="Process soccer videos with object detection and tracking")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained YOLO model")
    
    # Video arguments
    parser.add_argument("--source", type=str, required=True,
                        help="Path to input video")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output video")
    parser.add_argument("--json", type=str, default=None,
                        help="Optional path to save bounding box data as JSON. If not specified, auto-generates from output path (e.g., output.mp4 -> output_detections.json)")
    
    # Detection arguments
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Confidence threshold for detections")
    parser.add_argument("--nms-threshold", type=float, default=0.5,
                        help="NMS threshold for non-ball detections")
    parser.add_argument("--ball-class-id", type=int, default=0,
                        help="Class ID for ball")
    
    # Annotator arguments (optional customization)
    parser.add_argument("--ellipse-colors", type=str, nargs="+",
                        default=['#00BFFF', '#FF1493', '#FFD700'],
                        help="Colors for ellipse annotations")
    parser.add_argument("--ellipse-thickness", type=int, default=2,
                        help="Ellipse annotation thickness")
    parser.add_argument("--triangle-color", type=str, default='#FFD700',
                        help="Color for triangle (ball) annotations")
    
    args = parser.parse_args()
    
    # Create annotator config
    config = AnnotatorConfig(
        ellipse_colors=args.ellipse_colors,
        ellipse_thickness=args.ellipse_thickness,
        triangle_color=args.triangle_color
    )
    
    # Initialize processor
    print(f"🤖 Loading model: {args.model_path}")
    processor = VideoProcessor(
        model_path=args.model_path,
        ball_class_id=args.ball_class_id,
        conf_threshold=args.conf,
        nms_threshold=args.nms_threshold,
        config=config
    )
    
    # Process video
    print(f"📹 Processing video: {args.source}")
    print(f"💾 Output will be saved to: {args.output}")
    print(f"📊 Detection data will be saved automatically")
    
    processor.process_video(
        source_path=args.source,
        target_path=args.output,
        reset_tracker=True,
        json_path=args.json
    )
    
    print(f"\n✅ Processing complete!")
    print(f"📁 Output saved to: {args.output}")


if __name__ == "__main__":
    main()


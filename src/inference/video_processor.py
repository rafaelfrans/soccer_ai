"""Video processing and annotation utilities."""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import json
import os
import numpy as np
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm


@dataclass
class AnnotatorConfig:
    """Configuration for video annotators."""
    ellipse_colors: list = None
    ellipse_thickness: int = 2
    label_colors: list = None
    label_text_color: str = '#000000'
    label_text_position: str = 'BOTTOM_CENTER'
    triangle_color: str = '#FFD700'
    triangle_base: int = 25
    triangle_height: int = 21
    triangle_outline_thickness: int = 1
    
    def __post_init__(self):
        if self.ellipse_colors is None:
            self.ellipse_colors = ['#00BFFF', '#FF1493', '#FFD700']
        if self.label_colors is None:
            self.label_colors = ['#00BFFF', '#FF1493', '#FFD700']


class VideoProcessor:
    """Process videos with object detection and tracking."""
    
    def __init__(
        self,
        model_path: str,
        ball_class_id: int = 0,
        conf_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        config: Optional[AnnotatorConfig] = None
    ):
        """
        Initialize video processor.
        
        Args:
            model_path: Path to trained YOLO model
            ball_class_id: Class ID for ball (default: 0)
            conf_threshold: Confidence threshold for detections
            nms_threshold: NMS threshold for non-ball detections
            config: Annotator configuration
        """
        self.model = YOLO(model_path)
        self.ball_class_id = ball_class_id
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.config = config or AnnotatorConfig()
        
        # Initialize annotators
        self.ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(self.config.ellipse_colors),
            thickness=self.config.ellipse_thickness
        )
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(self.config.label_colors),
            text_color=sv.Color.from_hex(self.config.label_text_color),
            text_position=getattr(sv.Position, self.config.label_text_position)
        )
        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex(self.config.triangle_color),
            base=self.config.triangle_base,
            height=self.config.triangle_height,
            outline_thickness=self.config.triangle_outline_thickness
        )
        
        # Initialize tracker
        self.tracker = sv.ByteTrack()
    
    def process_frame(self, frame, return_detections: bool = False):
        """
        Process a single frame.
        
        Args:
            frame: Input frame (numpy array)
            return_detections: If True, return detection data along with annotated frame
        
        Returns:
            Annotated frame, or tuple of (annotated_frame, detection_data) if return_detections=True
        """
        # Run inference
        result = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # Separate ball detections
        ball_detections = detections[detections.class_id == self.ball_class_id]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
        
        # Process other detections (players, goalkeepers, referees)
        all_detections = detections[detections.class_id != self.ball_class_id]
        all_detections = all_detections.with_nms(threshold=self.nms_threshold, class_agnostic=True)
        # Adjust class IDs (subtract 1 since ball is class 0)
        all_detections.class_id -= 1
        all_detections = self.tracker.update_with_detections(detections=all_detections)
        
        # Create labels with tracker IDs
        labels = [
            f"#{tracker_id}"
            for tracker_id in all_detections.tracker_id
        ]
        
        # Annotate frame
        annotated_frame = frame.copy()
        annotated_frame = self.ellipse_annotator.annotate(
            scene=annotated_frame,
            detections=all_detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=all_detections,
            labels=labels
        )
        annotated_frame = self.triangle_annotator.annotate(
            scene=annotated_frame,
            detections=ball_detections
        )
        
        if return_detections:
            detection_data = self._extract_detection_data(all_detections, ball_detections)
            return annotated_frame, detection_data
        
        return annotated_frame
    
    def _extract_detection_data(
        self, 
        tracked_detections: sv.Detections, 
        ball_detections: sv.Detections
    ) -> Dict:
        """
        Extract detection data from detections objects.
        
        Args:
            tracked_detections: Detections for tracked objects (players, goalkeepers, referees)
            ball_detections: Detections for ball
        
        Returns:
            Dictionary containing detection data
        """
        detection_data = {
            "tracked_objects": [],
            "ball": []
        }
        
        # Extract tracked objects (players, goalkeepers, referees)
        if len(tracked_detections) > 0:
            for i in range(len(tracked_detections)):
                bbox = tracked_detections.xyxy[i].tolist()
                class_id = int(tracked_detections.class_id[i])
                tracker_id = int(tracked_detections.tracker_id[i]) if tracked_detections.tracker_id is not None else None
                confidence = float(tracked_detections.confidence[i]) if tracked_detections.confidence is not None else None
                
                detection_data["tracked_objects"].append({
                    "bbox": bbox,  # [x1, y1, x2, y2]
                    "class_id": class_id,
                    "tracker_id": tracker_id,
                    "confidence": confidence
                })
        
        # Extract ball detections
        if len(ball_detections) > 0:
            for i in range(len(ball_detections)):
                bbox = ball_detections.xyxy[i].tolist()
                class_id = int(ball_detections.class_id[i])
                confidence = float(ball_detections.confidence[i]) if ball_detections.confidence is not None else None
                
                detection_data["ball"].append({
                    "bbox": bbox,  # [x1, y1, x2, y2]
                    "class_id": class_id,
                    "confidence": confidence
                })
        
        return detection_data
    
    def process_video(
        self,
        source_path: str,
        target_path: str,
        reset_tracker: bool = True,
        json_path: Optional[str] = None
    ):
        """
        Process entire video and save annotated output.
        
        Args:
            source_path: Path to input video
            target_path: Path to output video
            reset_tracker: Whether to reset tracker before processing
            json_path: Optional path to save bounding box data as JSON.
                      If None, auto-generates from target_path (e.g., output.mp4 -> output_detections.json)
        """
        if reset_tracker:
            self.tracker.reset()
        
        # Auto-generate JSON path from target_path if not provided
        if json_path is None:
            base_name = os.path.splitext(target_path)[0]
            json_path = f"{base_name}_detections.json"
        
        video_info = sv.VideoInfo.from_video_path(source_path)
        frame_generator = sv.get_video_frames_generator(source_path)
        
        # Always collect detection data
        all_detections = []
        
        with sv.VideoSink(target_path, video_info=video_info) as video_sink:
            for frame_idx, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames)):
                annotated_frame, detection_data = self.process_frame(frame, return_detections=True)
                # Add frame number to detection data
                frame_data = {
                    "frame_number": frame_idx,
                    **detection_data
                }
                all_detections.append(frame_data)
                
                video_sink.write_frame(annotated_frame)
        
        # Always save detection data to JSON
        output_data = {
            "video_info": {
                "source_path": source_path,
                "fps": video_info.fps,
                "width": video_info.width,
                "height": video_info.height,
                "total_frames": video_info.total_frames
            },
            "detections": all_detections
        }
        
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"📊 Detection data saved to: {json_path}")


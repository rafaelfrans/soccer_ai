"""Video processing and annotation utilities."""

from dataclasses import dataclass
from typing import Optional
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
    
    def process_frame(self, frame):
        """
        Process a single frame.
        
        Args:
            frame: Input frame (numpy array)
        
        Returns:
            Annotated frame
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
        
        return annotated_frame
    
    def process_video(
        self,
        source_path: str,
        target_path: str,
        reset_tracker: bool = True
    ):
        """
        Process entire video and save annotated output.
        
        Args:
            source_path: Path to input video
            target_path: Path to output video
            reset_tracker: Whether to reset tracker before processing
        """
        if reset_tracker:
            self.tracker.reset()
        
        video_info = sv.VideoInfo.from_video_path(source_path)
        frame_generator = sv.get_video_frames_generator(source_path)
        
        with sv.VideoSink(target_path, video_info=video_info) as video_sink:
            for frame in tqdm(frame_generator, total=video_info.total_frames):
                annotated_frame = self.process_frame(frame)
                video_sink.write_frame(annotated_frame)


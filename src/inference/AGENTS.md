# Inference Module

Processes soccer videos with YOLO detection + ByteTrack tracking.

## `VideoProcessor`

Main class. Initialized with a YOLO model path, then call `process_video()`.

### Processing Pipeline (per frame)

1. Run YOLO inference → `sv.Detections`
2. Separate ball detections (class 0) from others
3. Pad ball bounding boxes by 10px (balls are small)
4. Apply NMS to non-ball detections (class-agnostic)
5. Subtract 1 from non-ball class IDs (so goalkeeper=0, player=1, referee=2 for the color palette)
6. Update ByteTrack tracker with non-ball detections
7. Annotate: ellipses for players/goalkeepers/referees, triangles for ball, labels with tracker IDs

### Output

- Annotated video (mp4)
- Detection JSON (`*_detections.json`) — auto-generated alongside video, contains per-frame bounding boxes with tracker IDs and confidence scores

## `AnnotatorConfig`

Dataclass for annotation styling (colors, thickness, etc.). Defaults are sensible — only override for visual customization.

## Gotcha

Non-ball class IDs are shifted by -1 internally (`class_id -= 1`) for color palette indexing. This is intentional — the 3-color palette maps to goalkeeper/player/referee after the ball class is removed.

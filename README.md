# Soccer AI

Object detection and tracking system for soccer videos using YOLO.

## Features

- **Object Detection**: Detects ball, goalkeeper, player, and referee in soccer videos
- **Object Tracking**: Tracks players and other objects across video frames using ByteTrack
- **Video Annotation**: Annotates videos with bounding boxes, labels, and tracking IDs

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
soccer_ai/
├── src/
│   └── soccer_ai/
│       ├── data/          # Dataset download and preprocessing
│       ├── models/        # Model training utilities
│       ├── inference/     # Video processing and inference
│       └── utils/         # Utility functions
├── train.py              # Training entry point
├── inference.py          # Inference entry point
└── requirements.txt      # Dependencies
```

## Usage

### Training

Train a model on soccer player detection:

```bash
python train.py \
  --roboflow-api-key YOUR_API_KEY \
  --model-path /path/to/pretrained/model.pt \
  --batch 10 \
  --epochs 35 \
  --imgsz 1280
```

Fine-tune on small dataset (freeze backbone, lower learning rate):

```bash
python train.py \
  --roboflow-api-key YOUR_API_KEY \
  --model-path /path/to/pretrained/model.pt \
  --freeze 24 \
  --lr0 1e-4 \
  --lrf 0.01 \
  --batch 10 \
  --epochs 35 \
  --imgsz 1280
```

**Arguments:**
- `--roboflow-api-key`: Roboflow API key (required)
- `--roboflow-workspace`: Roboflow workspace name (default: soccer-ai-lkex8)
- `--roboflow-project`: Roboflow project name (default: soccer-players-xy9vk-ebc0t)
- `--roboflow-version`: Dataset version (default: 1)
- `--model-path`: Path to pretrained model (required)
- `--batch`: Batch size (default: 10)
- `--epochs`: Number of epochs (default: 35)
- `--imgsz`: Image size (default: 1280)
- `--device`: Device ID, 0 for GPU or 'cpu' (default: 0)
- `--patience`: Early stopping patience (default: 5)
- `--freeze`: Number of layers to freeze (e.g., 24 freezes entire backbone for fine-tuning detection layers)
- `--lr0`: Initial learning rate (e.g., 1e-4 for fine-tuning on small datasets)
- `--lrf`: Final learning rate factor (final_lr = lr0 * lrf, e.g., 0.01)
- `--project`: Project directory name (default: soccerai_training)
- `--name`: Run name (optional)
- `--skip-download`: Skip dataset download (use existing dataset)
- `--dataset-path`: Path to existing dataset (if skip-download)

### Inference

Process a video with object detection and tracking:

```bash
python inference.py \
  --model-path /path/to/trained/model.pt \
  --source /path/to/input/video.mp4 \
  --output /path/to/output/video.mp4 \
  --conf 0.3
```

**Arguments:**
- `--model-path`: Path to trained model (required)
- `--source`: Path to input video (required)
- `--output`: Path to output video (required)
- `--conf`: Confidence threshold (default: 0.3)
- `--nms-threshold`: NMS threshold for non-ball detections (default: 0.5)
- `--ball-class-id`: Class ID for ball (default: 0)
- `--ellipse-colors`: Colors for ellipse annotations (default: #00BFFF #FF1493 #FFD700)
- `--ellipse-thickness`: Ellipse thickness (default: 2)
- `--triangle-color`: Color for ball triangle annotation (default: #FFD700)

## Classes

The model detects 4 classes:
- `ball` (class 0)
- `goalkeeper` (class 1)
- `player` (class 2)
- `referee` (class 3)

## License

MIT License - see LICENSE file for details

# Soccer AI

Object detection and tracking system for soccer videos using YOLO (ultralytics) + ByteTrack (supervision).

## Tech Stack

- **Python 3** with `ultralytics`, `roboflow`, `supervision`, `tqdm`, `pyyaml`, `trackeval` (for optional tracking metrics)
- **YOLO** for object detection (YOLOv8/v11 models)
- **ByteTrack** (via `supervision`) for multi-object tracking
- **Roboflow** for dataset management

## Classes (4 total)

| ID | Class      |
|----|------------|
| 0  | ball       |
| 1  | goalkeeper |
| 2  | player     |
| 3  | referee    |

## Project Layout

```
soccer_ai/
├── train.py              # Training entry point (replaces Kaggle notebook)
├── inference.py           # Inference entry point (replaces Colab notebook)
├── merge_datasets.py      # Dataset merging utility
├── eval.py                # Evaluate predictions JSON vs ground-truth JSON
├── eval_sets/             # Versioned eval split manifests + GT schema notes
├── scripts/               # prepare_soccer_net_eval.py (SoccerNet-v3 → clip + GT JSON)
├── requirements.txt       # Dependencies
└── src/
    ├── data/
    │   ├── dataset.py         # Download, fix YAML, remap labels
    │   └── dataset_merger.py  # Merge ball-only + full datasets
    ├── models/
    │   └── trainer.py         # YOLO training wrapper
    ├── inference/
    │   └── video_processor.py # VideoProcessor class + AnnotatorConfig
    ├── eval/                  # mAP, ball metrics, optional MOT via TrackEval
    └── utils/                 # (reserved for future utilities)
```

## Entry Points

- `python train.py --help` — Train a model (downloads from Roboflow, remaps labels, trains)
- `python inference.py --help` — Process a video with detection + tracking
- `python merge_datasets.py --help` — Merge a ball-only dataset with the full 4-class dataset
- `python eval.py --help` — Score `*_detections.json` (or model-only export) against ground-truth JSON

## Label Format

YOLO format: `class_id center_x center_y width height` (normalized 0-1).

Roboflow raw labels use different class IDs that get remapped during training:
- 4 → 0 (ball), 6 → 1 (goalkeeper), 11 → 2 (player), 16 → 3 (referee)

## Environments

- **Training**: Kaggle (GPU) — upload project, `pip install -r requirements.txt`, run `train.py`
- **Inference**: Colab or local — clone repo, install deps, run `inference.py`
- **Development**: Local with Cursor or Claude Code

## Before merging to `main`

Agents and contributors must pass local CI **before** merging or pushing to `main` (matches GitHub Actions):

```bash
pip install -r requirements.txt -r requirements-dev.txt   # once per environment
./scripts/ci_local.sh
```

Do not merge until this exits successfully. See `.cursor/rules/merge-gate-main.mdc`.

## Conventions

- Entry point scripts live in project root, all logic lives in `src/`
- Use `argparse` for CLI arguments — no hardcoded paths or config values
- JSON detection data is auto-exported alongside output videos (`*_detections.json`)
- See `WORKFLOW_GUIDE.md` for full development and deployment workflows
- See `DATASET_MIXING_GUIDE.md` for dataset merging strategies

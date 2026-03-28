# Data Module

Handles dataset download, preprocessing, and merging.

## Key Functions

- `soccer_net_v3.labels_v3_to_eval_gt()` — Converts SoccerNet-v3 `Labels-v3.json` to eval GT JSON (`class_id` 0–3). Used by `scripts/prepare_soccer_net_eval.py`
- `download_dataset()` — Downloads from Roboflow in YOLOv8 format
- `fix_data_yaml()` — Overwrites `data.yaml` to enforce 4 classes: `[ball, goalkeeper, player, referee]`
- `remap_labels()` — Remaps Roboflow class IDs to our 0-3 scheme. Removes labels not in the class map and deletes orphaned images
- `merge_datasets()` — Merges a ball-only dataset with the full 4-class dataset, limiting ball-only ratio to prevent class imbalance

## Class ID Mapping (Roboflow → Ours)

```
4  → 0  (ball)
6  → 1  (goalkeeper)
11 → 2  (player)
16 → 3  (referee)
```

All other Roboflow classes are **dropped** (lines removed from label files).

## Dataset Structure (YOLOv8)

```
dataset/
├── data.yaml
├── train/
│   ├── images/   # .jpg files
│   └── labels/   # .txt files (YOLO format)
├── valid/         # Roboflow uses "valid", not "val"
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Gotchas

- Roboflow names the validation split `valid/`, not `val/` — the code handles this
- `remap_labels()` deletes label files (and their images) if no valid classes remain after remapping
- When merging ball-only datasets: images with visible players but no player labels will teach the model to treat players as background — the `max_ball_only_ratio` parameter (default 0.3) mitigates this

# Eval module

## Ground truth

JSON with `detections[]` per frame and `objects[]` with `bbox` (xyxy pixels), `class_id` 0–3, optional `track_id`. See [eval_sets/README.md](../../eval_sets/README.md).

## Predictions

- **full**: `*_detections.json` from `VideoProcessor` — `tracked_objects` + `ball`; ball boxes are un-padded for metrics.
- **model_only**: same top-level shape but per-frame `objects` only (canonical `class_id`), from `process_video(..., eval_mode="model_only")`.

## Metrics

- **detection**: mAP @0.5, mAP @0.5:0.95 (mean over IoU thresholds), per-class AP, mean ball center distance when IoU≥0.5 on ball.
- **tracking** (`eval.py --tracking`): HOTA, CLEAR (MOTA, …), Identity (IDF1, …) via TrackEval on person classes only, MOTChallenge file layout.

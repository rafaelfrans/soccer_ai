# Evaluation splits (SoccerNet and manual GT)

Use a **versioned manifest** to pin which raw videos and ground-truth JSON files belong to your primary benchmark. That keeps scores comparable across model checkpoints and pipeline changes.

## Ground-truth JSON schema

One file per video, aligned frame-by-frame with inference output (`frame_number` matches `VideoProcessor` / `*_detections.json`).

```json
{
  "video_info": {
    "width": 1920,
    "height": 1080,
    "total_frames": 300
  },
  "detections": [
    {
      "frame_number": 0,
      "objects": [
        {
          "bbox": [x1, y1, x2, y2],
          "class_id": 0,
          "track_id": 1
        }
      ]
    }
  ]
}
```

- `class_id`: `0` ball, `1` goalkeeper, `2` player, `3` referee (same as training / AGENTS.md).
- `track_id`: optional; required only if you run `python eval.py --tracking` (HOTA / MOTA / IDF1 on **person** classes). Use stable IDs per person across frames.
- Ball boxes should be **tight**; the eval un-pads full-pipeline predictions by 10px before IoU.

## SoccerNet

SoccerNet is a family of [benchmarks and datasets](https://www.soccer-net.org/) for broadcast soccer. It does not ship this repo’s four-class labels directly; typical workflow is:

1. Take clips or frames from a SoccerNet split that allows your use case (check the license for each download).
2. Remap any public labels (or add your own) to `class_id` 0–3.
3. List videos and GT paths in a manifest YAML (see `example_manifest.yaml`).
4. Run full inference on the **same** source resolution, then `eval.py`.

Useful references:

- SoccerNet-v2: *SoccerNet-v2: A Dataset and Benchmarks for Holistic Understanding of Broadcast Soccer Videos* ([arXiv:2011.13367](https://arxiv.org/abs/2011.13367)).
- Player/ball detection baselines often report mAP on SoccerNet-derived crops; compare your **system** metrics only after remapping labels and using the same eval protocol.

## Commands

```bash
# Full pipeline vs GT (ball boxes un-padded automatically when mode=full)
python eval.py -g path/to/gt.json -p path/to/out_detections.json --mode full

# Raw YOLO vs same GT (export with: inference.py --eval-mode model_only --json-only ...)
python eval.py -g path/to/gt.json -p path/to/out_model_only_detections.json --mode model_only

# Tracking (needs track_id in GT and tracker_id in full-pipeline preds)
python eval.py -g gt.json -p pred.json --mode full --tracking
```

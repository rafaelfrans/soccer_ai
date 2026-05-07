# Evaluation splits (SoccerNet and manual GT)

## Easiest path: SoccerNet-v3 → your model → numbers

**SoccerNet does not hand you a single “GT video + JSON” file for the v3 frame split.** For each game you get:

- **`Labels-v3.json`** — all bounding boxes, classes, and image metadata, in evaluation order.
- **`Frames-v3.zip`** — the still images those labels refer to (action + replay frames), not a pre-encoded MP4.

To compare your detector/tracker to those labels, this repo **stitches the images into `clip.mp4` in the same order** as the labels and writes **`gt.json`** in the [schema below](#ground-truth-json-schema). One command (after `pip install -r requirements-optional.txt` for download mode, and **ffmpeg** on your PATH):

```bash
python scripts/soccer_net_benchmark.py \
  --model-path /path/to/best.pt \
  --work-dir data/sn_bench \
  --download --soccer-net-root data/SoccerNet --split test --game-index 0
```

That writes `clip.mp4`, `gt.json`, runs `inference.py`, and saves `eval_report.json` in `--work-dir`. If `No module named 'SoccerNet'` appears after `pip install`, you installed into a **different** Python than the one running the script — use `python3 -m pip install -r requirements-optional.txt` (same `python3` you use to run the benchmark). If you already have `Labels-v3.json` and `Frames-v3.zip` (or an extracted frames folder) on disk:

```bash
python scripts/soccer_net_benchmark.py \
  --model-path /path/to/best.pt \
  --work-dir data/sn_bench \
  --from-files --labels /path/to/Labels-v3.json --frames /path/to/Frames-v3.zip
```

Offline / CI-style smoke test (no network): `--sample`. To only run inference + eval on an existing prepared folder: `--skip-prepare`. Tracking metrics (HOTA / MOTA / IDF1): add `--tracking` if your GT includes jersey `track_id`s (SoccerNet maps numeric jersey IDs when present).

### Download only `Labels-v3.json` + `Frames-v3.zip` (no clip yet)

If you only want the official SoccerNet files on disk (smaller step than building `clip.mp4`), use:

```bash
python3 -m pip install -r requirements-optional.txt
python3 scripts/download_soccer_net_v3_game.py \
  --soccer-net-root data/SoccerNet \
  --split test \
  --game-index 0
```

It prints absolute paths to **`Labels-v3.json`** and **`Frames-v3.zip`**. Repeat with `--game-index 1` for a second benchmark clip.

Put **`soccer-net-root` on a local SSD folder** (avoid iCloud-synced-only paths) so opening large zips does not time out. Then build eval assets:

```bash
python3 scripts/prepare_soccer_net_eval.py \
  --labels <printed Labels-v3.json> \
  --frames <printed Frames-v3.zip> \
  --output-dir data/sn_clip_a
```

We cannot download dataset files for you from this assistant — run the commands above on your machine (subject to SoccerNet’s license and terms).

---

## What counts as ground truth

- **Human / official labels** — e.g. SoccerNet-v3 `Labels-v3.json` (converted to `gt.json` by `scripts/prepare_soccer_net_eval.py`), or your own CVAT/Roboflow exports. The eval pipeline compares **predicted JSON** to this.
- **Not GT** — raw clips under `testing/test_videos/` (e.g. `gunn1s.mp4`) have **no** bundled per-frame boxes in this repo. Inference alone on those files cannot yield a meaningful `eval.py` score without a separate label file.

## SoccerNet: broadcast video vs v3 frames

Per [SoccerNet Data](https://www.soccer-net.org/data), **full broadcast `.mkv` videos** (500+ games) require completing their **NDA** flow. That is separate from the **SoccerNet-v3** “action and replay images” release used here: **`Labels-v3.json` + `Frames-v3.zip`** per game (human-drawn boxes on those frames), downloaded via the **`SoccerNet`** pip package.

Use a **versioned manifest** to pin which clips and `gt.json` files belong to your benchmark.

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

## SoccerNet-v3 automated prep

`soccer_ai` includes a helper that downloads **one** SoccerNet-v3 game (`Labels-v3.json` + `Frames-v3.zip`), converts labels to this repo’s GT JSON, builds a **clip MP4** (ffmpeg), and prints paths for inference + eval.

**Dependencies (not in base `requirements.txt`):** `pip install -r requirements-optional.txt`, **ffmpeg** on `PATH`, and network access to the SoccerNet host.

If download fails with **SSL certificate** errors (common behind some proxies), pass **`--insecure-ssl`** (disables cert verification for that run only).

**Note:** `--max-frames N` caps at **N** but cannot exceed the number of labeled frames in that game’s annotation order (a short game may have fewer than N images).

**Offline smoke test (no download, 2-frame toy clip):**

```bash
python scripts/prepare_soccer_net_eval.py --sample --output-dir data/sn_eval_sample
```

**Or** use only local files (no download this step; same output as above):

```bash
python scripts/prepare_soccer_net_eval.py \
  --labels /path/to/Labels-v3.json \
  --frames /path/to/Frames-v3.zip \
  --output-dir data/sn_eval_clip
# --frames can also be a directory of extracted images (basenames must match the label files).
```

**One-shot** prep + inference + `eval.py` (writes `eval_report.json` in the work directory):

```bash
python scripts/soccer_net_benchmark.py --model-path YOUR.pt --work-dir data/sn_bench --download --soccer-net-root data/SoccerNet
```

**Real data (one game from the frames split, with download):**

```bash
python scripts/prepare_soccer_net_eval.py --soccer-net-root data/SoccerNet --output-dir data/sn_eval_clip --split test --game-index 0
# Optional: --max-frames 120  for a shorter MP4
```

Then run your model and eval:

```bash
python inference.py --model-path YOUR.pt --source data/sn_eval_clip/clip.mp4 --output data/sn_eval_clip/out.mp4 --json-only
python eval.py -g data/sn_eval_clip/gt.json -p data/sn_eval_clip/out_detections.json --mode full
```

Conversion maps SoccerNet-v3 box classes (ball, team players, goalkeepers, referees) to `class_id` 0–3; field lines, goals, staff, etc. are **skipped**. Jersey `ID` becomes `track_id` when numeric.

## SoccerNet (general)

SoccerNet is a family of [benchmarks and datasets](https://www.soccer-net.org/) for broadcast soccer. Licenses vary by split; cite the dataset papers when publishing.

Useful references:

- SoccerNet-v2: *SoccerNet-v2: A Dataset and Benchmarks for Holistic Understanding of Broadcast Soccer Videos* ([arXiv:2011.13367](https://arxiv.org/abs/2011.13367)).
- SoccerNet-v3 dataloader and labels: [SoccerNet-v3](https://github.com/SoccerNet/SoccerNet-v3) (bounding boxes on action/replay frames).

## Commands

```bash
# Full pipeline vs GT (ball boxes un-padded automatically when mode=full)
python eval.py -g path/to/gt.json -p path/to/out_detections.json --mode full

# Raw YOLO vs same GT (export with: inference.py --eval-mode model_only --json-only ...)
python eval.py -g path/to/gt.json -p path/to/out_model_only_detections.json --mode model_only

# Tracking (needs track_id in GT and tracker_id in full-pipeline preds)
python eval.py -g gt.json -p pred.json --mode full --tracking
```

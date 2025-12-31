# Using Soccer AI on Kaggle

## Kaggle File System Structure

Kaggle has a different file system than Colab:

```
/kaggle/
├── input/          # Read-only input datasets (you add these)
├── working/        # Read-write output directory (your workspace)
└── temp/          # Temporary files
```

**Key differences from Colab:**
- ❌ No Google Drive mounting
- ✅ Use `/kaggle/working/` for outputs
- ✅ Use `/kaggle/input/` for input datasets

---

## Training on Kaggle

### Step 1: Upload Your Project

**Option A: Upload as Dataset (Recommended)**
1. Zip your `soccer_ai` project folder
2. Go to Kaggle → Datasets → New Dataset
3. Upload the zip file
4. Add it as input to your notebook

**Option B: Clone from GitHub**
```python
# In Kaggle notebook:
!git clone https://github.com/YOUR_USERNAME/soccer_ai.git
%cd soccer_ai
!pip install -r requirements.txt
```

### Step 2: Upload Your Model

Upload your pretrained model as a Kaggle dataset:
1. Go to Kaggle → Datasets → New Dataset
2. Upload your `best.pt` file
3. Add it as input to your notebook

It will be available at: `/kaggle/input/your-model-dataset/best.pt`

### Step 3: Run Training

**Option A: Download from Roboflow (Recommended - No need to upload dataset!)**

If your dataset is on Roboflow, just download it directly:

```python
# In Kaggle notebook:
!python train.py \
  --roboflow-api-key YOUR_ROBOFLOW_API_KEY \
  --model-path /kaggle/input/your-model-dataset/best.pt \
  --project /kaggle/working/soccerai_training \
  --name kaggle-run-1 \
  --epochs 35 \
  --batch 10
```

**What happens:**
- Automatically downloads dataset from Roboflow to `/kaggle/working/`
- Automatically fixes data.yaml and remaps labels
- Trains the model
- Saves results to `/kaggle/working/soccerai_training/kaggle-run-1/`

**No need to upload the dataset!** The script downloads it automatically.

**Option B: Use Pre-uploaded Dataset**

If you already uploaded the dataset to Kaggle:

```python
!python train.py \
  --skip-download \
  --dataset-path /kaggle/input/your-dataset/soccer-players-1 \
  --model-path /kaggle/input/your-model-dataset/best.pt \
  --project /kaggle/working/soccerai_training \
  --name kaggle-run-1 \
  --epochs 35 \
  --batch 10
```

**Important:** Use `/kaggle/working/` for `--project` because:
- `/kaggle/working/` is read-write (you can save files)
- `/kaggle/input/` is read-only (you can't save there)

### Step 4: Download Results

After training completes:

**Option A: Download from Kaggle UI**
1. Go to your notebook
2. Click "Output" tab on the right
3. Download the files you need:
   - `/kaggle/working/soccerai_training/kaggle-run-1/weights/best.pt`
   - Results, plots, etc.

**Option B: Save to Kaggle Dataset**
```python
# Create a new dataset with your trained model
# This makes it available for future notebooks
```

---

## Complete Kaggle Workflow Example

### Using Roboflow Download (Recommended)

```python
# Cell 1: Setup
!git clone https://github.com/YOUR_USERNAME/soccer_ai.git
%cd soccer_ai
!pip install -r requirements.txt

# Cell 2: Train (downloads dataset automatically from Roboflow)
!python train.py \
  --roboflow-api-key YOUR_ROBOFLOW_API_KEY \
  --model-path /kaggle/input/pretrained-models/dec27-best.pt \
  --project /kaggle/working/soccerai_training \
  --name kaggle-training-$(date +%Y%m%d) \
  --epochs 35 \
  --batch 10 \
  --device 0

# Cell 3: Check results
!ls -lh /kaggle/working/soccerai_training/*/weights/

# Cell 4: The best model is in Output tab:
# /kaggle/working/soccerai_training/kaggle-training-*/weights/best.pt
```

### Using Pre-uploaded Dataset

```python
# Cell 1: Setup
!git clone https://github.com/YOUR_USERNAME/soccer_ai.git
%cd soccer_ai
!pip install -r requirements.txt

# Cell 2: Train (using uploaded dataset)
!python train.py \
  --skip-download \
  --dataset-path /kaggle/input/soccer-players-dataset/soccer-players-1 \
  --model-path /kaggle/input/pretrained-models/dec27-best.pt \
  --project /kaggle/working/soccerai_training \
  --name kaggle-training-$(date +%Y%m%d) \
  --epochs 35 \
  --batch 10 \
  --device 0

# Cell 3: Check results
!ls -lh /kaggle/working/soccerai_training/*/weights/
```

---

## Path Reference for Kaggle

### Input Paths (Read-Only)
```
/kaggle/input/
├── your-dataset/           # Your Roboflow dataset (if uploaded)
│   └── soccer-players-1/
├── your-model/             # Your pretrained model
│   └── best.pt
└── other-datasets/         # Other Kaggle datasets you add
```

### Output Paths (Read-Write)
```
/kaggle/working/
├── soccerai_training/      # Your --project path
│   └── kaggle-run-1/      # Your --name
│       ├── weights/
│       │   ├── best.pt
│       │   └── last.pt
│       ├── results.png
│       └── ...
└── other-outputs/          # Any other files you create
```

---

## Tips for Kaggle

1. **Save to `/kaggle/working/`** - This is the only writable directory
2. **Download from Output tab** - After training, download your model from the UI
3. **Use descriptive names** - Include date/time in `--name` to track runs
4. **Check GPU availability** - Make sure GPU is enabled in notebook settings
5. **Monitor output size** - Kaggle has limits on output size

---

## Common Issues

**Q: Can I save to Google Drive from Kaggle?**
A: No, Kaggle doesn't support Google Drive. Use `/kaggle/working/` and download from the Output tab.

**Q: How do I get my trained model off Kaggle?**
A: Download it from the Output tab, or create a Kaggle dataset with it.

**Q: Can I continue training from a previous run?**
A: Yes! Upload your previous `best.pt` as input and use it as `--model-path`.

**Q: Where should I put my dataset?**
A: Upload it as a Kaggle dataset, then it's available at `/kaggle/input/your-dataset-name/`

---

## Example: Full Training Session (Roboflow Download)

```python
# === SETUP ===
!git clone https://github.com/YOUR_USERNAME/soccer_ai.git
%cd soccer_ai
!pip install -r requirements.txt

# === TRAIN (Downloads dataset from Roboflow automatically) ===
!python train.py \
  --roboflow-api-key YOUR_ROBOFLOW_API_KEY \
  --model-path /kaggle/input/pretrained/yolo11n.pt \
  --project /kaggle/working/soccerai_training \
  --name run-$(date +%Y%m%d-%H%M%S) \
  --epochs 35 \
  --batch 10 \
  --imgsz 1280

# === RESULTS ===
# Check what was created
!find /kaggle/working/soccerai_training -name "best.pt"

# The model will be in:
# /kaggle/working/soccerai_training/run-YYYYMMDD-HHMMSS/weights/best.pt
# Download it from the Output tab!

# The dataset was downloaded to:
# /kaggle/working/soccer-players-1/ (or similar, check the output)
```

---

## Summary

**For Kaggle:**
- ✅ Use `/kaggle/working/` for `--project` path
- ✅ Upload datasets to `/kaggle/input/` via Kaggle UI
- ✅ Download results from Output tab
- ❌ No Google Drive mounting

**Example command:**
```bash
--project /kaggle/working/soccerai_training
```


# Understanding Your New Soccer AI System

## What We Did: The Big Picture

### Before (Your Old Workflow)
You had **two separate notebooks**:
1. **Kaggle notebook** (`soccerai-training.ipynb`) - for training your model
2. **Colab notebook** (`soccerai_inference.ipynb`) - for running inference on videos

Each notebook was a big file with all the code mixed together. To use them, you had to:
- Open the notebook
- Run cells one by one
- Manually change paths and settings in the code
- Download/upload files between platforms

### After (Your New System)
We **organized everything into a clean structure**:
- **Modular code** in `src/` folder (organized by purpose)
- **Two simple entry points**: `train.py` and `inference.py`
- **No notebooks needed** - everything runs from the command line

---

## What is an "Entry Point"?

Think of an entry point like a **main door** to your house. Instead of having to know which room has what, you just use the front door and it guides you.

**Entry Point = A simple script you run to do a specific task**

- `train.py` = The entry point for training (replaces your Kaggle notebook)
- `inference.py` = The entry point for inference (replaces your Colab notebook)

You don't need to open notebooks or edit code. Just run these scripts with the right options!

---

## How Your Old Workflow Maps to the New System

### OLD: Training on Kaggle

**What you did:**
1. Open Kaggle notebook
2. Run cells to download dataset from Roboflow
3. Fix the data.yaml file
4. Remap label IDs
5. Train the model
6. Download the `best.pt` file

**NEW: Training (using `train.py`)**

```bash
python train.py \
  --roboflow-api-key YOUR_KEY \
  --model-path /path/to/pretrained/model.pt \
  --batch 10 \
  --epochs 35
```

**What happens:**
- Automatically downloads dataset (same as your notebook)
- Automatically fixes data.yaml (same as your notebook)
- Automatically remaps labels (same as your notebook)
- Trains the model (same as your notebook)
- Saves `best.pt` in the output folder

**All the same steps, but automated!**

---

### OLD: Inference on Colab

**What you did:**
1. Open Colab notebook
2. Mount Google Drive
3. Load your trained model (`best.pt`)
4. Load a video
5. Process each frame (detect objects, track them, annotate)
6. Save annotated video

**NEW: Inference (using `inference.py`)**

```bash
python inference.py \
  --model-path /path/to/best.pt \
  --source /path/to/input/video.mp4 \
  --output /path/to/output/video.mp4
```

**What happens:**
- Loads your trained model
- Processes the video frame by frame
- Detects objects (ball, players, etc.)
- Tracks objects across frames
- Annotates with boxes, labels, and tracking IDs
- Saves the annotated video

**All the same steps, but simpler!**

---

## How to Use the System (Step by Step)

### Step 1: Install Dependencies

First time only:
```bash
pip install -r requirements.txt
```

This installs all the libraries you need (ultralytics, roboflow, supervision, etc.)

---

### Step 2: Training a Model

**On Kaggle (or any machine with GPU):**

```bash
python train.py \
  --roboflow-api-key qyt9IpiieOgN8CHs58x1 \
  --model-path /kaggle/input/models/dec26-best.pt \
  --batch 10 \
  --epochs 35 \
  --imgsz 1280 \
  --device 0
```

**What each part means:**
- `--roboflow-api-key`: Your Roboflow API key (same as before)
- `--model-path`: Path to your starting model (like the one you used in Kaggle)
- `--batch 10`: How many images to process at once (same as before)
- `--epochs 35`: How many training cycles (same as before)
- `--imgsz 1280`: Image size (same as before)
- `--device 0`: Use GPU 0 (same as before)

**Result:** You get a `best.pt` file in `soccerai_training/[run-name]/weights/best.pt`

---

### Step 3: Running Inference

**On your local machine (or Colab, or anywhere):**

```bash
python inference.py \
  --model-path /path/to/best.pt \
  --source /path/to/your/video.mp4 \
  --output /path/to/annotated_video.mp4 \
  --conf 0.3
```

**What each part means:**
- `--model-path`: Path to your trained `best.pt` file
- `--source`: Path to the video you want to process
- `--output`: Where to save the annotated video
- `--conf 0.3`: Confidence threshold (same as your notebook used)

**Result:** An annotated video with detected and tracked objects!

---

## The Code Organization (What's in `src/`?)

Think of `src/` as organized drawers in a filing cabinet:

```
src/
├── data/          → Everything about datasets (download, fix, remap)
├── models/        → Everything about training models
├── inference/     → Everything about processing videos
└── utils/         → Helper functions (currently empty, for future use)
```

**Why organize it this way?**
- Easy to find code
- Easy to modify one part without breaking others
- Easy to reuse code
- Professional structure

---

## Real-World Example: Complete Workflow

### Scenario: Train a new model and test it on a video

**1. Train on Kaggle:**
```bash
# Upload train.py and src/ folder to Kaggle
# In Kaggle notebook or terminal:
python train.py \
  --roboflow-api-key qyt9IpiieOgN8CHs58x1 \
  --model-path yolo11n.pt \
  --name my-new-model
```

**2. Download the best model:**
- Download `soccerai_training/my-new-model/weights/best.pt` from Kaggle

**3. Run inference locally:**
```bash
# On your computer:
python inference.py \
  --model-path ./best.pt \
  --source ./test_video.mp4 \
  --output ./result.mp4
```

**4. Watch your annotated video!**

---

## Key Benefits of the New System

1. **No more notebooks** - Run everything from command line
2. **Reusable** - Same code works on Kaggle, Colab, or your computer
3. **Organized** - Easy to find and modify code
4. **Flexible** - Change settings without editing code (use command-line arguments)
5. **Professional** - Standard Python project structure

---

---

## Using on Colab and Kaggle (Cloud Platforms)

### Important: Colab/Kaggle Reset Every Time!

You're right - Colab and Kaggle reset their runtime every time you disconnect. This means:
- ✅ You need to upload your project files each time
- ✅ You need to install requirements.txt each time
- ✅ Your code changes are lost unless you save them

**Solution:** Upload your entire project folder, not just individual files!

---

### Using Inference on Colab

**Step 1: Upload Your Project**

You need to upload **the entire project**, not just `inference.py`:

```
soccer_ai/
├── inference.py          ← Entry point
├── requirements.txt      ← Dependencies list
└── src/                  ← All your code modules
    ├── __init__.py
    ├── inference/
    │   ├── __init__.py
    │   └── video_processor.py
    └── ... (other modules)
```

**Option A: Upload via Colab UI**
1. Open Colab
2. Click the folder icon (📁) on the left sidebar
3. Click "Upload" button
4. Upload your entire `soccer_ai` folder (or zip it first, then upload and unzip)

**Option B: Clone from GitHub (Recommended!)**
```python
# In Colab cell:
!git clone https://github.com/YOUR_USERNAME/soccer_ai.git
%cd soccer_ai
```

**Option C: Mount Google Drive**
```python
# In Colab cell:
from google.colab import drive
drive.mount('/content/drive')

# Copy your project to Drive, then:
%cd /content/drive/MyDrive/soccer_ai
```

**Step 2: Install Dependencies (Every Time!)**

Since Colab resets, you must install dependencies every session:

```python
# In Colab cell:
!pip install -r requirements.txt
```

**Step 3: Run Inference**

```python
# In Colab cell:
!python inference.py \
  --model-path /content/drive/MyDrive/models/best.pt \
  --source /content/drive/MyDrive/videos/input.mp4 \
  --output /content/drive/MyDrive/videos/output.mp4
```

**Why you need the whole project:**
- `inference.py` imports from `src.inference` → needs `src/` folder
- `src/inference/video_processor.py` uses libraries → needs `requirements.txt` installed
- Everything is connected!

---

### Using Training on Kaggle

**Step 1: Upload Your Project**

Same as Colab - upload the entire project folder:

```
soccer_ai/
├── train.py             ← Entry point
├── requirements.txt      ← Dependencies list
└── src/                  ← All your code modules
    ├── data/
    ├── models/
    └── ...
```

**Option A: Upload as Dataset**
1. Zip your `soccer_ai` folder
2. Go to Kaggle → Datasets → New Dataset
3. Upload the zip file
4. Add it as input to your notebook

**Option B: Upload via Kaggle UI**
1. In Kaggle notebook, click "+ Add data"
2. Upload your files

**Step 2: Install Dependencies**

```python
# In Kaggle notebook cell:
!pip install -r requirements.txt
```

**Step 3: Run Training**

```python
# In Kaggle notebook cell:
!python train.py \
  --roboflow-api-key YOUR_KEY \
  --model-path /kaggle/input/models/pretrained.pt \
  --batch 10 \
  --epochs 35
```

---

## Development Workflow: Editing Code and Version Control

### The Development Cycle

When you want to improve your code, follow this cycle:

```
1. Edit code locally
2. Test it
3. Commit to Git
4. Push to GitHub
5. Pull on Colab/Kaggle to use updated code
```

---

### Step-by-Step Development Workflow

#### Step 1: Edit Code Locally

**On your computer:**

```bash
# Navigate to your project
cd /Users/rafael/Documents/GitHub/soccer_ai

# Edit files in your editor (VS Code, Cursor, etc.)
# For example, edit src/inference/video_processor.py
```

**What can you edit?**
- `src/inference/video_processor.py` - Change how videos are processed
- `src/models/trainer.py` - Change training logic
- `src/data/dataset.py` - Change data preprocessing
- `train.py` or `inference.py` - Add new command-line arguments

#### Step 2: Test Your Changes

**Test locally first:**

```bash
# Test inference locally (if you have a model):
python inference.py \
  --model-path ./test_model.pt \
  --source ./test_video.mp4 \
  --output ./test_output.mp4

# Or test training (if you have a small dataset):
python train.py --help  # Check if arguments work
```

#### Step 3: Check What Changed

```bash
# See what files you modified:
git status

# See the actual changes:
git diff
```

#### Step 4: Commit Your Changes

**Commit = Save a snapshot of your code**

```bash
# Stage your changes (tell Git what to save):
git add .

# Or stage specific files:
git add src/inference/video_processor.py

# Commit with a descriptive message:
git commit -m "Improve video processing: add better tracking"

# Good commit messages explain WHAT and WHY:
# "Fix bug: handle empty frames in video processing"
# "Add feature: support for custom annotation colors"
# "Refactor: simplify dataset download logic"
```

**Commit message best practices:**
- ✅ "Fix: handle edge case when no detections found"
- ✅ "Add: support for batch video processing"
- ✅ "Update: improve tracking accuracy"
- ❌ "changes" (too vague)
- ❌ "fix stuff" (not descriptive)

#### Step 5: Push to GitHub

**Push = Upload your commits to GitHub**

```bash
# Push to your branch:
git push origin rafael/initial-commit

# Or if it's your first push:
git push -u origin rafael/initial-commit
```

**What happens:**
- Your code is now on GitHub
- You can access it from anywhere
- Others can see your changes (if repo is public)

#### Step 6: Use Updated Code on Colab/Kaggle

**On Colab:**

```python
# Pull latest code from GitHub:
!git clone https://github.com/YOUR_USERNAME/soccer_ai.git
# Or if already cloned:
%cd soccer_ai
!git pull

# Install dependencies:
!pip install -r requirements.txt

# Run with your updated code:
!python inference.py --model-path ... --source ... --output ...
```

**On Kaggle:**

```python
# Add your GitHub repo as input, or:
!git clone https://github.com/YOUR_USERNAME/soccer_ai.git
%cd soccer_ai
!pip install -r requirements.txt
!python train.py ...
```

---

### Git Basics Cheat Sheet

**Check status:**
```bash
git status                    # What files changed?
```

**See changes:**
```bash
git diff                      # Show all changes
git diff src/inference/       # Show changes in specific folder
```

**Save changes:**
```bash
git add .                     # Stage all changes
git add file.py               # Stage specific file
git commit -m "Description"    # Save snapshot
```

**Upload to GitHub:**
```bash
git push origin branch-name   # Upload commits
```

**Download from GitHub:**
```bash
git pull                      # Get latest changes
```

**Create new branch (for experiments):**
```bash
git checkout -b new-feature   # Create and switch to new branch
# Make changes, commit, push
git push origin new-feature
```

**Switch branches:**
```bash
git checkout rafael/initial-commit  # Switch to existing branch
```

---

### Common Development Scenarios

#### Scenario 1: Fix a Bug

```bash
# 1. Edit the buggy file
# 2. Test it
git add src/inference/video_processor.py
git commit -m "Fix: handle None detections in video processing"
git push origin rafael/initial-commit
```

#### Scenario 2: Add a New Feature

```bash
# 1. Edit files to add feature
# 2. Test it
git add .
git commit -m "Add: support for custom confidence thresholds per class"
git push origin rafael/initial-commit
```

#### Scenario 3: Experiment Safely

```bash
# Create experimental branch:
git checkout -b experiment/new-tracking

# Make changes, test
git add .
git commit -m "Experiment: try different tracking algorithm"

# If it works, merge back:
git checkout rafael/initial-commit
git merge experiment/new-tracking

# If it doesn't work, just switch back:
git checkout rafael/initial-commit
# (experiment branch stays there for reference)
```

---

### File Organization for Cloud Platforms

**What to upload to Colab/Kaggle:**

```
✅ Upload these:
├── inference.py          (or train.py)
├── requirements.txt
└── src/                  (entire folder!)
    ├── __init__.py
    ├── inference/
    ├── models/
    └── data/

❌ Don't need these:
├── .git/                 (Git history - too big)
├── LICENSE               (not needed for running)
├── README.md             (nice but not required)
└── WORKFLOW_GUIDE.md     (nice but not required)
```

**Quick upload script for Colab:**

```python
# In Colab, create a cell with:
import os
from google.colab import drive

drive.mount('/content/drive')

# Copy your project (if it's on Drive):
!cp -r /content/drive/MyDrive/soccer_ai /content/
%cd /content/soccer_ai

# Or clone from GitHub:
!git clone https://github.com/YOUR_USERNAME/soccer_ai.git
%cd soccer_ai

# Always install dependencies:
!pip install -r requirements.txt
```

---

## Common Questions

**Q: Can I still use notebooks?**
A: The notebooks are still in your Downloads folder if you need them, but the new system is better!

**Q: Do I need to change my workflow?**
A: Not really! You still:
- Train on Kaggle (just use `train.py` instead of notebook)
- Run inference (just use `inference.py` instead of notebook)
- Download/upload models the same way

**Q: What if I want to customize something?**
A: Edit the code in `src/` folders, or add command-line arguments to `train.py` or `inference.py`

**Q: Can I run this on my computer?**
A: Yes! As long as you have Python and install dependencies. For training, you'll want a GPU (Kaggle/Colab). For inference, CPU works but GPU is faster.

**Q: Why do I need to upload the whole project to Colab?**
A: Because `inference.py` imports from `src.inference`, which imports other modules. Python needs the entire folder structure to find all the code.

**Q: Do I need to install requirements.txt every time on Colab?**
A: Yes! Colab resets the environment each time. Always run `!pip install -r requirements.txt` at the start of each session.

**Q: How do I keep my code in sync between local and Colab?**
A: Use Git! Edit locally → commit → push to GitHub → pull on Colab. This keeps everything synchronized.

**Q: What if I make a mistake in a commit?**
A: You can fix it! `git commit --amend` to change the last commit, or make a new commit that fixes the issue.

**Q: Should I commit after every small change?**
A: It's better to commit logical units of work. For example: "Fix bug" or "Add feature" - not every single line edit.

---

## Summary

**Old way:** Notebooks → Run cells → Manually change code → Download files → Lose changes on reset

**New way:** 
- Edit code locally → Test → Commit → Push to GitHub
- On Colab/Kaggle: Clone from GitHub → Install requirements → Run scripts
- Keep everything synchronized with Git!

Everything you did before still works, just organized, version-controlled, and easier to maintain! 🎉


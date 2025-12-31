# Quick Reference Card

## 🚀 Running Inference on Colab

```python
# 1. Clone or upload your project
!git clone https://github.com/YOUR_USERNAME/soccer_ai.git
%cd soccer_ai

# 2. Install dependencies (ALWAYS do this!)
!pip install -r requirements.txt

# 3. Run inference
!python inference.py \
  --model-path /content/drive/MyDrive/models/best.pt \
  --source /content/drive/MyDrive/videos/input.mp4 \
  --output /content/drive/MyDrive/videos/output.mp4
```

## 🏋️ Running Training on Kaggle

```python
# 1. Upload entire project folder (or clone from GitHub)
# 2. Install dependencies
!pip install -r requirements.txt

# 3. Run training
!python train.py \
  --roboflow-api-key YOUR_KEY \
  --model-path /kaggle/input/models/pretrained.pt \
  --batch 10 \
  --epochs 35
  --
```

## 💻 Development Workflow

```bash
# 1. Edit code locally
# 2. Test it
python inference.py --help

# 3. Check what changed
git status
git diff

# 4. Commit changes
git add .
git commit -m "Description of changes"
git push origin rafael/initial-commit

# 5. On Colab/Kaggle: Pull latest
!git pull
```

## 📦 What Files to Upload

**✅ Upload entire project:**
- `inference.py` or `train.py`
- `requirements.txt`
- `src/` folder (entire folder!)

**❌ Don't need:**
- `.git/` folder
- Documentation files (unless you want them)

## 🔑 Key Points

- **Colab/Kaggle reset** → Always install `requirements.txt` each session
- **Need entire project** → Not just the entry point script
- **Use Git** → Keep code synced between local and cloud
- **Commit often** → Save your work with descriptive messages


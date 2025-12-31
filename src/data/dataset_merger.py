"""Dataset merging utilities for combining multiple datasets."""

import os
import shutil
import yaml
import random
from pathlib import Path
from typing import Tuple, Optional
import glob


def merge_datasets(
    primary_dataset_path: str,
    ball_only_dataset_path: str,
    output_path: str,
    ball_class_id: int = 0,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
    max_ball_only_ratio: float = 0.3
) -> str:
    """
    Merge a ball-only dataset with a full 4-class dataset.
    
    This function:
    1. Keeps all annotations from primary dataset (4 classes)
    2. Extracts only ball annotations from ball-only dataset
    3. Merges images and labels
    4. Creates new train/val/test splits
    
    Args:
        primary_dataset_path: Path to your existing 4-class dataset
        ball_only_dataset_path: Path to ball-only dataset
        output_path: Where to save merged dataset
        ball_class_id: Class ID for ball (default: 0)
        train_split: Training split ratio (default: 0.8)
        val_split: Validation split ratio (default: 0.1)
        test_split: Test split ratio (default: 0.1)
        seed: Random seed for reproducibility
        max_ball_only_ratio: Maximum ratio of ball-only images to total (default: 0.3)
    
    Returns:
        Path to merged dataset
    """
    random.seed(seed)
    
    # Validate splits
    assert abs(train_split + val_split + test_split - 1.0) < 0.01, "Splits must sum to 1.0"
    
    # Create output directory structure
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Load primary dataset
    print("📦 Loading primary dataset...")
    primary_images = {}
    primary_labels = {}
    
    for split in ['train', 'val', 'test']:
        primary_images[split] = list(Path(primary_dataset_path).glob(f'{split}/images/*'))
        primary_labels[split] = {}
        for img_path in primary_images[split]:
            label_path = img_path.parent.parent / 'labels' / (img_path.stem + '.txt')
            if label_path.exists():
                primary_labels[split][img_path.name] = label_path
    
    # Load ball-only dataset
    print("📦 Loading ball-only dataset...")
    ball_only_images = []
    ball_only_labels = {}
    
    # Try to find images in common structure
    possible_paths = [
        Path(ball_only_dataset_path) / 'images',
        Path(ball_only_dataset_path) / 'train' / 'images',
        Path(ball_only_dataset_path),
    ]
    
    ball_images_dir = None
    for path in possible_paths:
        if path.exists() and any(path.glob('*.jpg')) or any(path.glob('*.png')):
            ball_images_dir = path
            break
    
    if ball_images_dir is None:
        raise ValueError(f"Could not find images in ball-only dataset at {ball_only_dataset_path}")
    
    # Find all images
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        ball_only_images.extend(ball_images_dir.glob(ext))
        ball_only_images.extend(ball_images_dir.parent.glob(f'**/{ext}'))
    
    # Find corresponding labels
    possible_label_dirs = [
        Path(ball_only_dataset_path) / 'labels',
        Path(ball_only_dataset_path) / 'train' / 'labels',
        ball_images_dir.parent / 'labels',
    ]
    
    ball_labels_dir = None
    for path in possible_label_dirs:
        if path.exists() and any(path.glob('*.txt')):
            ball_labels_dir = path
            break
    
    if ball_labels_dir:
        for img_path in ball_only_images:
            label_path = ball_labels_dir / (img_path.stem + '.txt')
            if label_path.exists():
                # Extract only ball annotations
                ball_annotations = _extract_ball_annotations(label_path, ball_class_id)
                if ball_annotations:  # Only keep if has ball annotations
                    ball_only_labels[img_path.name] = ball_annotations
    
    # Limit ball-only images to maintain balance
    total_primary = sum(len(primary_images[s]) for s in ['train', 'val', 'test'])
    max_ball_only = int(total_primary * max_ball_only_ratio)
    
    if len(ball_only_images) > max_ball_only:
        print(f"⚠️  Limiting ball-only images from {len(ball_only_images)} to {max_ball_only} to maintain balance")
        ball_only_images = random.sample(ball_only_images, max_ball_only)
        # Update labels dict
        ball_only_labels = {img.name: ball_only_labels.get(img.name, []) 
                           for img in ball_only_images if img.name in ball_only_labels}
    
    # Merge datasets
    print("🔄 Merging datasets...")
    
    # Copy primary dataset
    for split in ['train', 'val', 'test']:
        for img_path in primary_images[split]:
            # Copy image
            shutil.copy2(img_path, output_path / split / 'images' / img_path.name)
            
            # Copy label
            if img_path.name in primary_labels[split]:
                shutil.copy2(
                    primary_labels[split][img_path.name],
                    output_path / split / 'labels' / (img_path.stem + '.txt')
                )
    
    # Add ball-only images to training set (or distribute across splits)
    ball_only_list = list(ball_only_images)
    random.shuffle(ball_only_list)
    
    train_count = int(len(ball_only_list) * train_split)
    val_count = int(len(ball_only_list) * val_split)
    
    for i, img_path in enumerate(ball_only_list):
        if i < train_count:
            split = 'train'
        elif i < train_count + val_count:
            split = 'val'
        else:
            split = 'test'
        
        # Copy image
        shutil.copy2(img_path, output_path / split / 'images' / img_path.name)
        
        # Create label file with ball annotations only
        if img_path.name in ball_only_labels:
            label_path = output_path / split / 'labels' / (img_path.stem + '.txt')
            with open(label_path, 'w') as f:
                f.writelines(ball_only_labels[img_path.name])
        else:
            # Create empty label file (image with no annotations)
            label_path = output_path / split / 'labels' / (img_path.stem + '.txt')
            label_path.touch()
    
    # Create data.yaml
    print("📝 Creating data.yaml...")
    data_yaml = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 4,
        'names': ['ball', 'goalkeeper', 'player', 'referee']
    }
    
    with open(output_path / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"✅ Merged dataset saved to: {output_path}")
    print(f"📊 Statistics:")
    print(f"   - Primary dataset: {total_primary} images")
    print(f"   - Ball-only dataset: {len(ball_only_list)} images")
    print(f"   - Total: {total_primary + len(ball_only_list)} images")
    
    return str(output_path)


def _extract_ball_annotations(label_path: Path, ball_class_id: int) -> list:
    """
    Extract only ball annotations from a label file.
    
    Args:
        label_path: Path to label file
        ball_class_id: Class ID for ball
    
    Returns:
        List of ball annotation lines
    """
    ball_annotations = []
    
    if not label_path.exists():
        return ball_annotations
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            # Check if this is a ball annotation
            # Ball-only dataset might have ball as class 0 or different ID
            # We'll accept any annotation and map it to ball (class 0)
            if len(parts) >= 5:  # Valid YOLO format: class x y w h
                # Map to ball class (class 0)
                parts[0] = str(ball_class_id)
                ball_annotations.append(' '.join(parts) + '\n')
    
    return ball_annotations


#!/usr/bin/env python3
"""
Utility script to merge ball-only dataset with your existing 4-class dataset.
"""

import argparse
from src.data import merge_datasets


def main():
    parser = argparse.ArgumentParser(
        description="Merge ball-only dataset with existing 4-class dataset"
    )
    
    parser.add_argument(
        "--primary-dataset",
        type=str,
        required=True,
        help="Path to your existing 4-class dataset (from Roboflow)"
    )
    
    parser.add_argument(
        "--ball-only-dataset",
        type=str,
        required=True,
        help="Path to ball-only dataset (from Kaggle or other source)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path where merged dataset will be saved"
    )
    
    parser.add_argument(
        "--ball-class-id",
        type=int,
        default=0,
        help="Class ID for ball in your system (default: 0)"
    )
    
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Training split ratio (default: 0.8)"
    )
    
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
    )
    
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Test split ratio (default: 0.1)"
    )
    
    parser.add_argument(
        "--max-ball-ratio",
        type=float,
        default=0.3,
        help="Maximum ratio of ball-only images to total (default: 0.3)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    print("🔄 Merging datasets...")
    print(f"   Primary dataset: {args.primary_dataset}")
    print(f"   Ball-only dataset: {args.ball_only_dataset}")
    print(f"   Output: {args.output}")
    
    output_path = merge_datasets(
        primary_dataset_path=args.primary_dataset,
        ball_only_dataset_path=args.ball_only_dataset,
        output_path=args.output,
        ball_class_id=args.ball_class_id,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        max_ball_only_ratio=args.max_ball_ratio
    )
    
    print(f"\n✅ Merged dataset created at: {output_path}")
    print(f"\n📝 Next steps:")
    print(f"   1. Train on merged dataset:")
    print(f"      python train.py --skip-download --dataset-path {output_path} --model-path YOUR_MODEL.pt")


if __name__ == "__main__":
    main()


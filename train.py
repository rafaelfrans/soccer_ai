#!/usr/bin/env python3
"""
Training entry point for soccer AI model.
"""

import os
import argparse
from pathlib import Path
from src.data import download_dataset, fix_data_yaml, remap_labels
from src.models import train_model


def main():
    parser = argparse.ArgumentParser(description="Train soccer AI object detection model")
    
    # Dataset arguments
    parser.add_argument("--roboflow-api-key", type=str, required=True,
                        help="Roboflow API key")
    parser.add_argument("--roboflow-workspace", type=str, default="soccer-ai-lkex8",
                        help="Roboflow workspace name")
    parser.add_argument("--roboflow-project", type=str, default="soccer-players-xy9vk-ebc0t",
                        help="Roboflow project name")
    parser.add_argument("--roboflow-version", type=int, default=1,
                        help="Dataset version number")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to pretrained model or model config")
    parser.add_argument("--batch", type=int, default=10,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=35,
                        help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size")
    parser.add_argument("--device", type=int, default=0,
                        help="Device ID (0 for GPU, cpu for CPU)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    
    # Fine-tuning arguments
    parser.add_argument("--freeze", type=int, default=None,
                        help="Number of layers to freeze (e.g., 24 freezes entire backbone for fine-tuning detection layers)")
    parser.add_argument("--lr0", type=float, default=None,
                        help="Initial learning rate (e.g., 1e-4 for fine-tuning on small datasets)")
    parser.add_argument("--lrf", type=float, default=None,
                        help="Final learning rate factor (final_lr = lr0 * lrf, e.g., 0.01)")
    
    # Output arguments
    parser.add_argument("--project", type=str, default="soccerai_training",
                        help="Project directory path or name. Examples: '/kaggle/working/soccerai_training' (Kaggle), '/content/drive/MyDrive/soccer_ai/training' (Colab), or just 'soccerai_training' (current dir)")
    parser.add_argument("--name", type=str, default=None,
                        help="Run name")
    
    # Data preprocessing
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download (use existing dataset)")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Path to existing dataset (if skip-download)")
    
    args = parser.parse_args()
    
    # Download dataset
    if args.skip_download:
        if args.dataset_path is None:
            raise ValueError("--dataset-path required when using --skip-download")
        dataset_location = args.dataset_path
    else:
        print("📦 Downloading dataset...")
        dataset_location = download_dataset(
            api_key=args.roboflow_api_key,
            workspace=args.roboflow_workspace,
            project=args.roboflow_project,
            version=args.roboflow_version
        )
        print(f"✅ Dataset downloaded to: {dataset_location}")
    
    # Fix data.yaml
    print("\n🔧 Fixing data.yaml...")
    yaml_path = f'{dataset_location}/data.yaml'
    fix_data_yaml(yaml_path)
    print("✅ data.yaml fixed to 4 classes")
    
    # Remap labels
    print("\n🔄 Remapping label class IDs...")
    class_map = {
        4: 0,   # ball
        6: 1,   # goalkeeper
        11: 2,  # player
        16: 3   # referee
    }
    
    train_r, train_rm = remap_labels(f'{dataset_location}/train/labels', class_map)
    valid_r, valid_rm = remap_labels(f'{dataset_location}/valid/labels', class_map)
    test_r, test_rm = remap_labels(f'{dataset_location}/test/labels', class_map)
    
    print(f"✅ Train: {train_r} remapped, {train_rm} removed")
    print(f"✅ Valid: {valid_r} remapped, {valid_rm} removed")
    print(f"✅ Test: {test_r} remapped, {test_rm} removed")
    
    # Train model
    print(f"\n🤖 Using model: {args.model_path}")
    print("\n🚀 Starting training...")
    
    results_path = train_model(
        model_path=args.model_path,
        data_yaml=yaml_path,
        batch=args.batch,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        plots=True,
        freeze=args.freeze,
        lr0=args.lr0,
        lrf=args.lrf
    )
    
    print("\n✅ Training complete!")
    print(f"📁 Results saved to: {results_path}")
    print(f"⬇️  Best model: {results_path}/weights/best.pt")


if __name__ == "__main__":
    main()


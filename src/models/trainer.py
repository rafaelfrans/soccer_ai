"""Model training utilities."""

from ultralytics import YOLO
from pathlib import Path
from typing import Optional


def train_model(
    model_path: str,
    data_yaml: str,
    batch: int = 10,
    epochs: int = 35,
    imgsz: int = 640,
    device: int = 0,
    patience: int = 5,
    project: str = "soccerai_training",
    name: Optional[str] = None,
    plots: bool = True,
    freeze: Optional[int] = None,
    lr0: Optional[float] = None,
    lrf: Optional[float] = None
) -> str:
    """
    Train YOLO model using ultralytics.
    
    Args:
        model_path: Path to pretrained model or model config
        data_yaml: Path to data.yaml file
        batch: Batch size
        epochs: Number of training epochs
        imgsz: Image size
        device: Device ID (0 for GPU, cpu for CPU)
        patience: Early stopping patience
        project: Project directory name
        name: Run name (optional)
        plots: Whether to generate plots
        freeze: Number of layers to freeze (e.g., 10 freezes entire backbone)
        lr0: Initial learning rate (e.g., 1e-4 for fine-tuning)
        lrf: Final learning rate factor (final_lr = lr0 * lrf)
    
    Returns:
        Path to training results directory
    """
    model = YOLO(model_path)
    
    # Build training arguments
    train_args = {
        "data": data_yaml,
        "batch": batch,
        "epochs": epochs,
        "imgsz": imgsz,
        "device": device,
        "patience": patience,
        "project": project,
        "name": name,
        "plots": plots
    }
    
    # Add optional training parameters if provided
    if freeze is not None:
        train_args["freeze"] = freeze
    if lr0 is not None:
        train_args["lr0"] = lr0
    if lrf is not None:
        train_args["lrf"] = lrf
    
    results = model.train(**train_args)
    
    # Return path to results (best model will be in weights/best.pt)
    if name:
        results_path = Path(project) / name
    else:
        # Find the latest run
        project_path = Path(project)
        runs = sorted(project_path.glob("*/"), key=lambda x: x.stat().st_mtime, reverse=True)
        if runs:
            results_path = runs[0]
        else:
            results_path = project_path
    
    return str(results_path)


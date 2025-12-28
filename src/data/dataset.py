"""Dataset download and preprocessing utilities."""

import os
import glob
import yaml
from typing import Dict, Tuple
from roboflow import Roboflow


def download_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int = 1,
    format: str = "yolov8"
) -> str:
    """
    Download dataset from Roboflow.
    
    Args:
        api_key: Roboflow API key
        workspace: Roboflow workspace name
        project: Roboflow project name
        version: Dataset version number
        format: Dataset format (default: yolov8)
    
    Returns:
        Path to downloaded dataset
    """
    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    version_obj = project_obj.version(version)
    dataset = version_obj.download(format)
    
    return dataset.location


def fix_data_yaml(yaml_path: str, num_classes: int = 4, class_names: list = None) -> None:
    """
    Fix data.yaml to set correct number of classes and class names.
    
    Args:
        yaml_path: Path to data.yaml file
        num_classes: Number of classes (default: 4)
        class_names: List of class names (default: ['ball', 'goalkeeper', 'player', 'referee'])
    """
    if class_names is None:
        class_names = ['ball', 'goalkeeper', 'player', 'referee']
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    data['nc'] = num_classes
    data['names'] = class_names
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def remap_labels(
    label_dir: str,
    class_map: Dict[int, int]
) -> Tuple[int, int]:
    """
    Remap label class IDs in YOLO format label files.
    
    Args:
        label_dir: Directory containing label files
        class_map: Dictionary mapping old class IDs to new class IDs
    
    Returns:
        Tuple of (remapped_count, removed_count)
    """
    label_files = glob.glob(f'{label_dir}/*.txt')
    remapped = 0
    removed = 0
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            old_class = int(parts[0])
            if old_class in class_map:
                parts[0] = str(class_map[old_class])
                new_lines.append(' '.join(parts) + '\n')
        
        if new_lines:
            with open(label_file, 'w') as f:
                f.writelines(new_lines)
            remapped += 1
        else:
            # Remove empty label file and corresponding image
            os.remove(label_file)
            image_file = label_file.replace('/labels/', '/images/').replace('.txt', '.jpg')
            if os.path.exists(image_file):
                os.remove(image_file)
            removed += 1
    
    return remapped, removed


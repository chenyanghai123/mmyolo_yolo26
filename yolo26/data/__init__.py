"""Dataset and data pipeline (pure PyTorch).

Annotation format:
    data_path/Annotations/class_names_list.txt  (one class name per line)
    data_path/Annotations/train.csv / val.csv   (columns: image_path, anno_label, anno_json_path)
    annotation json: labelme-style with shapes[{label, bbox:[x1,y1,x2,y2] or points:[[x,y],...]}]
"""
from .dataset import DetDataset, collate_fn
from .transforms import (
    LoadAnnotations,
    JointPad,
    JointResize,
    JointRandomCrop,
    JointRandomFlip,
    Normalize,
    Pad,
    DefaultFormatBundle,
    Collect,
    Compose,
)

__all__ = [
    "DetDataset",
    "collate_fn",
    "LoadAnnotations",
    "JointPad",
    "JointResize",
    "JointRandomCrop",
    "JointRandomFlip",
    "Normalize",
    "Pad",
    "DefaultFormatBundle",
    "Collect",
    "Compose",
]

"""Detection dataset (pure PyTorch)."""
import os
import csv
import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset


def _read_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    if "ok" not in names:
        names = names + ["ok"]
    cat2idx = {n: i for i, n in enumerate(names)}
    return names, cat2idx, len(names)


class DetDataset(Dataset):
    """Detection dataset.

    Args:
        data_path: root path containing Annotations/ and images/
        info_file: e.g. "Annotations/train.csv" (relative to data_path)
        pipeline: list of transform dicts or a Compose
        class_names_file: e.g. "Annotations/class_names_list.txt"
        delete_ok: if True, drop samples labeled "ok" (no defect)
        filter_empty_gt: if True, drop samples with no gt bbox
    """

    def __init__(self, data_path, info_file, pipeline, class_names_file="Annotations/class_names_list.txt",
                 delete_ok=True, filter_empty_gt=False):
        super().__init__()
        self.data_path = data_path
        self.pipeline = pipeline
        self.delete_ok = delete_ok
        self.filter_empty_gt = filter_empty_gt

        cn_path = os.path.join(data_path, class_names_file)
        self.class_names, self.cat2idx, self.num_classes = _read_class_names(cn_path)

        self.data_infos = self._load_annotations(os.path.join(data_path, info_file))

    def _load_annotations(self, info_path):
        infos = []
        with open(info_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                next(reader)  # skip header
            except StopIteration:
                return infos
            for row in reader:
                if len(row) == 0:
                    continue
                file_name = row[0].strip().replace("\\", "/")
                if not os.path.isabs(file_name):
                    file_name = os.path.join(self.data_path, file_name)
                anno_label = row[1].strip() if len(row) > 1 else "ok"
                if anno_label == "ok":
                    anno_file = ""
                    if self.delete_ok:
                        continue
                else:
                    anno_file = row[2].strip().replace("\\", "/") if len(row) > 2 else ""
                    if anno_file and not os.path.isabs(anno_file):
                        anno_file = os.path.join(self.data_path, anno_file)
                infos.append({"file_name": file_name, "anno_file": anno_file, "label": anno_label})
        return infos

    def __len__(self):
        return len(self.data_infos)

    def _get_ann_info(self, idx):
        info = self.data_infos[idx]
        if info["anno_file"] == "" or not os.path.exists(info["anno_file"]):
            bboxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            return dict(bboxes=bboxes, labels=labels)
        with open(info["anno_file"], "r", encoding="utf-8") as f:
            jf = json.load(f)
        img_w = jf.get("imageWidth", 1024)
        img_h = jf.get("imageHeight", 1024)
        labels = []
        bboxes = []
        for shape in jf.get("shapes", []):
            if not shape.get("enable", True):
                continue
            label = shape["label"]
            if label not in self.cat2idx:
                continue
            if "bbox" in shape:
                x1, y1, x2, y2 = shape["bbox"]
            else:
                pts = np.array(shape["points"], dtype=np.float32)
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
            x1 = max(int(x1), 0)
            y1 = max(int(y1), 0)
            x2 = min(math.ceil(x2), img_w - 1)
            y2 = min(math.ceil(y2), img_h - 1)
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            bboxes.append([x1, y1, x2, y2])
            labels.append(self.cat2idx[label])
        return dict(
            bboxes=np.array(bboxes, dtype=np.float32).reshape(-1, 4),
            labels=np.array(labels, dtype=np.int64),
        )

    def __getitem__(self, idx):
        info = self.data_infos[idx]
        import cv2
        img = cv2.imread(info["file_name"])
        if img is None:
            raise FileNotFoundError(info["file_name"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ann = self._get_ann_info(idx)
        results = dict(
            img=img,
            img_id=idx,
            ori_shape=img.shape,
            gt_bboxes=ann["bboxes"],
            gt_labels=ann["labels"],
            bbox_fields=["gt_bboxes"],
            img_fields=["img"],
        )
        if self.filter_empty_gt and len(ann["bboxes"]) == 0:
            return None
        return self.pipeline(results)


def collate_fn(batch):
    """Collate function: filter None, stack img, keep gt lists."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    imgs = torch.stack([b["img"] for b in batch], dim=0)
    data_metas = []
    gt_bboxes = []
    gt_labels = []
    for b in batch:
        meta = dict(
            img_shape=tuple(b["img"].shape[-2:]),
            ori_shape=b.get("ori_shape", b["img"].shape[-2:]),
            scale_factor=b.get("scale_factor", (1.0, 1.0, 1.0, 1.0)),
            pad_param=b.get("pad_param", None),
        )
        data_metas.append(meta)
        gt_bboxes.append(b["gt_bboxes"])
        gt_labels.append(b["gt_labels"])
    return dict(img=imgs, data_metas=data_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels)

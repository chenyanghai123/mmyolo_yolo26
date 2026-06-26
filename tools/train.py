"""Train YOLO26 detection models (pure PyTorch, no mmcv/mmyolo).

Usage:
    python tools/train.py --config configs/det_tiny.yaml \
        --data-path /path/to/dataset --work-dir /path/to/work \
        --epochs 100 --batch-size 8 --img-size 1024 1024 --lr 0.01

Config file format: YAML with sections Model, Data, Scheduler, Runtime.
Placeholders {num-class}, {epoch}, {sw}, {sh} are auto-replaced.
"""
import os
import argparse
import time
import math
import copy
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolo26.models import (
    YOLO26,
    YOLO26NMSFree,
    YOLO26Head,
    YOLO26NMSFreeHead,
    BatchTaskAlignedAssigner,
)
from yolo26.data import (
    DetDataset,
    collate_fn,
    Compose,
    LoadAnnotations,
    JointPad,
    JointResize,
    JointRandomCrop,
    JointRandomFlip,
    Normalize,
    Pad,
    DefaultFormatBundle,
    Collect,
)
from yolo26.utils import compute_map


# -------- EMA helper --------
class ExpMomentumEMA:
    def __init__(self, model, momentum=0.0001):
        self.momentum = momentum
        self.model = model
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ep, p in zip(self.ema.state_dict().values(), model.state_dict().values()):
                ep.copy_(ep * (1 - self.momentum) + p * self.momentum)


# -------- Scheduler --------
def build_scheduler(optimizer, warmup_iters, total_iters, lr0, lrf=0.01):
    """Linear warmup + cosine annealing (YOLO-style)."""
    def lr_lambda(it):
        if it < warmup_iters:
            return 0.01 + 0.99 * it / max(warmup_iters, 1)  # warmup from 0.01*lr0 to lr0
        progress = (it - warmup_iters) / max(total_iters - warmup_iters, 1)
        return lrf + (1 - lrf) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_optimizer(model, lr=0.01, momentum=0.937, weight_decay=5e-4):
    """SGD with bias/BN no-decay (YOLO-style paramwise)."""
    g0, g1, g2 = [], [], []  # weights, weights+bn, biases
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            for p in m.parameters():
                g2.append(p)
            continue
        for k, p in m.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            if k.endswith("bias"):
                g2.append(p)
            elif k.endswith("weight"):
                g1.append(p)
            else:
                g0.append(p)
    return torch.optim.SGD(
        [{"params": g0, "lr": lr, "weight_decay": 0.0},
         {"params": g1, "lr": lr, "weight_decay": weight_decay},
         {"params": g2, "lr": lr, "weight_decay": 0.0}],
        lr=lr, momentum=momentum,
    )


def build_pipeline(cfg_pipeline, img_size):
    """Convert a pipeline config list to a Compose, substituting size placeholders."""
    pipeline = []
    for t in cfg_pipeline:
        t = copy.deepcopy(t)
        for k, v in list(t.items()):
            if isinstance(v, str):
                v = v.replace("{sw}", str(img_size[0])).replace("{sh}", str(img_size[1]))
                t[k] = v
            elif isinstance(v, list):
                t[k] = [
                    (s.replace("{sw}", str(img_size[0])).replace("{sh}", str(img_size[1]))
                     if isinstance(s, str) else s)
                    for s in v
                ]
        # JointPad/JointResize/JointRandomCrop use 'size' / 'img_scale' / 'crop_size'
        for key in ("size", "crop_size"):
            if key in t and isinstance(t[key], list) and any(isinstance(s, str) for s in t[key]):
                t[key] = [int(s) if isinstance(s, str) else s for s in t[key]]
        if "img_scale" in t:
            new = []
            for s in t["img_scale"]:
                if isinstance(s, list):
                    new.append([int(x) if isinstance(x, str) else x for x in s])
                else:
                    new.append(int(s) if isinstance(s, str) else s)
            t["img_scale"] = new
        pipeline.append(t)
    return Compose(pipeline)


def parse_args():
    p = argparse.ArgumentParser("YOLO26 train (pure PyTorch)")
    p.add_argument("--config", required=True, help="Path to yaml config (e.g. configs/det_tiny.yaml)")
    p.add_argument("--data-path", required=True, help="Dataset root path")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--img-size", type=int, nargs=2, default=[1024, 1024], metavar=("W", "H"))
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-interval", type=int, default=5)
    p.add_argument("--log-interval", type=int, default=50)
    return p.parse_args()


def prepare_config(config_path, args, num_classes):
    """Load yaml, replace placeholders, return cfg dict."""
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    sw, sh = args.img_size[0], args.img_size[1]
    content = (
        content.replace("{sw}", str(sw))
        .replace("{sh}", str(sh))
        .replace("{epoch}", str(args.epochs))
        .replace("{epoch-10}", str(max(args.epochs - 10, 1)))
        .replace("{num-class}", str(num_classes))
    )
    cfg = yaml.safe_load(content)
    # Merge BASE references if present
    cfg = merge_base(cfg, os.path.dirname(os.path.abspath(config_path)))
    return cfg


def merge_base(cfg, config_dir):
    """Recursively merge 'BASE' references (relative to base_cfg dir)."""
    def _resolve(section, base_key="BASE"):
        if not isinstance(section, dict):
            return section
        if base_key in section:
            base_path = section.pop(base_key)
            # Resolve path: try config_dir/base_cfg/<base_path> first, then config_dir/<base_path>
            candidates = [
                os.path.join(config_dir, "base_cfg", base_path),
                os.path.join(config_dir, base_path),
            ]
            found = None
            for c in candidates:
                if os.path.exists(c):
                    found = c
                    break
            if found is None:
                print(f"[warn] BASE not found: {base_path}")
                return section
            with open(found, "r", encoding="utf-8") as f:
                base_section = yaml.safe_load(f)
            # base_section is {SectionName: dict}; merge into the same top-level section
            for k, v in base_section.items():
                if k not in cfg:
                    cfg[k] = v
                elif isinstance(cfg[k], dict) and isinstance(v, dict):
                    # Deep-merge: base first, then override with cfg[k]
                    merged = _deep_merge(copy.deepcopy(v), cfg[k])
                    cfg[k] = merged
            return section
        return section

    for top_key in list(cfg.keys()):
        _resolve(cfg[top_key])
    return cfg


def _deep_merge(base, override):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def get_num_classes(data_path):
    cn_path = os.path.join(data_path, "Annotations/class_names_list.txt")
    if not os.path.exists(cn_path):
        return 1
    with open(cn_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    if "ok" not in names:
        names = names + ["ok"]
    return len(names)


@torch.no_grad()
def validate(model, loader, device, num_classes, iou_thr=0.5):
    model.eval()
    all_dets = []
    all_gts = []
    for batch in loader:
        if batch is None:
            continue
        imgs = batch["img"].to(device)
        data_metas = batch["data_metas"]
        gt_bboxes = batch["gt_bboxes"]
        gt_labels = batch["gt_labels"]
        results = model.simple_test(imgs, data_metas, rescale=True)
        for i, (det_bboxes, det_labels) in enumerate(results):
            if det_bboxes.numel() > 0:
                d = torch.cat([det_bboxes, det_labels.unsqueeze(-1).float()], dim=-1).cpu().numpy()
            else:
                d = np.zeros((0, 6), dtype=np.float32)
            all_dets.append(d)
            g = torch.cat([gt_bboxes[i], gt_labels[i].unsqueeze(-1).float()], dim=-1).cpu().numpy() if len(gt_bboxes[i]) else np.zeros((0, 5), dtype=np.float32)
            all_gts.append(g)
    return compute_map(all_dets, all_gts, num_classes, iou_thr=iou_thr)


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_classes = get_num_classes(args.data_path)
    print(f"[setup] num_classes = {num_classes}, device = {device}")

    cfg = prepare_config(args.config, args, num_classes)
    # Force num_classes consistency in model cfg
    cfg["Model"]["bbox_head"]["head_module"]["num_classes"] = num_classes

    # Build model
    model_type = cfg["Model"].pop("type")
    if model_type == "YOLO26":
        model = YOLO26(cfg["Model"])
    elif model_type == "YOLO26NMSFree":
        model = YOLO26NMSFree(cfg["Model"])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    model = model.to(device)

    # Build datasets
    train_pipeline = build_pipeline(cfg["Data"]["processors"]["train_pipeline"], args.img_size)
    val_pipeline = build_pipeline(cfg["Data"]["processors"].get("val_pipeline", cfg["Data"]["processors"]["train_pipeline"]), args.img_size)
    train_ds = DetDataset(
        data_path=args.data_path,
        info_file=cfg["Data"]["train_dataloader"]["dataset"].get("data_info_files", ["Annotations/train.csv"])[0],
        pipeline=train_pipeline,
        class_names_file=cfg["Data"]["train_dataloader"]["dataset"].get("class_names", "Annotations/class_names_list.txt"),
        delete_ok=cfg["Data"]["train_dataloader"]["dataset"].get("delete_ok", True),
    )
    val_ds = DetDataset(
        data_path=args.data_path,
        info_file=cfg["Data"]["val_dataloader"]["dataset"].get("data_info_files", ["Annotations/val.csv"])[0],
        pipeline=val_pipeline,
        class_names_file=cfg["Data"]["val_dataloader"]["dataset"].get("class_names", "Annotations/class_names_list.txt"),
        delete_ok=False,
        filter_empty_gt=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"[setup] train={len(train_ds)} val={len(val_ds)}")

    # Optimizer + scheduler
    optimizer = build_optimizer(model, lr=args.lr)
    iters_per_epoch = max(len(train_loader), 1)
    total_iters = args.epochs * iters_per_epoch
    warmup_iters = min(1000, max(3 * iters_per_epoch, 100))
    scheduler = build_scheduler(optimizer, warmup_iters, total_iters, args.lr)

    # EMA
    ema = None if args.no_ema else ExpMomentumEMA(model)

    best_map = 0.0
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "ema" in ckpt and ema is not None:
            ema.ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt.get("epoch", 0)
        best_map = ckpt.get("best_map", 0.0)
        print(f"[resume] from {args.resume}, epoch={start_epoch}, best_map={best_map:.4f}")

    is_nmsfree = isinstance(model, YOLO26NMSFree)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for it, batch in enumerate(train_loader):
            if batch is None:
                continue
            imgs = batch["img"].to(device, non_blocking=True)
            gt_bboxes = [b.to(device) for b in batch["gt_bboxes"]]
            gt_labels = [l.to(device) for l in batch["gt_labels"]]
            data_metas = batch["data_metas"]

            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    losses = model.forward_train(imgs, data_metas, gt_bboxes, gt_labels)
                loss = sum(losses.values())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses = model.forward_train(imgs, data_metas, gt_bboxes, gt_labels)
                loss = sum(losses.values())
                loss.backward()
                optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            running_loss += loss.item()
            if (it + 1) % args.log_interval == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  epoch {epoch+1}/{args.epochs} iter {it+1}/{iters_per_epoch} "
                      f"loss={loss.item():.4f} lr={lr_now:.5f} loss_items=" +
                      " ".join(f"{k}={v.item():.3f}" for k, v in losses.items()))
        # Update ProgLoss at the end of epoch (NMS-Free only)
        if is_nmsfree:
            model.bbox_head.update_prog_loss()
            print(f"  [progloss] epoch {epoch+1}: o2m={model.bbox_head._prog_o2m:.4f} o2o={model.bbox_head._prog_o2o:.4f}")

        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"[epoch {epoch+1}] avg_loss={avg_loss:.4f} time={time.time()-t0:.1f}s")

        # Validation
        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            eval_model = ema.ema if ema is not None else model
            metrics = validate(eval_model, val_loader, device, num_classes)
            print(f"[val epoch {epoch+1}] mAP@0.5 = {metrics['map']:.4f}")
            if metrics["map"] > best_map:
                best_map = metrics["map"]
                ckpt = dict(
                    model=eval_model.state_dict(),
                    epoch=epoch + 1,
                    best_map=best_map,
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                )
                if ema is not None:
                    ckpt["ema"] = ema.ema.state_dict()
                torch.save(ckpt, os.path.join(args.work_dir, "best.pth"))
                print(f"  [ckpt] saved best.pth (mAP={best_map:.4f})")

    # Save final
    final_ckpt = dict(
        model=(ema.ema if ema is not None else model).state_dict(),
        epoch=args.epochs,
        best_map=best_map,
    )
    torch.save(final_ckpt, os.path.join(args.work_dir, "last.pth"))
    print(f"[done] best_map={best_map:.4f}, saved to {args.work_dir}")


if __name__ == "__main__":
    main()

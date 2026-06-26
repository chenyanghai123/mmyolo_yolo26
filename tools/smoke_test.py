# Smoke test: verify YOLO26 (NMS + NMS-Free) can forward + backward on dummy data.

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from yolo26.models import YOLO26, YOLO26NMSFree


def build_cfg(model_type, num_classes=3):
    return {
        "type": model_type,
        "pretrained_model_path": "",
        "pretrained_neck_path": "",
        "pretrained_head_path": "",
        "backbone": dict(
            arch="P5",
            last_stage_out_channels=1024,
            deepen_factor=0.5,
            widen_factor=0.25,
            in_channels=3,
            out_indices=(2, 3, 4),
        ),
        "neck": dict(
            deepen_factor=0.5,
            widen_factor=0.25,
            in_channels=[512, 512, 1024],
            out_channels=[256, 512, 1024],
            num_csp_blocks=2,
        ),
        "bbox_head": dict(
            head_module=dict(
                num_classes=num_classes,
                in_channels=[256, 512, 1024],
                widen_factor=0.25,
                featmap_strides=[8, 16, 32],
                reg_max=16,
            ),
        ),
        "train_cfg": dict(
            assigner=dict(topk=10, alpha=0.5, beta=6.0, eps=1e-9, use_ciou=True, stride=[8, 16, 32]),
            prog_loss=dict(init_o2m=0.8, final_o2m=0.1, max_epochs=10),
        ),
        "test_cfg": dict(score_thr=0.05, nms={"iou_threshold": 0.7}, max_per_img=300),
    }


def run_one(model_type):
    print(f"\n=== {model_type} ===")
    cfg = build_cfg(model_type)
    if model_type == "YOLO26":
        model = YOLO26(cfg)
    else:
        model = YOLO26NMSFree(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    B, C, H, W = 2, 3, 640, 640
    img = torch.randn(B, C, H, W, device=device)
    # Build 2 random GT per image
    gt_bboxes = []
    gt_labels = []
    for _ in range(B):
        cx = torch.rand(2, device=device) * W
        cy = torch.rand(2, device=device) * H
        w = torch.rand(2, device=device) * 100 + 20
        h = torch.rand(2, device=device) * 100 + 20
        b = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
        gt_bboxes.append(b)
        gt_labels.append(torch.randint(0, 3, (2,), device=device))
    data_metas = [dict(scale_factor=(1.0, 1.0, 1.0, 1.0), pad_param=None) for _ in range(B)]

    losses = model.forward_train(img, data_metas, gt_bboxes, gt_labels)
    total = sum(losses.values())
    print("losses:", {k: float(v.item()) for k, v in losses.items()})
    print("total:", float(total.item()))
    total.backward()
    print("backward OK")

    # Inference
    model.eval()
    with torch.no_grad():
        if model_type == "YOLO26":
            results = model.simple_test(img, data_metas, rescale=False)
        else:
            results = model.simple_test(img, data_metas, rescale=False)
    print(f"eval OK, num images: {len(results)}, first det shape: {results[0][0].shape}")

    # Param count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")


if __name__ == "__main__":
    run_one("YOLO26")
    run_one("YOLO26NMSFree")
    print("\nAll smoke tests passed.")

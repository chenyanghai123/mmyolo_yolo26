"""NMS / post-processing for YOLO26 (pure PyTorch)."""
import torch
import torchvision


def multiclass_nms(
    bboxes,
    scores,
    score_thr,
    nms_cfg,
    max_num=-1,
    score_factors=None,
    has_background=False,
):
    """Multi-class NMS for a single image.

    Args:
        bboxes: (N, 4) xyxy
        scores: (N, C) per-class scores
        score_thr: float, score threshold
        nms_cfg: dict, e.g. {"type": "common_nms", "iou_threshold": 0.7}
        max_num: max detections per image
        score_factors: optional multiplicative factor for scores
    Returns:
        det_bboxes: (M, 5) x1,y1,x2,y2,score
        det_labels: (M,)
    """
    iou_thr = nms_cfg.get("iou_threshold", 0.45) if nms_cfg else 0.45
    if score_factors is not None:
        scores = scores * score_factors

    # Filter by score threshold (max class score per box)
    max_scores, max_idx = scores.max(dim=1)
    keep = max_scores > score_thr
    bboxes = bboxes[keep]
    scores = scores[keep]
    max_scores = max_scores[keep]
    max_idx = max_idx[keep]

    if bboxes.numel() == 0:
        return bboxes.new_zeros((0, 5)), bboxes.new_zeros((0,), dtype=torch.long)

    # Per-class NMS via torchvision
    det_bboxes_list = []
    det_labels_list = []
    num_classes = scores.size(1)
    for cls in range(num_classes):
        cls_scores = scores[:, cls]
        cls_keep = cls_scores > score_thr
        if not cls_keep.any():
            continue
        cls_bboxes = bboxes[cls_keep]
        cls_scores = cls_scores[cls_keep]
        keep_idx = torchvision.ops.nms(cls_bboxes, cls_scores, iou_thr)
        cls_bboxes = cls_bboxes[keep_idx]
        cls_scores = cls_scores[keep_idx]
        det_bboxes_list.append(torch.cat([cls_bboxes, cls_scores.unsqueeze(-1)], dim=-1))
        det_labels_list.append(torch.full((cls_bboxes.size(0),), cls, dtype=torch.long, device=bboxes.device))

    if not det_bboxes_list:
        return bboxes.new_zeros((0, 5)), bboxes.new_zeros((0,), dtype=torch.long)

    det_bboxes = torch.cat(det_bboxes_list, dim=0)
    det_labels = torch.cat(det_labels_list, dim=0)
    # Sort by score desc, keep max_num
    sort_idx = det_bboxes[:, -1].argsort(descending=True)
    if max_num > 0:
        sort_idx = sort_idx[:max_num]
    return det_bboxes[sort_idx], det_labels[sort_idx]

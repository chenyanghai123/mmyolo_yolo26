"""Losses for YOLO26 detection (pure PyTorch)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def varifocal_loss(pred, target, weight=None, alpha=0.75, gamma=2.0, reduction="mean", avg_factor=None):
    """Varifocal loss (used as alternative cls loss target weighting, optional)."""
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    weight = target * (target > 0.0).float() + (1.0 - alpha) * (target <= 0.0).float()
    weight = weight * (target > 0.0).float() * (1.0 - pred_sigmoid).pow(gamma) + (
        target <= 0.0
    ).float() * pred_sigmoid.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * weight
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        if avg_factor is None:
            return loss.mean()
        return loss.sum() / max(avg_factor, 1.0)
    return loss


def bbox_iou(box1, box2, xywh=True, iou_mode="ciou", eps=1e-7):
    """Compute IoU/CIoU/GIoU between box1 (...,4) and box2 (...,4).

    If xywh=True, box is (cx,cy,w,h); else (x1,y1,x2,y2).
    Returns iou tensor of shape (...).
    """
    if xywh:
        b1_x1, b1_y1 = box1[..., 0] - box1[..., 2] / 2, box1[..., 1] - box1[..., 3] / 2
        b1_x2, b1_y2 = box1[..., 0] + box1[..., 2] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_y1 = box2[..., 0] - box2[..., 2] / 2, box2[..., 1] - box2[..., 3] / 2
        b2_x2, b2_y2 = box2[..., 0] + box2[..., 2] / 2, box2[..., 1] + box2[..., 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(min=0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(min=0)
    union = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0) + (
        b2_x2 - b2_x1
    ).clamp(min=0) * (b2_y2 - b2_y1).clamp(min=0) - inter + eps

    iou = inter / union
    if iou_mode in ("ciou", "giou"):
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        c_area = cw * ch + eps
        if iou_mode == "giou":
            return iou - (c_area - union) / c_area
        # CIoU
        cw = cw.clamp(min=eps)
        ch = ch.clamp(min=eps)
        rho2 = (
            (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
        ) / 4
        rho2 = rho2 / (cw ** 2 + ch ** 2 + eps)
        v = (4 / math.pi ** 2) * ((torch.atan2((b2_y2 - b2_y1).clamp(min=eps), (b2_x2 - b2_x1).clamp(min=eps))
                                   - torch.atan2((b1_y2 - b1_y1).clamp(min=eps), (b1_x2 - b1_x1).clamp(min=eps))) ** 2)
        with torch.no_grad():
            alpha_ciou = v / (1 - iou + v + eps)
        return iou - (rho2 + alpha_ciou * v)
    return iou


import math


class IoULoss(nn.Module):
    """IoU/CIoU/GIoU loss in xyxy format."""

    def __init__(self, iou_mode="ciou", bbox_format="xyxy", eps=1e-7,
                 reduction="mean", loss_weight=1.0, return_iou=True):
        super().__init__()
        assert iou_mode in ("ciou", "giou", "iou")
        assert bbox_format in ("xyxy", "xywh")
        self.iou_mode = iou_mode
        self.bbox_format = bbox_format
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.return_iou = return_iou

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        reduction = reduction_override or self.reduction
        if weight is not None and weight.dim() > 1:
            weight = weight.mean(-1)
        xywh = self.bbox_format == "xywh"
        iou = bbox_iou(pred, target, xywh=xywh, iou_mode=self.iou_mode, eps=self.eps)
        loss = 1.0 - iou
        if weight is not None:
            loss = loss * weight
        if reduction == "sum":
            loss = loss.sum()
        elif reduction == "mean":
            if avg_factor is None:
                loss = loss.mean()
            else:
                loss = loss.sum() / max(avg_factor, 1.0)
        # loss_weight applied externally by caller in head
        if self.return_iou:
            return self.loss_weight * loss, iou
        return self.loss_weight * loss


class DistributionFocalLoss(nn.Module):
    """Distribution Focal Loss for DFL (reg_max bins)."""

    def __init__(self, reg_max=16, reduction="mean", loss_weight=1.0):
        super().__init__()
        self.reg_max = reg_max
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        # pred: (N, reg_max), target: (N,)
        reduction = reduction_override or self.reduction
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        dis_left = target.long()
        dis_right = dis_left + 1
        weight_left = dis_right.float() - target
        weight_right = 1 - weight_left
        loss = (
            F.cross_entropy(pred, dis_left, reduction="none") * weight_left
            + F.cross_entropy(pred, dis_right, reduction="none") * weight_right
        )
        if weight is not None:
            loss = loss * weight
        if reduction == "sum":
            loss = loss.sum()
        elif reduction == "mean":
            if avg_factor is None:
                loss = loss.mean()
            else:
                loss = loss.sum() / max(avg_factor, 1.0)
        return self.loss_weight * loss


class BCELoss(nn.Module):
    """Binary cross entropy with logits (sigmoid + BCE)."""

    def __init__(self, reduction="none", loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        reduction = reduction_override or self.reduction
        loss = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
        if weight is not None:
            loss = loss * weight
        if reduction == "sum":
            loss = loss.sum()
        elif reduction == "mean":
            if avg_factor is None:
                loss = loss.mean()
            else:
                loss = loss.sum() / max(avg_factor, 1.0)
        return self.loss_weight * loss

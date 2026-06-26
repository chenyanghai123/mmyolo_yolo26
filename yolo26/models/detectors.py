"""YOLO26 detectors: YOLO26 (NMS) and YOLO26NMSFree (NMS-free).

Both detectors assemble backbone -> neck -> head, and implement:
- forward_train(img, data_metas, gt_bboxes, gt_labels, gt_bboxes_ignore) -> losses dict
- simple_test(img, data_metas, rescale=True) -> list of (det_bboxes, det_labels)
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import YOLO26CSPDarknet
from .neck import YOLO26PAFPN
from .head import YOLOv11HeadModule, YOLO26NMSFreeHeadModule
from .losses import IoULoss, DistributionFocalLoss, BCELoss, bbox_iou
from .assigner import BatchTaskAlignedAssigner
from .coder import MlvlPointGenerator, DistancePointBBoxCoder, IdentityBBoxCoder, distance2bbox, bbox2distance
from .postprocess import multiclass_nms


def get_world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


# ============================================================================
# YOLO26 Head (standard NMS) — DFL + CIoU + BatchTaskAlignedAssigner
# ============================================================================

class YOLO26Head(nn.Module):
    """YOLO26 detection head (standard NMS variant) with DFL, CIoU loss."""

    def __init__(
        self,
        head_module,
        num_classes,
        anchor_generator=None,
        bbox_coder=None,
        loss_cls=None,
        loss_bbox=None,
        loss_dfl=None,
        train_cfg=None,
        test_cfg=None,
    ):
        super().__init__()
        self.head_module = head_module
        self.num_classes = num_classes
        self.featmap_strides = head_module.featmap_strides
        self.reg_max = head_module.reg_max

        self.anchor_generator = anchor_generator or MlvlPointGenerator(
            offset=0.5, strides=self.featmap_strides
        )
        self.bbox_coder = bbox_coder or DistancePointBBoxCoder()
        self.loss_cls = loss_cls or BCELoss(reduction="none", loss_weight=0.5)
        self.loss_bbox = loss_bbox or IoULoss(
            iou_mode="ciou", bbox_format="xyxy", reduction="sum",
            loss_weight=7.5, return_iou=False
        )
        self.loss_dfl = loss_dfl or DistributionFocalLoss(
            reg_max=self.reg_max, reduction="mean", loss_weight=0.375
        )
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or {}

        # Build assigner
        assigner_cfg = self.train_cfg.get("assigner", {})
        assigner_cfg = copy.deepcopy(assigner_cfg)
        assigner_cfg.setdefault("num_classes", num_classes)
        assigner_cfg.setdefault("topk", 10)
        assigner_cfg.setdefault("alpha", 0.5)
        assigner_cfg.setdefault("beta", 6.0)
        assigner_cfg.setdefault("eps", 1e-9)
        assigner_cfg.setdefault("use_ciou", True)
        self.assigner = BatchTaskAlignedAssigner(**assigner_cfg)

        # Cached priors
        self.featmap_sizes_train = None
        self.flatten_priors_train = None
        self.stride_tensor = None
        self.num_level_priors = None

        self.featmap_sizes = None
        self.mlvl_priors = None

    def forward(self, feats):
        return self.head_module(feats)

    def forward_train(self, x, data_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, **kwargs):
        outs = self(x)
        # outs: (cls_scores, bbox_preds, bbox_dist_preds) during training
        losses = self.get_losses(*outs, gt_bboxes, gt_labels, data_metas, gt_bboxes_ignore)
        return losses

    def get_losses(self, cls_scores, bbox_preds, bbox_dist_preds,
                   gt_bboxes, gt_labels, data_metas, gt_bboxes_ignore=None):
        num_imgs = len(gt_labels)

        # Build padded batch GT tensor: (B, max_gt, 5) [label, x1, y1, x2, y2]
        device = cls_scores[0].device
        dtype = cls_scores[0].dtype
        max_gt = max((g.size(0) for g in gt_bboxes), default=0)
        if max_gt == 0:
            loss_zero = cls_scores[0].sum() * 0.0
            return dict(loss_cls=loss_zero, loss_bbox=loss_zero, loss_dfl=loss_zero)

        batch_gt = gt_bboxes[0].new_zeros(num_imgs, max_gt, 5)
        for i in range(num_imgs):
            n = gt_bboxes[i].size(0)
            if n > 0:
                batch_gt[i, :n, 0] = gt_labels[i]
                batch_gt[i, :n, 1:] = gt_bboxes[i]
        gt_labels_b = batch_gt[..., :1]    # (B, max_gt, 1)
        gt_bboxes_b = batch_gt[..., 1:]    # (B, max_gt, 4)
        pad_bbox_flag = (gt_bboxes_b.sum(-1, keepdim=True) > 0).float()

        # Build / cache priors
        current_featmap_sizes = [cs.shape[2:] for cs in cls_scores]
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes
            mlvl_priors = self.anchor_generator.grid_priors(
                self.featmap_sizes_train, dtype=dtype, device=device, with_stride=True
            )
            self.num_level_priors = [len(p) for p in mlvl_priors]
            self.flatten_priors_train = torch.cat(mlvl_priors, dim=0)  # (N, 4): cx, cy, sw, sh
            self.stride_tensor = self.flatten_priors_train[..., [2]]  # (N, 1)

        # Flatten predictions
        flatten_cls = torch.cat(
            [cs.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes) for cs in cls_scores], dim=1
        )
        flatten_bbox = torch.cat(
            [bp.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for bp in bbox_preds], dim=1
        )
        flatten_dist = torch.cat(
            [bd.reshape(num_imgs, -1, self.reg_max * 4) for bd in bbox_dist_preds], dim=1
        )

        # Decode pred bbox: distance (ltrb) * stride -> xyxy
        flatten_decoded = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2],
            flatten_bbox,
            self.stride_tensor[..., 0],
        )

        # Assign
        assigned = self.assigner(
            flatten_decoded.detach(),
            flatten_cls.detach().sigmoid(),
            self.flatten_priors_train,
            gt_labels_b,
            gt_bboxes_b,
            pad_bbox_flag,
        )
        assigned_scores = assigned["assigned_scores"]
        assigned_bboxes = assigned["assigned_bboxes"]
        fg_mask = assigned["fg_mask_pre_prior"]

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        # cls loss (BCE, reduction='none', summed over batch and normalized by assigned_scores_sum)
        loss_cls = self.loss_cls(flatten_cls, assigned_scores).sum() / assigned_scores_sum

        # bbox / dfl loss on positive samples only
        # scale to stride-relative for IoU loss
        assigned_bboxes_s = assigned_bboxes / self.stride_tensor  # (B, N, 4)
        flatten_decoded_s = flatten_decoded / self.stride_tensor

        num_pos = fg_mask.sum()
        if num_pos > 0:
            prior_mask = fg_mask.unsqueeze(-1).expand(-1, -1, 4)
            pred_pos = torch.masked_select(flatten_decoded_s, prior_mask).reshape(-1, 4)
            target_pos = torch.masked_select(assigned_bboxes_s, prior_mask).reshape(-1, 4)
            bbox_weight = torch.masked_select(assigned_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_bbox = self.loss_bbox(
                pred_pos, target_pos, weight=bbox_weight, avg_factor=assigned_scores_sum
            )

            # DFL: encode assigned_bboxes to ltrb (stride-relative, clipped to reg_max-1)
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.reg_max - 1, eps=0.01,
            )
            pred_dist_pos = flatten_dist[fg_mask]  # (n_pos, reg_max*4)
            target_ltrb_pos = torch.masked_select(assigned_ltrb, prior_mask).reshape(-1, 4)
            loss_dfl = self.loss_dfl(
                pred_dist_pos.reshape(-1, self.reg_max),
                target_ltrb_pos.reshape(-1),
                weight=bbox_weight.expand(-1, 4).reshape(-1),
                avg_factor=assigned_scores_sum,
            )
        else:
            loss_bbox = flatten_decoded_s.sum() * 0.0
            loss_dfl = flatten_decoded_s.sum() * 0.0

        scale = num_imgs * get_world_size()
        return dict(
            loss_cls=loss_cls * scale,
            loss_bbox=loss_bbox * scale,
            loss_dfl=loss_dfl * scale,
        )

    def get_bboxes(self, cls_scores, bbox_preds, data_metas=None, rescale=False, with_nms=True, cfg=None):
        cfg = cfg or self.test_cfg
        num_imgs = cls_scores[0].size(0)
        device = cls_scores[0].device
        dtype = cls_scores[0].dtype
        featmap_sizes = [cs.shape[2:] for cs in cls_scores]

        if featmap_sizes != self.featmap_sizes:
            self.featmap_sizes = featmap_sizes
            self.mlvl_priors = self.anchor_generator.grid_priors(
                featmap_sizes, dtype=dtype, device=device, with_stride=False
            )
        flatten_priors = torch.cat(self.mlvl_priors, dim=0)  # (N, 2)

        flatten_cls = []
        flatten_bbox = []
        stride_list = []
        for lvl, (cs, bp) in enumerate(zip(cls_scores, bbox_preds)):
            s = self.featmap_strides[lvl]
            flatten_cls.append(cs.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes).sigmoid())
            flatten_bbox.append(bp.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4))
            stride_list.append(
                flatten_bbox[-1].new_full((flatten_bbox[-1].size(1),), s).unsqueeze(0).expand(num_imgs, -1)
            )
        flatten_cls = torch.cat(flatten_cls, dim=1)  # (B, N, C)
        flatten_bbox = torch.cat(flatten_bbox, dim=1)  # (B, N, 4)
        flatten_stride = torch.cat(stride_list, dim=1)  # (B, N)

        # Decode: dist * stride -> xyxy
        flatten_decoded = distance2bbox(
            flatten_priors.unsqueeze(0).expand(num_imgs, -1, -1),
            flatten_bbox * flatten_stride.unsqueeze(-1),
        )

        if not with_nms:
            objectness = torch.ones(flatten_cls.shape[:2], device=device, dtype=dtype)
            return flatten_decoded, flatten_cls, objectness

        score_thr = cfg.get("score_thr", 0.001)
        nms_cfg = cfg.get("nms", {"iou_threshold": 0.7})
        max_num = cfg.get("max_per_img", 300)

        results = []
        for i in range(num_imgs):
            bboxes = flatten_decoded[i]
            scores = flatten_cls[i]
            if rescale and data_metas is not None:
                meta = data_metas[i]
                scale_factor = meta.get("scale_factor", (1.0, 1.0, 1.0, 1.0))
                if meta.get("pad_param") is not None:
                    pad = meta["pad_param"]
                    bboxes = bboxes - bboxes.new_tensor([pad[2], pad[0], pad[2], pad[0]])
                bboxes = bboxes / bboxes.new_tensor(scale_factor)
            det_bboxes, det_labels = multiclass_nms(
                bboxes, scores, score_thr, nms_cfg, max_num=max_num
            )
            results.append((det_bboxes, det_labels))
        return results


# ============================================================================
# YOLO26NMSFree Head — dual assignment (one2many + one2one), ProgLoss, no NMS
# ============================================================================

class YOLO26NMSFreeHead(nn.Module):
    """YOLO26 NMS-Free head.

    Training: dual-assignment loss (one2many topk=10 + one2one topk=7/1)
              with ProgLoss weighting (o2m 0.8 -> 0.1 linear decay)
    Inference: one2one branch only + top-k selection, NO NMS
    Uses IdentityBBoxCoder (direct ltrb, no exp, no DFL)
    """

    def __init__(
        self,
        head_module,
        num_classes,
        anchor_generator=None,
        bbox_coder=None,
        loss_cls=None,
        loss_bbox=None,
        train_cfg=None,
        test_cfg=None,
    ):
        super().__init__()
        self.head_module = head_module
        self.num_classes = num_classes
        self.featmap_strides = head_module.featmap_strides

        self.anchor_generator = anchor_generator or MlvlPointGenerator(
            offset=0.5, strides=self.featmap_strides
        )
        self.bbox_coder = IdentityBBoxCoder()
        self.loss_cls = loss_cls or BCELoss(reduction="none", loss_weight=0.5)
        self.loss_bbox = loss_bbox or IoULoss(
            iou_mode="ciou", bbox_format="xyxy", reduction="sum",
            loss_weight=7.5, return_iou=False
        )
        self.loss_dfl = None  # NMS-Free uses L1 instead of DFL
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or {}

        # one2many assigner (topk=10)
        a_cfg = copy.deepcopy(self.train_cfg.get("assigner", {}))
        a_cfg.setdefault("num_classes", num_classes)
        a_cfg.setdefault("topk", 10)
        a_cfg.setdefault("alpha", 0.5)
        a_cfg.setdefault("beta", 6.0)
        a_cfg.setdefault("eps", 1e-9)
        a_cfg.setdefault("use_ciou", True)
        self.assigner = BatchTaskAlignedAssigner(**a_cfg)

        # one2one assigner (topk=7, topk2=1) — STAL secondary filtering
        a_one_cfg = copy.deepcopy(a_cfg)
        a_one_cfg["topk"] = 7
        a_one_cfg["topk2"] = 1
        self.one_assigner = BatchTaskAlignedAssigner(**a_one_cfg)

        # ProgLoss weights
        prog = self.train_cfg.get("prog_loss", {})
        self._prog_updates = 0
        self._prog_total = 1.0
        self._prog_o2m_init = prog.get("init_o2m", 0.8)
        self._prog_o2m = self._prog_o2m_init
        self._prog_o2m_final = prog.get("final_o2m", 0.1)
        self._prog_o2o = self._prog_total - self._prog_o2m
        self._prog_max_epochs = prog.get("max_epochs", 300)

        # L1 aux regression loss weight (replaces DFL for reg_max=1)
        self._l1_loss_weight = self.train_cfg.get("l1_loss_weight", 1.5)

        self._cached_featmap_sizes = None
        self._cached_mlvl_priors = None

    def forward(self, feats):
        return self.head_module(feats)

    def update_prog_loss(self):
        """Call once per epoch: linear decay of one2many weight."""
        self._prog_updates += 1
        self._prog_o2m = self._prog_decay(self._prog_updates)
        self._prog_o2o = max(self._prog_total - self._prog_o2m, 0)

    def _prog_decay(self, x):
        return (
            max(1 - x / max(self._prog_max_epochs - 1, 1), 0)
            * (self._prog_o2m_init - self._prog_o2m_final)
            + self._prog_o2m_final
        )

    def forward_train(self, x, data_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, **kwargs):
        if self.training:
            o2m_cls, o2m_reg, o2o_cls, o2o_reg = self.head_module(x)
        else:
            o2o_cls, o2o_reg = self.head_module(x)
            o2m_cls, o2m_reg = o2o_cls, o2o_reg

        # one2many loss
        losses = self.get_losses(o2m_cls, o2m_reg, gt_bboxes, gt_labels, data_metas, assigner=self.assigner)
        # one2one loss
        losses_o2o = self.get_losses(o2o_cls, o2o_reg, gt_bboxes, gt_labels, data_metas, assigner=self.one_assigner)

        # ProgLoss weighting
        for k, v in losses.items():
            losses[k] = v * self._prog_o2m
        for k, v in losses_o2o.items():
            losses[f"{k}_one2one"] = v * self._prog_o2o
        return losses

    def get_losses(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, data_metas, assigner=None):
        device = cls_scores[0].device
        dtype = cls_scores[0].dtype
        assigner = assigner or self.assigner
        num_imgs = cls_scores[0].size(0)

        # 1. priors (cx, cy, stride)
        featmap_sizes = [cs.shape[2:] for cs in cls_scores]
        mlvl_priors = self.anchor_generator.grid_priors(
            featmap_sizes, dtype=dtype, device=device, with_stride=True
        )
        priors = torch.cat(mlvl_priors, dim=0)  # (N, 4)

        # 2. flatten preds
        flatten_cls = torch.cat(
            [cs.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes) for cs in cls_scores], dim=1
        )
        flatten_bbox = torch.cat(
            [bp.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for bp in bbox_preds], dim=1
        )

        # 3. pad GT to batch tensor
        max_gt = max((g.size(0) for g in gt_bboxes), default=0)
        if max_gt == 0:
            z = flatten_cls.sum() * 0.0
            return dict(loss_cls=z, loss_bbox=z, loss_l1=z)

        batch_gt_bboxes = []
        batch_gt_labels = []
        pad_bbox_flag = []
        for b in range(num_imgs):
            n = gt_bboxes[b].size(0)
            batch_gt_bboxes.append(torch.cat([gt_bboxes[b], gt_bboxes[b].new_zeros((max_gt - n, 4))], dim=0))
            batch_gt_labels.append(torch.cat([gt_labels[b].view(-1, 1), gt_labels[b].new_zeros((max_gt - n, 1))], dim=0))
            pad_bbox_flag.append(torch.cat([torch.ones((n, 1), device=device), torch.zeros((max_gt - n, 1), device=device)], dim=0))
        batch_gt_bboxes = torch.stack(batch_gt_bboxes)  # (B, max_gt, 4)
        batch_gt_labels = torch.stack(batch_gt_labels)  # (B, max_gt, 1)
        pad_bbox_flag = torch.stack(pad_bbox_flag)  # (B, max_gt, 1)

        # 4. decode direct ltrb -> xyxy (no exp)
        cx = priors[:, 0].view(1, -1).expand(num_imgs, -1)
        cy = priors[:, 1].view(1, -1).expand(num_imgs, -1)
        stride = priors[:, 2].view(1, -1).expand(num_imgs, -1)

        l = flatten_bbox[..., 0]
        t = flatten_bbox[..., 1]
        r = flatten_bbox[..., 2]
        b_val = flatten_bbox[..., 3]
        x1 = cx - l * stride
        y1 = cy - t * stride
        x2 = cx + r * stride
        y2 = cy + b_val * stride
        decoded_bboxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (B, N, 4)

        # 5. batch assign (TAL)
        assigned = assigner(
            decoded_bboxes.detach(),
            flatten_cls.detach().sigmoid(),
            priors,
            batch_gt_labels,
            batch_gt_bboxes,
            pad_bbox_flag,
        )
        fg_mask = assigned["fg_mask_pre_prior"]
        assigned_scores = assigned["assigned_scores"]
        assigned_bboxes = assigned["assigned_bboxes"]

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        # 6. cls loss
        loss_cls = (
            self.loss_cls(
                flatten_cls.reshape(-1, self.num_classes),
                assigned_scores.reshape(-1, self.num_classes),
            ).sum()
            / assigned_scores_sum
        )

        # 7. bbox IoU loss on positives
        if fg_mask.any():
            bbox_weight = assigned_scores.sum(-1)[fg_mask].unsqueeze(-1)
            loss_bbox = (
                self.loss_bbox(
                    decoded_bboxes[fg_mask],
                    assigned_bboxes[fg_mask],
                    weight=bbox_weight,
                    avg_factor=assigned_scores_sum,
                )
            )

            # 8. L1 aux loss (replaces DFL for reg_max=1)
            target_ltrb = torch.stack(
                [
                    (cx - assigned_bboxes[..., 0]) / stride,
                    (cy - assigned_bboxes[..., 1]) / stride,
                    (assigned_bboxes[..., 2] - cx) / stride,
                    (assigned_bboxes[..., 3] - cy) / stride,
                ],
                dim=-1,
            )  # (B, N, 4)

            h0, w0 = featmap_sizes[0]
            first_stride = priors[0, 2]
            imgsz = torch.tensor([h0 * first_stride, w0 * first_stride], device=device, dtype=dtype)

            pred_ltrb_fg = flatten_bbox[fg_mask] * stride[fg_mask].unsqueeze(-1)
            target_ltrb_fg = target_ltrb[fg_mask] * stride[fg_mask].unsqueeze(-1)
            pred_ltrb_fg[..., 0::2] = pred_ltrb_fg[..., 0::2] / imgsz[1]
            pred_ltrb_fg[..., 1::2] = pred_ltrb_fg[..., 1::2] / imgsz[0]
            target_ltrb_fg[..., 0::2] = target_ltrb_fg[..., 0::2] / imgsz[1]
            target_ltrb_fg[..., 1::2] = target_ltrb_fg[..., 1::2] / imgsz[0]

            loss_l1 = (
                (F.l1_loss(pred_ltrb_fg, target_ltrb_fg, reduction="none").mean(-1, keepdim=True) * bbox_weight).sum()
                / assigned_scores_sum
                * self._l1_loss_weight
            )
        else:
            loss_bbox = flatten_bbox.sum() * 0.0
            loss_l1 = flatten_bbox.sum() * 0.0

        scale = num_imgs * get_world_size()
        return dict(
            loss_cls=loss_cls * scale,
            loss_bbox=loss_bbox * scale,
            loss_l1=loss_l1 * scale,
        )

    def get_bboxes(self, cls_scores, bbox_preds, data_metas=None, rescale=False, with_nms=True, cfg=None):
        """NMS-free post-processing: decode + top-k selection."""
        cfg = cfg or self.test_cfg
        num_imgs = cls_scores[0].size(0)
        device = cls_scores[0].device
        dtype = cls_scores[0].dtype
        featmap_sizes = [cs.shape[2:] for cs in cls_scores]

        if featmap_sizes != self._cached_featmap_sizes:
            self._cached_featmap_sizes = featmap_sizes
            self._cached_mlvl_priors = self.anchor_generator.grid_priors(
                featmap_sizes, dtype=dtype, device=device, with_stride=True
            )

        flatten_scores = []
        flatten_bboxes = []
        for cls_score, bbox_pred, priors in zip(cls_scores, bbox_preds, self._cached_mlvl_priors):
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes).sigmoid()
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            cx = priors[:, 0].view(1, -1).expand(num_imgs, -1)
            cy = priors[:, 1].view(1, -1).expand(num_imgs, -1)
            stride = priors[:, 2].view(1, -1).expand(num_imgs, -1)
            x1 = cx - bbox_pred[..., 0] * stride
            y1 = cy - bbox_pred[..., 1] * stride
            x2 = cx + bbox_pred[..., 2] * stride
            y2 = cy + bbox_pred[..., 3] * stride
            flatten_bboxes.append(torch.stack([x1, y1, x2, y2], dim=-1))
            flatten_scores.append(cls_score)
        flatten_bboxes = torch.cat(flatten_bboxes, dim=1)
        flatten_scores = torch.cat(flatten_scores, dim=1)

        if not with_nms:
            objectness = torch.ones(flatten_scores.shape[:2], device=device, dtype=dtype)
            return flatten_bboxes, flatten_scores, objectness

        max_det = cfg.get("max_per_img", 300)
        score_thr = cfg.get("score_thr", 0.001)

        results = []
        for i in range(num_imgs):
            bboxes = flatten_bboxes[i]
            scores = flatten_scores[i]
            if rescale and data_metas is not None:
                meta = data_metas[i]
                scale_factor = meta.get("scale_factor", (1.0, 1.0, 1.0, 1.0))
                if meta.get("pad_param") is not None:
                    pad = meta["pad_param"]
                    bboxes = bboxes - bboxes.new_tensor([pad[2], pad[0], pad[2], pad[0]])
                bboxes = bboxes / bboxes.new_tensor(scale_factor)
            max_scores, max_labels = scores.max(dim=-1)
            keep = max_scores > score_thr
            max_scores = max_scores[keep]
            max_labels = max_labels[keep]
            bboxes = bboxes[keep]
            k = min(max_det, max_scores.shape[0])
            if k > 0:
                topk_scores, topk_idx = max_scores.topk(k)
                topk_labels = max_labels[topk_idx]
                topk_bboxes = bboxes[topk_idx]
                det_bboxes = torch.cat([topk_bboxes, topk_scores.unsqueeze(-1)], dim=-1)
                det_labels = topk_labels
            else:
                det_bboxes = bboxes.new_zeros((0, 5))
                det_labels = bboxes.new_zeros((0,), dtype=torch.long)
            results.append((det_bboxes, det_labels))
        return results


# ============================================================================
# Detectors: YOLO26 (NMS) and YOLO26NMSFree
# ============================================================================

class _BaseDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.type = None

    @property
    def with_neck(self):
        return hasattr(self, "neck") and self.neck is not None

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def init_weights(self, pretrained_backbone=None, pretrained_neck=None, pretrained_head=None):
        self.backbone.init_weights(pretrained_backbone)
        if self.with_neck:
            self.neck.init_weights(pretrained_neck)
        self.bbox_head.head_module.init_weights(pretrained_head)


class YOLO26(_BaseDetector):
    """YOLO26 detector (standard NMS variant)."""

    def __init__(self, cfg):
        super().__init__()
        self.type = "YOLO26"
        self.backbone = YOLO26CSPDarknet(**cfg["backbone"])
        self.neck = YOLO26PAFPN(**cfg["neck"])

        head_module = YOLOv11HeadModule(
            num_classes=cfg["bbox_head"]["head_module"]["num_classes"],
            in_channels=cfg["bbox_head"]["head_module"]["in_channels"],
            widen_factor=cfg["bbox_head"]["head_module"].get("widen_factor", 1.0),
            featmap_strides=cfg["bbox_head"]["head_module"].get("featmap_strides", [8, 16, 32]),
            reg_max=cfg["bbox_head"]["head_module"].get("reg_max", 16),
        )
        self.bbox_head = YOLO26Head(
            head_module=head_module,
            num_classes=cfg["bbox_head"]["head_module"]["num_classes"],
            train_cfg=cfg.get("train_cfg"),
            test_cfg=cfg.get("test_cfg"),
        )
        self.train_cfg = cfg.get("train_cfg")
        self.test_cfg = cfg.get("test_cfg")

        # Init weights (with optional pretrained paths)
        pb = cfg.get("pretrained_model_path", "")
        pn = cfg.get("pretrained_neck_path", "")
        ph = cfg.get("pretrained_head_path", "")
        self.init_weights(
            pretrained_backbone=pb if pb else None,
            pretrained_neck=pn if pn else None,
            pretrained_head=ph if ph else None,
        )

    def forward_train(self, img, data_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, **kwargs):
        batch_input_shape = tuple(img[0].size()[-2:])
        for m in data_metas:
            m["batch_input_shape"] = batch_input_shape
        x = self.extract_feat(img)
        return self.bbox_head.forward_train(x, data_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)

    def simple_test(self, img, data_metas, rescale=True, **kwargs):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # outs: (cls_scores, bbox_preds) at inference
        return self.bbox_head.get_bboxes(*outs, data_metas=data_metas, rescale=rescale)


class YOLO26NMSFree(_BaseDetector):
    """YOLO26 NMS-Free detector."""

    def __init__(self, cfg):
        super().__init__()
        self.type = "YOLO26NMSFree"
        self.backbone = YOLO26CSPDarknet(**cfg["backbone"])
        self.neck = YOLO26PAFPN(**cfg["neck"])

        head_module = YOLO26NMSFreeHeadModule(
            num_classes=cfg["bbox_head"]["head_module"]["num_classes"],
            in_channels=cfg["bbox_head"]["head_module"]["in_channels"],
            widen_factor=cfg["bbox_head"]["head_module"].get("widen_factor", 1.0),
            featmap_strides=cfg["bbox_head"]["head_module"].get("featmap_strides", [8, 16, 32]),
        )
        self.bbox_head = YOLO26NMSFreeHead(
            head_module=head_module,
            num_classes=cfg["bbox_head"]["head_module"]["num_classes"],
            train_cfg=cfg.get("train_cfg"),
            test_cfg=cfg.get("test_cfg"),
        )
        self.train_cfg = cfg.get("train_cfg")
        self.test_cfg = cfg.get("test_cfg")

        pb = cfg.get("pretrained_model_path", "")
        pn = cfg.get("pretrained_neck_path", "")
        ph = cfg.get("pretrained_head_path", "")
        self.init_weights(
            pretrained_backbone=pb if pb else None,
            pretrained_neck=pn if pn else None,
            pretrained_head=ph if ph else None,
        )

    def forward_train(self, img, data_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, **kwargs):
        batch_input_shape = tuple(img[0].size()[-2:])
        for m in data_metas:
            m["batch_input_shape"] = batch_input_shape
        x = self.extract_feat(img)
        return self.bbox_head.forward_train(x, data_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)

    def simple_test(self, img, data_metas, rescale=True, **kwargs):
        x = self.extract_feat(img)
        # Inference: head_module returns (o2o_cls, o2o_reg)
        o2o_cls, o2o_reg = self.bbox_head(x)
        return self.bbox_head.get_bboxes(o2o_cls, o2o_reg, data_metas=data_metas, rescale=rescale)

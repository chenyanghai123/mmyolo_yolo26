"""BatchTaskAlignedAssigner for YOLO26.

Returns dict with:
    assigned_labels: (B, N)
    assigned_bboxes: (B, N, 4)
    assigned_scores: (B, N, C)  -- already multiplied by norm_align_metric
    fg_mask_pre_prior: (B, N) bool
    assigned_gt_idxs: (B, N)
"""
import torch
import torch.nn as nn


def select_candidates_in_gts(priors, gt_bboxes, eps=1e-9):
    """Select positive priors whose center falls inside any gt box.

    Args:
        priors: (N, 2) or (N, 4) center points
        gt_bboxes: (B, n_gt, 4) xyxy
    Returns:
        is_in_gts: (B, N) bool
    """
    priors = priors[:, :2]
    bs, n_gt, _ = gt_bboxes.shape
    if n_gt == 0:
        return torch.zeros(bs, priors.shape[0], dtype=torch.bool, device=priors.device)
    # (B, n_gt, N)
    x_min = gt_bboxes[..., 0].unsqueeze(-1) <= priors[:, 0].unsqueeze(0).unsqueeze(0)
    x_max = priors[:, 0].unsqueeze(0).unsqueeze(0) <= gt_bboxes[..., 2].unsqueeze(-1)
    y_min = gt_bboxes[..., 1].unsqueeze(-1) <= priors[:, 1].unsqueeze(0).unsqueeze(0)
    y_max = priors[:, 1].unsqueeze(0).unsqueeze(0) <= gt_bboxes[..., 3].unsqueeze(-1)
    return (x_min & x_max & y_min & y_max).any(dim=1)


def select_highest_overlaps(pos_mask, overlaps, n_gt):
    """If a prior is assigned to multiple gts, select the one with highest IoU.

    Args:
        pos_mask: (B, n_gt, N)
        overlaps: (B, n_gt, N)
    Returns:
        assigned_gt_idxs: (B, N)
        fg_mask_pre_prior: (B, N)
        pos_mask: (B, n_gt, N)  -- only the selected gt is True per prior
    """
    # Work in float to support argmax/scatter uniformly on CUDA
    pos_mask_f = pos_mask.float()
    fg_mask_pre_prior = pos_mask_f.sum(dim=1) > 0
    # For priors assigned to multiple gts, pick the highest-overlap gt
    mask_multi_gts = (pos_mask_f.sum(dim=1, keepdim=True) > 1).repeat(1, n_gt, 1)
    if mask_multi_gts.any():
        masked_overlaps = overlaps * pos_mask_f  # (B, n_gt, N)
        max_idx = masked_overlaps.argmax(dim=1)  # (B, N)
        is_max = torch.zeros_like(pos_mask_f)
        is_max.scatter_(1, max_idx.unsqueeze(1), 1.0)
        pos_mask_f = torch.where(mask_multi_gts, is_max, pos_mask_f)
    assigned_gt_idxs = pos_mask_f.argmax(dim=1)  # (B, N)
    return assigned_gt_idxs, fg_mask_pre_prior, pos_mask_f.bool()


def iou_calculator(bboxes1, bboxes2, eps=1e-7):
    """Compute pairwise IoU between (B, N, 4) and (B, n_gt, 4)."""
    b1_x1, b1_y1, b1_x2, b1_y2 = bboxes1[..., 0], bboxes1[..., 1], bboxes1[..., 2], bboxes1[..., 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = bboxes2[..., 0], bboxes2[..., 1], bboxes2[..., 2], bboxes2[..., 3]

    # (B, n_gt, N)
    inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(2))
    inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(2))
    inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(2))
    inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(2))
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = (b1_x2 - b1_x1).clamp(min=0).unsqueeze(1) * (b1_y2 - b1_y1).clamp(min=0).unsqueeze(1) + (
        b2_x2 - b2_x1
    ).clamp(min=0).unsqueeze(2) * (b2_y2 - b2_y1).clamp(min=0).unsqueeze(2) - inter + eps
    return inter / union


class BatchTaskAlignedAssigner(nn.Module):
    """BatchTaskAlignedAssigner with optional STAL (stride-aware expansion + topk2)."""

    def __init__(
        self,
        num_classes,
        topk=13,
        alpha=1.0,
        beta=6.0,
        eps=1e-7,
        use_ciou=False,
        stride=None,
        topk2=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.topk = topk
        self.topk2 = topk2 if topk2 is not None else topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.use_ciou = use_ciou
        self.stride = stride  # list[int] for STAL, None disables

    @torch.no_grad()
    def forward(self, pred_bboxes, pred_scores, priors, gt_labels, gt_bboxes, pad_bbox_flag):
        """Assign targets.

        Args:
            pred_bboxes: (B, N, 4) xyxy, detached
            pred_scores: (B, N, C) sigmoid scores, detached
            priors: (N, 4) or (N, 3) center + stride
            gt_labels: (B, n_gt, 1)
            gt_bboxes: (B, n_gt, 4) xyxy
            pad_bbox_flag: (B, n_gt, 1) 1 for valid gt
        Returns:
            assigned_result dict.
        """
        priors_xy = priors[:, :2]
        bs = pred_scores.size(0)
        n_gt = gt_bboxes.size(1)

        assigned_labels = gt_bboxes.new_full(pred_scores[..., 0].shape, self.num_classes, dtype=torch.long)
        assigned_bboxes = gt_bboxes.new_zeros(pred_bboxes.shape)
        assigned_scores = gt_bboxes.new_zeros(pred_scores.shape)
        fg_mask_pre_prior = gt_bboxes.new_zeros(pred_scores[..., 0].shape, dtype=torch.bool)
        assigned_gt_idxs = gt_bboxes.new_zeros(pred_scores[..., 0].shape, dtype=torch.long)

        if n_gt == 0:
            return dict(
                assigned_labels=assigned_labels,
                assigned_bboxes=assigned_bboxes,
                assigned_scores=assigned_scores,
                fg_mask_pre_prior=fg_mask_pre_prior,
                assigned_gt_idxs=assigned_gt_idxs,
            )

        # 1) is_in_gts mask (optionally STAL-expanded)
        if self.stride is not None:
            is_in_gts = self._select_candidates_in_gts_stal(priors, gt_bboxes, pad_bbox_flag)
        else:
            is_in_gts = select_candidates_in_gts(priors, gt_bboxes)

        # 2) alignment metric = sqrt(cls_score^alpha * iou^beta), restricted to in-gt priors
        overlaps = iou_calculator(pred_bboxes, gt_bboxes)  # (B, n_gt, N)
        pred_scores = pred_scores.permute(0, 2, 1)  # (B, C, N)
        gt_labels_long = gt_labels.long().clamp(min=0, max=self.num_classes - 1).squeeze(-1)  # (B, n_gt)
        # cls score per (B, n_gt, N) by indexing gt label
        bs_idx, gt_idx = torch.meshgrid(
            torch.arange(bs, device=gt_labels.device),
            torch.arange(n_gt, device=gt_labels.device),
            indexing="ij",
        )
        gt_scores = pred_scores[bs_idx, gt_labels_long, :]  # (B, n_gt, N)

        alignment_metric = gt_scores.pow(self.alpha) * overlaps.pow(self.beta)
        # mask out invalid gts and out-of-gt priors
        pad_mask = pad_bbox_flag.squeeze(-1).unsqueeze(-1)  # (B, n_gt, 1)
        pos_mask = is_in_gts.unsqueeze(1) & (pad_mask.expand(-1, -1, is_in_gts.size(1)) > 0)  # (B, n_gt, N)
        alignment_metric = alignment_metric * pos_mask.float()
        overlaps_masked = overlaps * pos_mask.float()

        # 3) select topk per gt
        topk = min(self.topk, alignment_metric.size(-1))
        topk_metrics, topk_idxs = alignment_metric.topk(topk, dim=-1, largest=True)  # (B, n_gt, topk)
        topk_mask = pos_mask.gather(-1, topk_idxs)  # ensure not picking padded slots
        # build pos_mask from topk (as float for scatter compatibility)
        pos_mask_topk = torch.zeros_like(pos_mask, dtype=torch.float32)
        pos_mask_topk.scatter_(-1, topk_idxs, topk_mask.float())
        pos_mask = pos_mask_topk.bool()

        # 4) select highest overlap if multi-assigned
        assigned_gt_idxs, fg_mask_pre_prior, pos_mask = select_highest_overlaps(
            pos_mask, overlaps, n_gt
        )

        # 5) STAL topk2 secondary filtering (only when topk2 != topk)
        if self.topk2 != self.topk:
            pos_mask_f = pos_mask.float()
            alignment_metric_masked = alignment_metric * pos_mask_f
            topk2 = min(self.topk2, alignment_metric_masked.size(-1))
            topk2_idx = alignment_metric_masked.topk(topk2, dim=-1, largest=True).indices  # (B, n_gt, topk2)
            topk2_mask = torch.zeros_like(pos_mask_f)
            topk2_mask.scatter_(-1, topk2_idx, 1.0)
            pos_mask_f = pos_mask_f * topk2_mask
            fg_mask_pre_prior = pos_mask_f.sum(dim=1) > 0
            assigned_gt_idxs = pos_mask_f.argmax(dim=1)
            pos_mask = pos_mask_f.bool()

        # 6) gather assigned targets
        assigned_labels, assigned_bboxes, assigned_scores = self.get_targets(
            gt_labels, gt_bboxes, assigned_gt_idxs, fg_mask_pre_prior
        )

        # 7) normalize assigned_scores by alignment metric
        alignment_metric = alignment_metric * pos_mask.float()
        pos_align_metrics = alignment_metric.max(axis=1, keepdim=True)[0]  # (B, 1, N)
        pos_overlaps = (overlaps * pos_mask.float()).max(axis=1, keepdim=True)[0]  # (B, 1, N)
        norm_align_metric = (
            (alignment_metric * pos_overlaps / (pos_align_metrics + self.eps)).max(axis=1)[0].unsqueeze(-1)
        )  # (B, N, 1)
        assigned_scores = assigned_scores * norm_align_metric

        return dict(
            assigned_labels=assigned_labels,
            assigned_bboxes=assigned_bboxes,
            assigned_scores=assigned_scores,
            fg_mask_pre_prior=fg_mask_pre_prior,
            assigned_gt_idxs=assigned_gt_idxs,
        )

    def get_targets(self, gt_labels, gt_bboxes, assigned_gt_idxs, fg_mask_pre_prior):
        """Compute assigned labels/bboxes/scores per prior.

        Args:
            gt_labels: (B, n_gt, 1)
            gt_bboxes: (B, n_gt, 4)
            assigned_gt_idxs: (B, N)
            fg_mask_pre_prior: (B, N) bool
        Returns:
            assigned_labels: (B, N)
            assigned_bboxes: (B, N, 4)
            assigned_scores: (B, N, C)
        """
        bs, n_gt, _ = gt_bboxes.shape
        N = assigned_gt_idxs.size(1)
        device = gt_bboxes.device
        # assigned_labels
        gt_labels_long = gt_labels.long().squeeze(-1)  # (B, n_gt)
        # index labels by assigned_gt_idxs (clamp to 0 for negative samples, mask later)
        idx = assigned_gt_idxs.clamp(min=0)
        assigned_labels = torch.gather(gt_labels_long, 1, idx)  # (B, N)
        assigned_labels = torch.where(fg_mask_pre_prior, assigned_labels, assigned_labels.new_full(assigned_labels.shape, self.num_classes))
        # assigned_bboxes
        assigned_bboxes = torch.gather(gt_bboxes, 1, idx.unsqueeze(-1).expand(-1, -1, 4))  # (B, N, 4)
        assigned_bboxes = assigned_bboxes * fg_mask_pre_prior.unsqueeze(-1).float()
        # assigned_scores (one-hot)
        assigned_scores = torch.zeros(bs, N, self.num_classes, device=device, dtype=gt_bboxes.dtype)
        valid_labels = assigned_labels.clamp(min=0, max=self.num_classes - 1)
        assigned_scores.scatter_(2, valid_labels.unsqueeze(-1), 1.0)
        assigned_scores = assigned_scores * fg_mask_pre_prior.unsqueeze(-1).float()
        return assigned_labels, assigned_bboxes, assigned_scores

    def _select_candidates_in_gts_stal(self, priors, gt_bboxes, pad_bbox_flag, eps=1e-9):
        """STAL: stride-aware small target expansion before center-in-gt check."""
        # xyxy -> xywh
        gt_xywh = gt_bboxes.clone()
        gt_xywh[..., 0] = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
        gt_xywh[..., 1] = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
        gt_xywh[..., 2] = gt_bboxes[..., 2] - gt_bboxes[..., 0]
        gt_xywh[..., 3] = gt_bboxes[..., 3] - gt_bboxes[..., 1]

        smallest_stride = self.stride[0]
        second_stride = self.stride[1] if len(self.stride) > 1 else smallest_stride * 2
        wh = gt_xywh[..., 2:]
        wh_mask = wh < smallest_stride
        stride_val = torch.tensor(second_stride, dtype=wh.dtype, device=wh.device)
        gt_xywh[..., 2:] = torch.where((wh_mask * pad_bbox_flag).bool(), stride_val, gt_xywh[..., 2:])

        expanded_gt = gt_bboxes.clone()
        expanded_gt[..., 0] = gt_xywh[..., 0] - gt_xywh[..., 2] / 2
        expanded_gt[..., 1] = gt_xywh[..., 1] - gt_xywh[..., 3] / 2
        expanded_gt[..., 2] = gt_xywh[..., 0] + gt_xywh[..., 2] / 2
        expanded_gt[..., 3] = gt_xywh[..., 1] + gt_xywh[..., 3] / 2
        return select_candidates_in_gts(priors, expanded_gt, eps=eps)

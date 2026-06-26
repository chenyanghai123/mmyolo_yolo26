"""BBox coders and prior generators for YOLO26 (pure PyTorch)."""
import torch


def distance2bbox(points, distance, max_shape=None):
    """Decode distance (ltrb) to bbox (xyxy).

    Args:
        points: (..., 2) center points
        distance: (..., 4) l/t/r/b
    Returns:
        bbox: (..., 4) x1,y1,x2,y2
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], dim=-1)


def bbox2distance(points, bbox, max_dis=16.0, eps=0.01):
    """Encode bbox (xyxy) to distance (ltrb) for DFL target."""
    left = points[..., 0] - bbox[..., 0]
    top = points[..., 1] - bbox[..., 1]
    right = bbox[..., 2] - points[..., 0]
    bottom = bbox[..., 3] - points[..., 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], dim=-1)


class MlvlPointGenerator:
    """Multi-level grid point generator."""

    def __init__(self, strides=(8, 16, 32), offset=0.5):
        self.strides = [(s, s) if isinstance(s, int) else s for s in strides]
        self.offset = offset
        self.num_levels = len(self.strides)

    def grid_priors(self, featmap_sizes, dtype=torch.float32, device="cpu", with_stride=False):
        assert len(featmap_sizes) == self.num_levels
        return [
            self.single_level_grid_priors(featmap_sizes[i], i, dtype=dtype, device=device, with_stride=with_stride)
            for i in range(self.num_levels)
        ]

    def single_level_grid_priors(self, featmap_size, level_idx, dtype=torch.float32, device="cpu", with_stride=False):
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (torch.arange(0, feat_w, device=device, dtype=dtype) + self.offset) * stride_w
        shift_y = (torch.arange(0, feat_h, device=device, dtype=dtype) + self.offset) * stride_h
        shift_yy, shift_xx = torch.meshgrid(shift_y, shift_x, indexing="ij")
        shift_xx = shift_xx.reshape(-1)
        shift_yy = shift_yy.reshape(-1)
        if not with_stride:
            return torch.stack([shift_xx, shift_yy], dim=-1)
        sw = shift_xx.new_full((shift_xx.shape[0],), stride_w)
        sh = shift_yy.new_full((shift_yy.shape[0],), stride_h)
        return torch.stack([shift_xx, shift_yy, sw, sh], dim=-1)


class DistancePointBBoxCoder:
    """Decode distance (ltrb*stride) to bbox (xyxy); encode bbox to distance for DFL."""

    def __init__(self, clip_border=True):
        self.clip_border = clip_border

    def decode(self, points, pred_bboxes, stride=None, max_shape=None):
        if self.clip_border is False:
            max_shape = None
        if stride is not None:
            # stride: (N,) or (B, N, 1) -> broadcast
            pred_bboxes = pred_bboxes * stride[None, :, None] if stride.dim() == 1 else pred_bboxes * stride
        return distance2bbox(points, pred_bboxes, max_shape=max_shape)

    def encode(self, points, gt_bboxes, max_dis=16.0, eps=0.01):
        return bbox2distance(points, gt_bboxes, max_dis=max_dis, eps=eps)


class IdentityBBoxCoder:
    """Identity coder: returns pred as-is (used by NMS-Free head)."""

    def decode(self, points, pred_bboxes, stride=None, max_shape=None):
        return pred_bboxes

    def encode(self, points, gt_bboxes, max_dis=None, eps=None):
        return gt_bboxes

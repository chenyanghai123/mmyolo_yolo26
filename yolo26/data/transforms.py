"""Data transforms (pure PyTorch + numpy + cv2)."""
import random
import numpy as np
import torch


class Compose:
    """Compose transforms from a list of (dict|callable)."""

    def __init__(self, transforms):
        self.transforms = []
        for t in transforms:
            if isinstance(t, dict):
                cls = globals()[t["type"]]
                params = {k: v for k, v in t.items() if k != "type"}
                self.transforms.append(cls(**params))
            else:
                self.transforms.append(t)

    def __call__(self, results):
        for t in self.transforms:
            results = t(results)
            if results is None:
                return None
        return results


class LoadAnnotations:
    """No-op placeholder (annotations already loaded by dataset)."""

    def __init__(self, with_bbox=True):
        self.with_bbox = with_bbox

    def __call__(self, results):
        return results


def _impad(img, padding=None, shape=None, pad_val=0):
    if padding is not None:
        return np.pad(img, ((padding[1], padding[3]), (padding[0], padding[2]), (0, 0)),
                       mode="constant", constant_values=pad_val)
    h, w = shape
    padded = np.full((h, w, img.shape[2]), pad_val, dtype=img.dtype)
    padded[: img.shape[0], : img.shape[1]] = img
    return padded


class JointPad:
    """Pad image and shift bboxes (symmetric center-aligned padding)."""

    def __init__(self, size, sym_pad=True, pad_val=0):
        self.size = (size[0], size[1]) if isinstance(size, (list, tuple)) else (size, size)
        self.sym_pad = sym_pad
        self.pad_val = pad_val

    def __call__(self, results):
        img = results["img"]
        h, w = img.shape[:2]
        pad_h = max(self.size[1] - h, 0)
        pad_w = max(self.size[0] - w, 0)
        if pad_h == 0 and pad_w == 0:
            results["pad_param"] = None
            return results
        if self.sym_pad:
            top = int(round(pad_h // 2 - 0.1))
            left = int(round(pad_w // 2 - 0.1))
            bottom = pad_h - top
            right = pad_w - left
            img = _impad(img, padding=(left, top, right, bottom), pad_val=self.pad_val)
            pad_param = np.array([top, bottom, left, right], dtype=np.int32)
            # shift bboxes
            if len(results["gt_bboxes"]):
                results["gt_bboxes"][:, 0::2] += left
                results["gt_bboxes"][:, 1::2] += top
        else:
            img = _impad(img, shape=(h + pad_h, w + pad_w), pad_val=self.pad_val)
            pad_param = np.array([0, pad_h, 0, pad_w], dtype=np.int32)
        results["img"] = img
        results["pad_shape"] = img.shape
        results["pad_param"] = pad_param
        return results


class JointResize:
    """Resize image + bboxes, optionally keep ratio."""

    def __init__(self, img_scale=None, ratio_range=None, keep_ratio=True, bbox_clip_border=True):
        if img_scale is None:
            img_scale = [640, 640]
        if isinstance(img_scale[0], (list, tuple)):
            self.img_scale = [tuple(s) for s in img_scale]
        else:
            self.img_scale = [tuple(img_scale)]
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.bbox_clip_border = bbox_clip_border

    def _sample_scale(self):
        scale = self.img_scale[0]
        if self.ratio_range is not None:
            r = random.uniform(self.ratio_range[0], self.ratio_range[1])
            return (int(scale[0] * r), int(scale[1] * r))
        if len(self.img_scale) > 1:
            return random.choice(self.img_scale)
        return scale

    def __call__(self, results):
        import cv2
        img = results["img"]
        h, w = img.shape[:2]
        target = self._sample_scale()
        if self.keep_ratio:
            # img_scale is [[H, W]]
            th, tw = target
            r = min(th / h, tw / w)
            new_h, new_w = int(round(h * r)), int(round(w * r))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            w_s, h_s = new_w / w, new_h / h
        else:
            th, tw = target
            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
            w_s, h_s = tw / w, th / h
        results["img"] = img
        scale_factor = (w_s, h_s, w_s, h_s)
        if len(results["gt_bboxes"]):
            bboxes = results["gt_bboxes"] * np.array(scale_factor, dtype=np.float32)
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img.shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img.shape[0])
            results["gt_bboxes"] = bboxes
        results["scale_factor"] = scale_factor
        results["img_shape"] = img.shape
        # Re-scale prior pad_param (if any) to match new image size
        if results.get("pad_param") is not None:
            pp = results["pad_param"]
            results["pad_param"] = np.array([
                int(pp[0] * h_s + 0.5), int(pp[1] * h_s + 0.5),
                int(pp[2] * w_s + 0.5), int(pp[3] * w_s + 0.5)
            ], dtype=np.int32)
        return results


class JointRandomCrop:
    """Random crop image + bboxes to crop_size."""

    def __init__(self, crop_size, crop_type="absolute", allow_negative_crop=True, bbox_clip_border=True):
        self.crop_size = (crop_size[0], crop_size[1]) if isinstance(crop_size, (list, tuple)) else (crop_size, crop_size)
        self.crop_type = crop_type
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border

    def __call__(self, results):
        img = results["img"]
        h, w = img.shape[:2]
        ch = min(self.crop_size[0], h)
        cw = min(self.crop_size[1], w)
        margin_h = max(h - ch, 0)
        margin_w = max(w - cw, 0)
        offset_h = random.randint(0, margin_h)
        offset_w = random.randint(0, margin_w)
        img = img[offset_h:offset_h + ch, offset_w:offset_w + cw]
        results["img"] = img
        results["img_shape"] = img.shape
        if len(results["gt_bboxes"]):
            bboxes = results["gt_bboxes"] - np.array([offset_w, offset_h, offset_w, offset_h], dtype=np.float32)
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img.shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img.shape[0])
            valid = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
            if not valid.any() and not self.allow_negative_crop:
                return None
            results["gt_bboxes"] = bboxes[valid]
            results["gt_labels"] = results["gt_labels"][valid]
        return results


class JointRandomFlip:
    """Random horizontal/vertical flip image + bboxes."""

    def __init__(self, prob=0.5, direction="horizontal"):
        self.prob = prob
        self.direction = direction

    def __call__(self, results):
        if random.random() > self.prob:
            return results
        img = results["img"]
        h, w = img.shape[:2]
        if self.direction == "horizontal":
            img = img[:, ::-1]
            if len(results["gt_bboxes"]):
                b = results["gt_bboxes"].copy()
                x1 = w - b[:, 2]
                x2 = w - b[:, 0]
                b[:, 0] = x1
                b[:, 2] = x2
                results["gt_bboxes"] = b
        else:
            img = img[::-1, :]
            if len(results["gt_bboxes"]):
                b = results["gt_bboxes"].copy()
                y1 = h - b[:, 3]
                y2 = h - b[:, 1]
                b[:, 1] = y1
                b[:, 3] = y2
                results["gt_bboxes"] = b
        results["img"] = np.ascontiguousarray(img)
        return results


class Normalize:
    """Normalize image (mean/std), optionally to_rgb."""

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        img = results["img"].astype(np.float32)
        if self.to_rgb:
            img = img[:, :, ::-1]
        img = (img - self.mean) / self.std
        results["img"] = img
        results["img_norm_cfg"] = dict(mean=self.mean.tolist(), std=self.std.tolist(), to_rgb=self.to_rgb)
        return results


class Pad:
    """Pad image to size_divisor at right-bottom."""

    def __init__(self, size_divisor=32, pad_val=0.447):
        self.size_divisor = size_divisor
        self.pad_val = pad_val

    def __call__(self, results):
        img = results["img"]
        h, w = img.shape[:2]
        new_h = int(np.ceil(h / self.size_divisor)) * self.size_divisor
        new_w = int(np.ceil(w / self.size_divisor)) * self.size_divisor
        if new_h == h and new_w == w:
            return results
        padded = np.full((new_h, new_w, img.shape[2]), self.pad_val, dtype=img.dtype)
        padded[:h, :w] = img
        results["img"] = padded
        results["pad_shape"] = padded.shape
        return results


class DefaultFormatBundle:
    """Convert img + bboxes + labels to tensors (CHW float32, etc.)."""

    def __call__(self, results):
        img = results["img"]
        if img.ndim == 3:
            img = img.transpose(2, 0, 1)  # HWC -> CHW
        results["img"] = torch.from_numpy(np.ascontiguousarray(img)).float()
        if "gt_bboxes" in results and not torch.is_tensor(results["gt_bboxes"]):
            results["gt_bboxes"] = torch.from_numpy(np.ascontiguousarray(results["gt_bboxes"])).float()
        if "gt_labels" in results and not torch.is_tensor(results["gt_labels"]):
            results["gt_labels"] = torch.from_numpy(np.ascontiguousarray(results["gt_labels"])).long()
        return results


class Collect:
    """Pick only specified keys (missing keys skipped)."""

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        return {k: results[k] for k in self.keys if k in results}

"""YOLO26 head modules.

- YOLOv11HeadModule: detection head with DFL (reg_max=16), used by the standard NMS detector.
- YOLO26NMSFreeHeadModule: dual one2many/one2one heads, direct ltrb (no DFL), for NMS-free.
"""
import copy
import math
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from .common import Conv, DepthwiseSeparableConv, make_divisible


def multi_apply(fn, *args):
    """Apply fn to multiple zipped args, return list of tuples, then transposed."""
    results = list(zip(*[fn(*a) for a in zip(*args)]))
    return tuple(list(r) for r in results)


class YOLOv11HeadModule(nn.Module):
    """YOLOv11 head module: decoupled cls/reg with DFL (reg_max > 1).

    During training, forward_single returns (cls_logit, bbox_pred, bbox_dist_pred).
    During inference, returns (cls_logit, bbox_pred).
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        widen_factor=1.0,
        num_base_priors=1,
        featmap_strides=(8, 16, 32),
        reg_max=16,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.featmap_strides = list(featmap_strides)
        self.num_levels = len(self.featmap_strides)
        self.num_base_priors = num_base_priors
        self.reg_max = reg_max

        self.in_channels = [make_divisible(c, widen_factor) for c in in_channels]
        self._init_layers()
        self.register_buffer("proj", torch.arange(self.reg_max, dtype=torch.float), persistent=False)

    def _init_layers(self):
        self.reg_preds = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        reg_out_channels = max((self.reg_max, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)
        for i in range(self.num_levels):
            self.reg_preds.append(nn.Sequential(
                Conv(self.in_channels[i], reg_out_channels, k=3, p=1),
                Conv(reg_out_channels, reg_out_channels, k=3, p=1),
                nn.Conv2d(reg_out_channels, 4 * self.reg_max, kernel_size=1),
            ))
            self.cls_preds.append(nn.Sequential(
                DepthwiseSeparableConv(self.in_channels[i], cls_out_channels, k=3, p=1),
                DepthwiseSeparableConv(cls_out_channels, cls_out_channels, k=3, p=1),
                nn.Conv2d(cls_out_channels, self.num_classes, kernel_size=1),
            ))

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            from ..utils.checkpoint import load_checkpoint
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, _BatchNorm):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
            for reg_pred, cls_pred, stride in zip(self.reg_preds, self.cls_preds, self.featmap_strides):
                reg_pred[-1].bias.data[:] = 1.0  # box bias
                cls_pred[-1].bias.data[: self.num_classes] = math.log(
                    5 / self.num_classes / (640 / stride) ** 2
                )

    def forward(self, x):
        if len(x) != self.num_levels:
            raise ValueError("length of x should be equal to num_levels")
        return multi_apply(self.forward_single, x, self.cls_preds, self.reg_preds)

    def forward_single(self, x, cls_pred, reg_pred):
        b, _, h, w = x.shape
        cls_logit = cls_pred(x)
        bbox_dist_preds = reg_pred(x)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape([-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        return cls_logit, bbox_preds


class YOLO26NMSFreeHeadModule(nn.Module):
    """YOLO26 NMS-Free head module with separate one2many/one2one heads.

    - Direct ltrb prediction (no DFL, reg_max=1)
    - one2many branch (cls_preds/reg_preds): topk=10 training loss
    - one2one branch (one2one_cls_preds/one2one_reg_preds): topk=1 training loss + NMS-free inference
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        widen_factor=1.0,
        num_base_priors=1,
        featmap_strides=(8, 16, 32),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.featmap_strides = list(featmap_strides)
        self.num_levels = len(self.featmap_strides)
        self.num_base_priors = num_base_priors
        self.in_channels = [make_divisible(c, widen_factor) for c in in_channels]
        self._init_layers()
        # Independent one2one heads (separate params)
        self.one2one_cls_preds = copy.deepcopy(self.cls_preds)
        self.one2one_reg_preds = copy.deepcopy(self.reg_preds)

    def _init_layers(self):
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        cls_out = max(self.in_channels[0], self.num_classes)
        reg_out = max(self.in_channels[0] // 4, 16)
        for i in range(self.num_levels):
            self.cls_preds.append(nn.Sequential(
                DepthwiseSeparableConv(self.in_channels[i], cls_out, k=3, p=1),
                DepthwiseSeparableConv(cls_out, cls_out, k=3, p=1),
                nn.Conv2d(cls_out, self.num_classes, kernel_size=1),
            ))
            self.reg_preds.append(nn.Sequential(
                Conv(self.in_channels[i], reg_out, k=3, p=1),
                Conv(reg_out, reg_out, k=3, p=1),
                nn.Conv2d(reg_out, 4, kernel_size=1),
            ))

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            from ..utils.checkpoint import load_checkpoint
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, _BatchNorm):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
            # cls bias init (prior = 5/nc/(640/s)^2)
            for cls_pred, stride in zip(self.cls_preds, self.featmap_strides):
                cls_pred[-1].bias.data[: self.num_classes] = math.log(
                    5 / self.num_classes / (640 / stride) ** 2
                )
            for cls_pred, stride in zip(self.one2one_cls_preds, self.featmap_strides):
                cls_pred[-1].bias.data[: self.num_classes] = math.log(
                    5 / self.num_classes / (640 / stride) ** 2
                )
            # reg bias = 2.0 (direct ltrb)
            for reg_pred in self.reg_preds:
                reg_pred[-1].bias.data[:] = 2.0
            for reg_pred in self.one2one_reg_preds:
                reg_pred[-1].bias.data[:] = 2.0

    def forward(self, feats):
        if len(feats) != self.num_levels:
            raise ValueError("length of feats should match num_levels")
        if self.training:
            o2m_cls, o2m_reg = multi_apply(self._forward_single, feats, self.cls_preds, self.reg_preds)
            # Detach features for one2one branch (gradients don't flow back)
            feats_d = [f.detach() for f in feats]
            o2o_cls, o2o_reg = multi_apply(self._forward_single, feats_d, self.one2one_cls_preds, self.one2one_reg_preds)
            return o2m_cls, o2m_reg, o2o_cls, o2o_reg
        else:
            o2o_cls, o2o_reg = multi_apply(self._forward_single, feats, self.one2one_cls_preds, self.one2one_reg_preds)
            return o2o_cls, o2o_reg

    def _forward_single(self, x, cls_pred, reg_pred):
        return cls_pred(x), reg_pred(x)

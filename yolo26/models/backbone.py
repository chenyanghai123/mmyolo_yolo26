"""YOLO26 CSP-Darknet backbone."""
import torch
import torch.nn as nn

from .common import (
    Conv,
    C3K2v2,
    SPPF26,
    C2PSA,
    make_divisible,
    make_round,
)


class YOLO26CSPDarknet(nn.Module):
    """YOLO26 CSP-Darknet backbone.

    Architecture (P5), per stage setting:
        [in_channels, mid_channels, out_channels, num_blocks,
         use_spp, use_c3k, expand_ratio]
    """

    arch_settings = {
        "P5": [
            [64, 128, 256, 2, False, False, 0.25],
            [256, 256, 512, 2, False, False, 0.25],
            [512, 512, 512, 2, False, True, 0.5],
            [512, 1024, 1024, 2, True, True, 0.5],
        ],
    }

    def __init__(
        self,
        arch="P5",
        deepen_factor=1.0,
        widen_factor=1.0,
        in_channels=3,
        out_indices=(2, 3, 4),
        last_stage_out_channels=1024,
        frozen_stages=-1,
        norm_eval=False,
    ):
        super().__init__()
        self.arch_settings[arch][-1][1] = last_stage_out_channels
        self.arch_settings[arch][-1][2] = last_stage_out_channels

        self.arch_setting = self.arch_settings[arch]
        self.stage_num = len(self.arch_setting)
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.widen_factor = widen_factor
        self.deepen_factor = deepen_factor
        self.norm_eval = norm_eval

        self.stem = self._build_stem_layer()
        self.layers = ["stem"]
        for idx, setting in enumerate(self.arch_setting):
            stage = self._build_stage_layer(idx, setting)
            self.add_module(f"stage{idx + 1}", nn.Sequential(*stage))
            self.layers.append(f"stage{idx + 1}")

        self._freeze_stages()

    # ---- builders ----
    def _build_stem_layer(self):
        return Conv(
            self.in_channels,
            make_divisible(self.arch_setting[0][0], self.widen_factor),
            k=3, s=2, p=1,
        )

    def _build_stage_layer(self, stage_idx, setting):
        (in_ch, mid_ch, out_ch, num_blocks, use_spp, use_c3k, expand_ratio) = setting
        if make_divisible(self.arch_setting[-1][2], self.widen_factor) == 512:
            use_c3k = True

        in_ch = make_divisible(in_ch, self.widen_factor)
        mid_ch = make_divisible(mid_ch, self.widen_factor)
        out_ch = make_divisible(out_ch, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)

        stage = []
        # downsample conv (3x3, stride=2)
        stage.append(Conv(in_ch, mid_ch, k=3, s=2, p=1))
        # CSP block
        stage.append(
            C3K2v2(
                mid_ch, out_ch,
                c3k=use_c3k,
                num_blocks=num_blocks,
                expand_ratio=expand_ratio,
                add_identity=True,
            )
        )
        # SPPF
        if use_spp:
            stage.append(SPPF26(out_ch, out_ch, k=5, n=3))
        # C2PSA only at last stage
        if stage_idx == self.stage_num - 1:
            stage.append(
                C2PSA(out_ch, out_ch, n=int(2 * self.deepen_factor), expand_ratio=expand_ratio)
            )
        return stage

    # ---- freeze ----
    def _freeze_stages(self):
        if self.frozen_stages < 0:
            return
        if self.frozen_stages >= 0:
            for p in self.stem.parameters():
                p.requires_grad = False
            self.stem.eval()
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"stage{i}")
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    # ---- forward ----
    def forward(self, x):
        outs = []
        x = self.stem(x)
        outs.append(x)
        for i, name in enumerate(self.layers[1:]):
            stage = getattr(self, name)
            x = stage(x)
            outs.append(x)
        return [outs[i] for i in self.out_indices]

    # ---- init ----
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            from ..utils.checkpoint import load_checkpoint
            load_checkpoint(self, pretrained, strict=False)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.reset_parameters()

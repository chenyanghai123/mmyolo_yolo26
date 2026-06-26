"""YOLO26 PAFPN neck."""
import torch
import torch.nn as nn

from .common import Conv, C3k2For26, make_divisible, make_round


class YOLO26PAFPN(nn.Module):
    """YOLO26 Path Aggregation FPN.

    Structure (P5, 3 levels):
        Top-down:
            P5 -> up -> concat(P4) -> C3k2
            P4 -> up -> concat(P3) -> C3k2
        Bottom-up:
            P3 -> down -> concat(P4_td) -> C3k2
            P4_bu -> down -> concat(P5_td) -> C3k2(attn)
        out_conv per level: Identity
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        deepen_factor=1.0,
        widen_factor=1.0,
        num_csp_blocks=2,
        freeze_all=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        self.num_csp_blocks = num_csp_blocks
        self.freeze_all = freeze_all
        self.num_levels = len(in_channels)

        # reduce layers (YOLO26 uses Identity, no reduce conv)
        self.reduce_layers = nn.ModuleList(self._build_reduce_layer(i) for i in range(self.num_levels))

        # top-down: upsample + concat + C3k2
        self.upsample_layers = nn.ModuleList()
        self.top_down_layers = nn.ModuleList()
        for idx in range(self.num_levels - 1, 0, -1):
            self.upsample_layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.top_down_layers.append(self._build_top_down_layer(idx))

        # bottom-up: downsample + concat + C3k2
        self.downsample_layers = nn.ModuleList()
        self.bottom_up_layers = nn.ModuleList()
        for idx in range(self.num_levels - 1):
            self.downsample_layers.append(self._build_downsample_layer(idx))
            self.bottom_up_layers.append(self._build_bottom_up_layer(idx))

        # out layers: Identity (YOLO26 has no explicit out_conv)
        self.out_layers = nn.ModuleList(nn.Identity() for _ in range(self.num_levels))

    # ---- builders ----
    def _build_reduce_layer(self, idx):
        return nn.Identity()

    def _build_top_down_layer(self, idx):
        in_ch = make_divisible(self.in_channels[idx - 1] + self.in_channels[idx], self.widen_factor)
        out_ch = make_divisible(self.out_channels[idx - 1], self.widen_factor)
        return C3k2For26(
            in_ch, out_ch,
            n=make_round(self.num_csp_blocks, self.deepen_factor),
            c3k=True, attn=False, shortcut=True,
        )

    def _build_downsample_layer(self, idx):
        ch = make_divisible(self.out_channels[idx], self.widen_factor)
        return Conv(ch, ch, k=3, s=2, p=1)

    def _build_bottom_up_layer(self, idx):
        in_ch = make_divisible(self.out_channels[idx] + self.out_channels[idx + 1], self.widen_factor)
        out_ch = make_divisible(self.out_channels[idx + 1], self.widen_factor)
        use_attn = (idx == self.num_levels - 2)  # PSA at the highest (P5) level
        n = make_round(self.num_csp_blocks, self.deepen_factor)
        if use_attn:
            n = 1
        return C3k2For26(
            in_ch, out_ch, n=n,
            c3k=True, attn=use_attn, shortcut=True,
        )

    # ---- forward ----
    def forward(self, inputs):
        assert len(inputs) == self.num_levels
        # reduce (identity)
        reduce_outs = [self.reduce_layers[i](inputs[i]) for i in range(self.num_levels)]

        # top-down
        inner_outs = [reduce_outs[-1]]
        for idx in range(self.num_levels - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[self.num_levels - 1 - idx](feat_high)
            top_down_inputs = torch.cat([upsample_feat, feat_low], dim=1)
            inner_out = self.top_down_layers[self.num_levels - 1 - idx](top_down_inputs)
            inner_outs.insert(0, inner_out)

        # bottom-up
        outs = [inner_outs[0]]
        for idx in range(self.num_levels - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](torch.cat([downsample_feat, feat_high], dim=1))
            outs.append(out)

        # out (identity)
        return tuple(self.out_layers[i](outs[i]) for i in range(self.num_levels))

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            from ..utils.checkpoint import load_checkpoint
            load_checkpoint(self, pretrained, strict=False)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.reset_parameters()

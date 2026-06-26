"""Common building blocks for YOLO26."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(x, widen_factor=1.0, divisor=8):
    """Ensure x*widen_factor is divisible by divisor."""
    return math.ceil(x * widen_factor / divisor) * divisor


def make_round(x, deepen_factor=1.0):
    """Round x*deepen_factor to int >=1."""
    return max(round(x * deepen_factor), 1) if x > 1 else x


class Conv(nn.Module):
    """Conv2d + BN + SiLU."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DepthwiseSeparableConv(nn.Module):
    """DepthwiseSeparableConv used in YOLOv11 head (DW + PW)."""

    def __init__(self, c1, c2, k=3, s=1, p=1, act=True):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, k, s, p, groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn2(self.pw(self.act(self.bn1(self.dw(x))))))


class Bottleneck(nn.Module):
    """Standard residual bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster CSP Bottleneck with 2 convolutions (YOLOv8 C2f)."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        # Bottleneck k=(k1, k2): cv1 uses k1, cv2 uses k2 (both ints)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3 with customizable kernel size."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        # C3k uses (k, k) kernels inside Bottleneck
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2(C2f):
    """YOLOv11 C3k2: C2f where inner blocks can be C3k."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class C3k2For26(C2f):
    """YOLO26 C3k2 variant: supports attn (PSA) and C3k inner block.

    Block choices per layer:
        - attn=True  -> Bottleneck + PSABlockFor26
        - c3k=True   -> C3k
        - otherwise  -> Bottleneck
    """

    def __init__(self, c1, c2, n=1, c3k=False, attn=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c3k = c3k
        self.attn = attn
        self.m = nn.ModuleList(self._build_block(self.c, shortcut, g) for _ in range(n))

    def _build_block(self, c, shortcut, g):
        if self.attn:
            return nn.Sequential(
                Bottleneck(c, c, shortcut=shortcut, g=g),
                PSABlockFor26(c, attn_ratio=0.5, num_heads=max(c // 64, 1)),
            )
        if self.c3k:
            return C3k(c, c, n=2, shortcut=shortcut, g=g)
        return Bottleneck(c, c, shortcut=shortcut, g=g)


class CSPLayerWithTwoConv(nn.Module):
    """CSP layer with 2 convs, used in C3K2v2."""

    def __init__(self, in_channels, out_channels, expand_ratio=0.5, num_blocks=1, add_identity=True):
        super().__init__()
        self.mid_channels = int(out_channels * expand_ratio)
        self.main_conv = Conv(in_channels, 2 * self.mid_channels, 1)
        self.final_conv = Conv((2 + num_blocks) * self.mid_channels, out_channels, 1)
        self.blocks = nn.ModuleList(
            BottleneckV2(self.mid_channels, self.mid_channels, expansion=1, add_identity=add_identity)
            for _ in range(num_blocks)
        )

    def forward(self, x):
        x_main = list(self.main_conv(x).split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(b(x_main[-1]) for b in self.blocks)
        return self.final_conv(torch.cat(x_main, 1))


class BottleneckV2(nn.Module):
    """Bottleneck with 3x3 conv pair."""

    def __init__(self, in_channels, out_channels, expansion=0.5, kernel_size=(1, 3), padding=(0, 1), add_identity=True):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden, kernel_size[0], 1, padding[0])
        self.conv2 = Conv(hidden, out_channels, kernel_size[1], 1, padding[1])
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        identity = x
        out = self.conv2(self.conv1(x))
        return out + identity if self.add_identity else out


class C3K2v2(CSPLayerWithTwoConv):
    """C3K2v2: CSPLayerWithTwoConv where inner blocks are C3k when c3k=True."""

    def __init__(self, in_channels, out_channels, c3k=False, num_blocks=1, expand_ratio=0.5, add_identity=True):
        super().__init__(in_channels, out_channels, expand_ratio, num_blocks, add_identity)
        c_ = self.mid_channels
        self.blocks = nn.ModuleList(
            C3k(c_, c_, n=2, shortcut=add_identity) if c3k
            else BottleneckV2(c_, c_, expansion=1.0, kernel_size=(3, 3), padding=(1, 1), add_identity=add_identity)
            for _ in range(num_blocks)
        )


class Attention(nn.Module):
    """YOLOv11 Attention: conv-qkv + MHSA + pe + proj."""

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=nn.SiLU())
        self.proj = Conv(dim, dim, 1, act=nn.SiLU())
        self.pe = Conv(dim, dim, 3, 1, 1, g=dim, act=nn.SiLU())

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)


class PSABlock(nn.Module):
    """PSABlock: Attention + FFN with residual."""

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, num_heads=num_heads, attn_ratio=attn_ratio)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSABlockFor26(PSABlock):
    """YOLO26 PSABlockFor26: identical to PSABlock (uses Attention)."""
    pass


class C2PSA(nn.Module):
    """C2PSA: CSP with PSA blocks (YOLOv11 C2PSA)."""

    def __init__(self, c1, c2, n=1, expand_ratio=0.5):
        super().__init__()
        assert c1 == c2
        self.mid_channels = int(c1 * expand_ratio)
        self.conv1 = Conv(c1, 2 * self.mid_channels, 1)
        self.conv2 = Conv(2 * self.mid_channels, c1, 1)
        self.blocks = nn.Sequential(
            *(PSABlock(self.mid_channels, attn_ratio=0.5, num_heads=self.mid_channels // 64) for _ in range(n))
        )

    def forward(self, x):
        a, b = self.conv1(x).split((self.mid_channels, self.mid_channels), dim=1)
        b = self.blocks(b)
        return self.conv2(torch.cat((a, b), 1))


class SPPF26(nn.Module):
    """YOLO26 SPPF: sequential max-pool with shortcut."""

    def __init__(self, c1, c2, k=5, n=3, shortcut=True):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.use_shortcut = c1 == c2

    def forward(self, x):
        y = self.cv1(x)
        outs = [y]
        for _ in range(self.n):
            outs.append(self.m(outs[-1]))
        y = self.cv2(torch.cat(outs, 1))
        return y + x if self.use_shortcut else y

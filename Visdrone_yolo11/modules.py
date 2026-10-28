"""Modular building blocks and layers for YOLO architectures."""
from typing import List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k: Union[int, List[int], Tuple[int, ...]], p: Optional[Union[int, List[int], Tuple[int, ...]]] = None, d: int = 1) -> Union[int, List[int]]:
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Attn

class Conv(nn.Module):
    """Standard Convolution block: Conv2d + BatchNorm2d + SiLU."""

    default_act = nn.SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: Optional[int] = None,
        g: int = 1,
        d: int = 1,
        act: Union[bool, nn.Module] = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    """Depth-wise convolution block."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, d: int = 1, act: Union[bool, nn.Module] = True):
        super().__init__(c1, c2, k=k, s=s, g=math.gcd(c1, c2), d=d, act=act)


class Bottleneck(nn.Module):
    """Standard Bottleneck with optional shortcut connection."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: Tuple[Union[int, Tuple[int, int]], Union[int, Tuple[int, int]]] = (3, 3),
        e: float = 0.5,
    ):
        super().__init__()
        c_ = int(c2 * e)
        k1 = k[0] if isinstance(k[0], int) else k[0][0]
        k2 = k[1] if isinstance(k[1], int) else k[1][0]
        self.cv1 = Conv(c1, c_, k=k1, s=1)
        self.cv2 = Conv(c_, c2, k=k2, s=1, g=g)
        self.add = shortcut and (c1 == c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut=shortcut, g=g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        k: int = 3,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut=shortcut, g=g, k=(k, k), e=1.0) for _ in range(n)))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut=shortcut, g=g, k=(3, 3), e=1.0) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2(C2f):
    """C3k2: C2f variant whose inner blocks are Bottleneck, C3k, or PSABlock."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut=shortcut, g=g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, n=2, shortcut=shortcut, g=g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut=shortcut, g=g)
            for _ in range(n)
        )


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF)."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and (c1 == c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        out = self.cv2(torch.cat(y, 1))
        return out + x if self.add else out


class Cross_Attention(nn.Module):
    """Multi-Head Cross-Attention module with positional encoding."""

    def __init__(self, slots_head: int, dim: int, total_dim: int = 4):
        super().__init__()
        if total_dim % dim != 0:
            raise "total dim has to be divisible by dim"
        self.num_heads = total_dim // dim
        self.head_dim = dim
        self.scale = dim ** -0.5
        self.kv = Conv(total_dim, 2 * total_dim, 1, act=False)
        self.to_q= nn.Conv2d(slots_head, self.num_heads, kernel_size= 1)
        self.to_v= nn.Conv2d(slots_head, self.num_heads, kernel_size= 1)
        self.proj = nn.Linear(dim * self.num_heads, dim * slots_head, 1, act=False)

    def forward(self, slots: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        x: B, D, H, W
        slots: B, h, S, dim
        """
        B, D, H, W = x.shape
        _, h, S, dim= slots.shape
        N = H * W
        kv = self.kv(x)
        q, v= self.to_q(slots)
        k = kv.view(B, self.num_heads, 2 * self.head_dim, N).split(
            [self.head_dim, self.head_dim], dim=2
        ) # k, v: [B, h, dim, N]
        dot = torch.einsum('bhsd, bhdn -> bhsn', q, k) * self.scale
        attn = torch.softmax(dot, dim= 1)
        x = torch.einsum('bshn, bdhn -> bshd', attn, v) # [B, S, h, d]
        x = self.proj(x.view(B, S, -1)).view(B, S, h, dim).permute(0, 2, 1, 3)
        slots = slots + x
        return slots


class PSABlock(nn.Module):
    """Position-Sensitive Attention Block (Attention + FFN)."""

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True):
        super().__init__()
        self.attn = Attention(c, num_heads=num_heads, attn_ratio=attn_ratio)
        self.ffn = nn.Sequential(
            Conv(c, c * 2, 1),
            Conv(c * 2, c, 1, act=False),
        )
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class SlotBlock(nn.Module):
    """Position-Sensitive Attention Block (Attention + FFN)."""

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True):
        super().__init__()
        self.attn = Attention(c, num_heads=num_heads, attn_ratio=attn_ratio)
        self.ffn = nn.Sequential(
            Conv(c, c * 2, 1),
            Conv(c * 2, c, 1, act=False),
        )
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class C2PSA(nn.Module):
    """C2PSA module with attention mechanism (PSA block wrapped in CSP style)."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        assert c1 == c2, f"C2PSA expects c1 == c2, got {c1} vs {c2}"
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), dim=1))


class DFL(nn.Module):
    """Distribution Focal Loss (DFL) integral convolution layer."""

    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape  # batch, channels (4*c1), anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, self.d)

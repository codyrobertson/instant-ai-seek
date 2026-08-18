#!/usr/bin/env python3
"""Hybrid multi-view teacher architectures (tournament T1/T2/T3).

Views:
  - full   : whole-image semantic branch (timm backbone)
  - hp     : high-frequency expert (unsharp residual of the image)
  - lp     : low-frequency expert (strongly blurred image)

T1 = full + hp + lp fused; T2 = same views + patch-shuffle augmentation
resistance; T3 = same + auxiliary generator-prototype contrastive head.

All branches share the input-size contract (384) and output P(AI) via a
2-class head on the fused representation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqExpert(nn.Module):
    """Small conv net over a frequency-filtered view (HP or LP)."""

    def __init__(self, in_ch: int = 3, width: int = 64, out_ch: int = 512):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.GELU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2), nn.GELU(),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 4), nn.GELU(),
            nn.Conv2d(width * 4, out_ch, 1, bias=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.body(x)).flatten(1)


def hp_view(x: torch.Tensor) -> torch.Tensor:
    """Unsharp-mask residual: x - blur(x) (normalized to unit scale)."""
    blur = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
    r = x - blur
    return r / (r.abs().amax(dim=(2, 3), keepdim=True) + 1e-6)


def lp_view(x: torch.Tensor) -> torch.Tensor:
    """Strong low-pass: stacked average pools (keeps HxW, texture removed)."""
    return F.avg_pool2d(F.avg_pool2d(x, kernel_size=9, stride=1, padding=4),
                        kernel_size=9, stride=1, padding=4)


class ColorExpert(nn.Module):
    """CoDA-style color-distribution probe: content-robust global color
    statistics (channel moments + quantized channel/RGB-histograms + HSV
    hue-saturation joint histogram) -> small MLP expert.

    Cheap, differentiable, and deliberately blind to scene layout, so it
    cannot learn the semantic shortcuts a CNN branch learns."""

    def __init__(self, in_ch: int = 3, n_bins: int = 8, out_ch: int = 768):
        super().__init__()
        self.n_bins = n_bins
        # features: per-channel mean/std/skew (3*3) + per-channel hist (3*n_bins)
        #           + RGB joint hist (n_bins**3 too big; use 3 pairwise?) -> keep
        #           channel hist + hue/sat joint hist (n_bins*n_bins)
        in_dim = 3 * 3 + 3 * n_bins + n_bins * n_bins
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.Linear(256, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # channel moments
        mu = x.mean(dim=(2, 3))                                   # B,C
        sd = x.std(dim=(2, 3))
        skew = ((x - mu.view(B, -1, 1, 1)) ** 3).mean(dim=(2, 3)) / (sd ** 3 + 1e-6)
        skew = skew.view(B, -1)
        # per-channel histograms (normalized)
        hists = []
        for c in range(x.shape[1]):
            xc = x[:, c].reshape(B, -1)
            xc_n = ((xc - xc.min(dim=1, keepdim=True).values) /
                    (xc.max(dim=1, keepdim=True).values - xc.min(dim=1, keepdim=True).values + 1e-6))
            b = torch.floor(xc_n * self.n_bins).clamp(0, self.n_bins - 1).long()
            onehot = torch.zeros(B, self.n_bins, device=x.device).scatter_add_(1, b, torch.ones_like(b, dtype=torch.float32))
            hists.append(onehot / onehot.sum(dim=1, keepdim=True).clamp(min=1))
        hist = torch.cat(hists, dim=1)                            # B, 3*n_bins
        # HSV hue/saturation joint histogram (via rgb2hsv on the fly)
        r, g, bch = x[:, 0], x[:, 1], x[:, 2]
        mx = torch.max(torch.max(r, g), bch)
        mn = torch.min(torch.min(r, g), bch)
        delta = mx - mn + 1e-9
        hue = torch.where(delta > 1e-4,
                          torch.where(mx == r, ((g - bch) / delta) % 6.0,
                                      torch.where(mx == g, (bch - r) / delta + 2.0, (r - g) / delta + 4.0)) / 6.0,
                          torch.zeros_like(mx))
        sat = torch.where(mx > 1e-6, delta / (mx + 1e-9), torch.zeros_like(mx))
        hq = torch.floor(hue.clamp(0, 1) * (self.n_bins - 1e-4)).long().clamp(0, self.n_bins - 1).reshape(B, -1)
        sq = torch.floor(sat.clamp(0, 1) * (self.n_bins - 1e-4)).long().clamp(0, self.n_bins - 1).reshape(B, -1)
        joint = (hq * self.n_bins + sq)
        jh = torch.zeros(B, self.n_bins * self.n_bins, device=x.device).scatter_add_(
            1, joint, torch.ones_like(joint, dtype=torch.float32))
        jh = jh / jh.sum(dim=1, keepdim=True).clamp(min=1)
        feats = torch.cat([mu, sd, skew, hist, jh], dim=1)
        return self.mlp(feats)


class HybridTeacher(nn.Module):
    """full-image backbone + optional frequency experts, fused 2-class head.

    views: tuple of view names in {"full", "hp", "lp"}.
    """

    def __init__(self, backbone: str = "convnext_tiny.in12k_ft_in1k",
                 views: tuple[str, ...] = ("full", "hp", "lp"),
                 freeze_full: bool = False):
        super().__init__()
        import timm

        self.views = tuple(views)
        self.full = timm.create_model(backbone, pretrained=True, num_classes=0)
        full_ch = self.full.num_features  # convnext_tiny: 768
        if freeze_full:
            for p in self.full.parameters():
                p.requires_grad_(False)

        self.experts = nn.ModuleDict()
        for v in self.views:
            if v == "color":
                self.experts[v] = ColorExpert(out_ch=full_ch)
            elif v != "full":
                self.experts[v] = FreqExpert(out_ch=full_ch)

        # fusion: one pooled feature vector per view
        self.pool = nn.AdaptiveAvgPool2d(1)
        n_views = len(self.views)
        self.head = nn.Sequential(
            nn.LayerNorm(full_ch * n_views),
            nn.Linear(full_ch * n_views, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2),
        )

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats: dict[str, torch.Tensor] = {}
        if "full" in self.views:
            feats["full"] = self.full.forward_features(x)  # (B, C, H, W)
        if "hp" in self.views:
            feats["hp"] = self.experts["hp"](hp_view(x))
        if "lp" in self.views:
            feats["lp"] = self.experts["lp"](lp_view(x))
        if "color" in self.views:
            feats["color"] = self.experts["color"](x)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        vecs = []
        for v in self.views:
            f = feats[v]
            if f.dim() == 4:
                f = self.pool(f).flatten(1)
            vecs.append(f)
        return self.head(torch.cat(vecs, dim=1))

    def fused_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-head representation (for prototype/contrastive head)."""
        feats = self.forward_features(x)
        vecs = []
        for v in self.views:
            f = feats[v]
            if f.dim() == 4:
                f = self.pool(f).flatten(1)
            vecs.append(f)
        return F.normalize(torch.cat(vecs, dim=1), dim=1)

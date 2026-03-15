from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _residual_kernel(device: torch.device) -> torch.Tensor:
    k = torch.tensor([
        [0.0, -1.0, 0.0],
        [-1.0, 4.0, -1.0],
        [0.0, -1.0, 0.0],
    ], device=device)
    return k.view(1, 1, 3, 3)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyBackbone(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.b1 = ConvBlock(32, 64, stride=2)
        self.b2 = ConvBlock(64, 128, stride=2)
        self.b3 = ConvBlock(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        return self.pool(x).flatten(1)


class EfficientBackbone(nn.Module):
    def __init__(self, variant: str = "b0", pretrained: bool = True):
        super().__init__()
        if variant == "b2":
            w = models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
            m = models.efficientnet_b2(weights=w)
            out_dim = 1408
        else:
            w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            m = models.efficientnet_b0(weights=w)
            out_dim = 1280
        self.features = m.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.pool(x).flatten(1)


class AdvancedAIDetector(nn.Module):
    def __init__(self, backbone: str = "tiny", pretrained_backbone: bool = True):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "tiny":
            self.rgb_branch = TinyBackbone(in_ch=3)
            self.fft_branch = TinyBackbone(in_ch=3)
            self.noise_branch = TinyBackbone(in_ch=3)
        elif backbone == "effb0":
            self.rgb_branch = EfficientBackbone("b0", pretrained=pretrained_backbone)
            self.fft_branch = EfficientBackbone("b0", pretrained=pretrained_backbone)
            self.noise_branch = EfficientBackbone("b0", pretrained=pretrained_backbone)
        elif backbone == "effb2":
            self.rgb_branch = EfficientBackbone("b2", pretrained=pretrained_backbone)
            self.fft_branch = EfficientBackbone("b2", pretrained=pretrained_backbone)
            self.noise_branch = EfficientBackbone("b2", pretrained=pretrained_backbone)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        feat = self.rgb_branch.out_dim
        self.fusion = nn.Sequential(
            nn.Linear(feat * 3, 768 if feat >= 1280 else 512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(768 if feat >= 1280 else 512, 192 if feat >= 1280 else 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192 if feat >= 1280 else 128, 1),
        )

    def _fft_features(self, x: torch.Tensor) -> torch.Tensor:
        x_gray = x.mean(dim=1, keepdim=True)
        fft = torch.fft.fft2(x_gray, norm="ortho")
        mag = torch.log1p(torch.abs(torch.fft.fftshift(fft, dim=(-2, -1))))
        mag = mag / (mag.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        return mag.repeat(1, 3, 1, 1)

    def _noise_residual(self, x: torch.Tensor) -> torch.Tensor:
        k = _residual_kernel(x.device)
        chans = []
        for c in range(3):
            rc = F.conv2d(x[:, c : c + 1], k, padding=1)
            chans.append(rc)
        return torch.tanh(torch.cat(chans, dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_rgb = self.rgb_branch(x)
        f_fft = self.fft_branch(self._fft_features(x))
        f_noise = self.noise_branch(self._noise_residual(x))
        return self.fusion(torch.cat([f_rgb, f_fft, f_noise], dim=1)).squeeze(1)


def build_model(backbone: str = "tiny", pretrained_backbone: bool = True) -> AdvancedAIDetector:
    return AdvancedAIDetector(backbone=backbone, pretrained_backbone=pretrained_backbone)

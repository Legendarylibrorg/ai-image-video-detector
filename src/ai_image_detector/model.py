from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .metadata import METADATA_FEATURE_NAMES


RESIDUAL_KERNEL = (
    (0.0, -1.0, 0.0),
    (-1.0, 4.0, -1.0),
    (0.0, -1.0, 0.0),
)


def _residual_kernel(device: torch.device) -> torch.Tensor:
    k = torch.tensor(RESIDUAL_KERNEL, device=device)
    return k.view(1, 1, 3, 3)


def model_runtime_spec(
    *,
    backbone: str,
    img_size: int,
    metadata_feature_dim: int = 0,
) -> dict[str, object]:
    if backbone == "tiny":
        feature_dim = 256
    elif backbone == "effb2":
        feature_dim = 1408
    elif backbone in {"convnext_tiny", "convnext_small"}:
        feature_dim = 768
    else:
        feature_dim = 1280
    metadata_hidden = 64 if metadata_feature_dim > 0 else 0
    fusion_hidden_1 = 768 if feature_dim >= 1280 else 512
    fusion_hidden_2 = 192 if feature_dim >= 1280 else 128
    return {
        "schema": "ai-image-detector-runtime-v1",
        "architecture": {
            "name": "AdvancedAIDetector",
            "backbone": backbone,
            "img_size": int(img_size),
            "branches": [
                {"name": "rgb_branch", "input_channels": 3, "preprocess": "rgb_identity"},
                {"name": "fft_branch", "input_channels": 3, "preprocess": "grayscale_fft_magnitude"},
                {"name": "noise_branch", "input_channels": 3, "preprocess": "laplacian_residual"},
            ],
            "metadata_feature_dim": int(metadata_feature_dim),
            "auxiliary_features": {
                "enabled": bool(metadata_feature_dim > 0),
                "branch_name": "metadata_branch",
                "sources": ["metadata", "provenance", "text_signals"] if metadata_feature_dim > 0 else [],
                "feature_names": list(METADATA_FEATURE_NAMES[:metadata_feature_dim]) if metadata_feature_dim > 0 else [],
            },
            "metadata_hidden_dim": int(metadata_hidden),
            "fusion": {
                "type": "mlp",
                "input_dim": int((feature_dim * 3) + metadata_hidden),
                "hidden_dims": [int(fusion_hidden_1), int(fusion_hidden_2)],
                "output_dim": 1,
            },
        },
        "preprocessing": {
            "resize": {"mode": "bilinear", "size": [int(img_size), int(img_size)]},
            "tensor_conversion": "torchvision.transforms.ToTensor",
            "normalization": None,
            "rgb_branch": {
                "source": "rgb",
                "channel_order": "RGB",
            },
            "fft_branch": {
                "source": "rgb_mean_grayscale",
                "fft": {
                    "function": "torch.fft.fft2",
                    "norm": "ortho",
                    "shift": "fftshift",
                    "magnitude": "abs",
                    "compression": "log1p",
                    "scale": "divide_by_per_image_max",
                    "repeat_channels": 3,
                },
            },
            "noise_branch": {
                "source": "rgb",
                "residual": {
                    "kernel": RESIDUAL_KERNEL,
                    "per_channel": True,
                    "padding": 1,
                    "post_activation": "tanh",
                },
            },
        },
        "output": {
            "probability_target": "ai",
            "classes": ["ai", "real"],
            "logit_to_probability": "sigmoid(logit / temperature)",
            "threshold_source": "checkpoint_or_calibration",
        },
    }


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


class ConvNeXtBackbone(nn.Module):
    def __init__(self, variant: str = "tiny", pretrained: bool = True):
        super().__init__()
        if variant == "small":
            weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.convnext_small(weights=weights)
        else:
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.convnext_tiny(weights=weights)
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.pool(x).flatten(1)


class AdvancedAIDetector(nn.Module):
    def __init__(self, backbone: str = "tiny", pretrained_backbone: bool = True, metadata_feature_dim: int = 0):
        super().__init__()
        self.backbone_name = backbone
        self.metadata_feature_dim = int(max(metadata_feature_dim, 0))

        if backbone == "tiny":
            self.rgb_branch = TinyBackbone(in_ch=3)
            self.fft_branch = TinyBackbone(in_ch=3)
            self.noise_branch = TinyBackbone(in_ch=3)
        elif backbone == "convnext_tiny":
            self.rgb_branch = ConvNeXtBackbone("tiny", pretrained=pretrained_backbone)
            self.fft_branch = ConvNeXtBackbone("tiny", pretrained=pretrained_backbone)
            self.noise_branch = ConvNeXtBackbone("tiny", pretrained=pretrained_backbone)
        elif backbone == "convnext_small":
            self.rgb_branch = ConvNeXtBackbone("small", pretrained=pretrained_backbone)
            self.fft_branch = ConvNeXtBackbone("small", pretrained=pretrained_backbone)
            self.noise_branch = ConvNeXtBackbone("small", pretrained=pretrained_backbone)
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
        metadata_hidden = 64 if self.metadata_feature_dim > 0 else 0
        if self.metadata_feature_dim > 0:
            self.metadata_branch = nn.Sequential(
                nn.Linear(self.metadata_feature_dim, metadata_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        else:
            self.metadata_branch = None
        self.fusion = nn.Sequential(
            nn.Linear((feat * 3) + metadata_hidden, 768 if feat >= 1280 else 512),
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

    def forward(self, x: torch.Tensor, metadata_features: torch.Tensor | None = None) -> torch.Tensor:
        f_rgb = self.rgb_branch(x)
        f_fft = self.fft_branch(self._fft_features(x))
        f_noise = self.noise_branch(self._noise_residual(x))
        features = [f_rgb, f_fft, f_noise]
        if self.metadata_feature_dim > 0:
            if metadata_features is None:
                metadata_features = torch.zeros(
                    x.shape[0],
                    self.metadata_feature_dim,
                    dtype=x.dtype,
                    device=x.device,
                )
            else:
                metadata_features = metadata_features.to(device=x.device, dtype=x.dtype)
            features.append(self.metadata_branch(metadata_features))
        return self.fusion(torch.cat(features, dim=1)).squeeze(1)


def build_model(
    backbone: str = "tiny",
    pretrained_backbone: bool = True,
    metadata_feature_dim: int = 0,
) -> AdvancedAIDetector:
    return AdvancedAIDetector(
        backbone=backbone,
        pretrained_backbone=pretrained_backbone,
        metadata_feature_dim=metadata_feature_dim,
    )

"""Stage 2 (spatial CNN) backbones (pretrained ResNet-18, MobileNetV3, custom ResNet) and SharedEyeEncoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tv_models

from configs.vision import BackboneConfig


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """Create a 3x3 convolution layer with padding=1 and no bias."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """Create a 1x1 pointwise convolution layer with no bias."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


def build_norm2d(num_channels: int, norm_type: str = "group", num_groups: int = 8) -> nn.Module:
    """Build BatchNorm2d or GroupNorm for 2D CNN feature maps."""
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "group":
        groups = min(num_groups, num_channels)
        while groups > 1 and num_channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(num_groups=groups, num_channels=num_channels)
    raise ValueError(f"Unsupported norm_type '{norm_type}'. Choose 'batch' or 'group'.")


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock (two 3x3 convolutions with residual skip connection)."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        norm_type: str = "group",
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = build_norm2d(planes, norm_type=norm_type, num_groups=num_groups)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = build_norm2d(planes, norm_type=norm_type, num_groups=num_groups)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block: (N, C_in, H, W) -> (N, C_out, H', W')."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """Custom ResNet-18 feature extractor built from scratch (no pretrained library calls).

    Input:
        x: (N, in_channels, H_eye, W_eye)
    Output:
        feats: (N, out_channels) where out_channels == base_channels * 8 (512 by default)
    """

    def __init__(self, config: BackboneConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else BackboneConfig()
        self.inplanes = self.config.base_channels
        self._norm_type = self.config.norm_type
        self._num_groups = self.config.num_groups

        if self.config.small_input_stem:
            self.conv1 = conv3x3(self.config.in_channels, self.inplanes, stride=1)
            self.bn1 = build_norm2d(
                self.inplanes, norm_type=self._norm_type, num_groups=self._num_groups
            )
            self.relu = nn.ReLU(inplace=True)
            self.maxpool: nn.Module = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(
                self.config.in_channels,
                self.inplanes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            self.bn1 = build_norm2d(
                self.inplanes, norm_type=self._norm_type, num_groups=self._num_groups
            )
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        c = self.config.base_channels
        l1, l2, l3, l4 = self.config.layers
        self.layer1 = self._make_layer(c, l1, stride=1)
        self.layer2 = self._make_layer(c * 2, l2, stride=2)
        self.layer3 = self._make_layer(c * 4, l3, stride=2)
        self.layer4 = self._make_layer(c * 8, l4, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_channels = c * 8 * BasicBlock.expansion

        self._init_weights(zero_init_residual=self.config.zero_init_residual)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample: nn.Module | None = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * BasicBlock.expansion, stride=stride),
                build_norm2d(
                    planes * BasicBlock.expansion,
                    norm_type=self._norm_type,
                    num_groups=self._num_groups,
                ),
            )

        layers: list[nn.Module] = [
            BasicBlock(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                norm_type=self._norm_type,
                num_groups=self._num_groups,
            )
        ]
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    self.inplanes,
                    planes,
                    norm_type=self._norm_type,
                    num_groups=self._num_groups,
                )
            )

        return nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual: bool = False) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, BasicBlock) and hasattr(module.bn2, "weight"):
                    nn.init.constant_(module.bn2.weight, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled feature vectors: (N, C, H, W) -> (N, out_channels)."""
        # (N, C, H, W) -> (N, 64, H/2, W/2) [or (N, 64, H, W) if small_input_stem]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # -> (N, 64, H/4, W/4)
        x = self.maxpool(x)

        # -> (N, 64, H/4, W/4)
        x = self.layer1(x)
        # -> (N, 128, H/8, W/8)
        x = self.layer2(x)
        # -> (N, 256, H/16, W/16)
        x = self.layer3(x)
        # -> (N, 512, H/32, W/32)
        x = self.layer4(x)

        # (N, 512, H', W') -> (N, 512, 1, 1)
        x = self.avgpool(x)
        # (N, 512, 1, 1) -> (N, 512)
        x = torch.flatten(x, 1)
        return x


class PretrainedResNet18(nn.Module):
    """Torchvision pretrained ResNet-18 feature extractor with optional early-stage freezing.

    Input:
        x: (N, 3, H_eye, W_eye)
    Output:
        feats: (N, 512)
    """

    def __init__(self, pretrained: bool = True, freeze_early_layers: bool = True) -> None:
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        net = tv_models.resnet18(weights=weights)
        net.fc = nn.Identity()  # type: ignore[assignment]
        self.net = net
        self.out_channels = 512
        self._first_conv: nn.Conv2d = self.net.conv1

        self._frozen_modules: list[nn.Module] = []
        if freeze_early_layers:
            self._frozen_modules = [
                self.net.conv1,
                self.net.bn1,
                self.net.layer1,
                self.net.layer2,
                self.net.layer3,
            ]
            for mod in self._frozen_modules:
                for param in mod.parameters():
                    param.requires_grad = False
                mod.eval()

    def train(self, mode: bool = True) -> PretrainedResNet18:
        super().train(mode)
        if mode:
            for mod in self._frozen_modules:
                mod.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled features: (N, 3, H, W) -> (N, 512)."""
        return self.net(x)


class PretrainedMobileNetV3(nn.Module):
    """Torchvision pretrained MobileNetV3 (small/large) feature extractor.

    Input:
        x: (N, 3, H_eye, W_eye)
    Output:
        feats: (N, 576) for small or (N, 960) for large
    """

    def __init__(
        self,
        variant: str = "mobilenet_v3_small",
        pretrained: bool = True,
        freeze_early_layers: bool = True,
    ) -> None:
        super().__init__()
        if variant == "mobilenet_v3_small":
            weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            net = tv_models.mobilenet_v3_small(weights=weights)
            self.out_channels = 576
            freeze_upto = 9
        elif variant == "mobilenet_v3_large":
            weights = tv_models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            net = tv_models.mobilenet_v3_large(weights=weights)
            self.out_channels = 960
            freeze_upto = 12
        else:
            raise ValueError(f"Unsupported MobileNetV3 variant: {variant}")

        self.features = net.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._first_conv: nn.Conv2d = self.features[0][0]  # type: ignore[assignment]

        self._frozen_modules: list[nn.Module] = []
        if freeze_early_layers:
            for idx in range(min(freeze_upto, len(self.features))):
                mod = self.features[idx]
                for param in mod.parameters():
                    param.requires_grad = False
                mod.eval()
                self._frozen_modules.append(mod)

    def train(self, mode: bool = True) -> PretrainedMobileNetV3:
        super().train(mode)
        if mode:
            for mod in self._frozen_modules:
                mod.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled features: (N, 3, H, W) -> (N, out_channels)."""
        # (N, 3, H, W) -> (N, C_out, H', W')
        x = self.features(x)
        # (N, C_out, H', W') -> (N, C_out, 1, 1) -> (N, C_out)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def build_backbone(config: BackboneConfig) -> nn.Module:
    """Instantiate Stage 2 spatial CNN backbone from BackboneConfig.arch."""
    arch = config.arch.lower()
    if arch == "resnet18":
        return PretrainedResNet18(
            pretrained=config.pretrained,
            freeze_early_layers=config.freeze_early_layers,
        )
    if arch in {"mobilenet_v3_small", "mobilenet_v3_large"}:
        return PretrainedMobileNetV3(
            variant=arch,
            pretrained=config.pretrained,
            freeze_early_layers=config.freeze_early_layers,
        )
    if arch == "custom_resnet":
        return ResNet18(config)
    raise ValueError(f"Unsupported backbone arch: {config.arch}")


class SharedEyeEncoder(nn.Module):
    """Shared-weight eye encoder with visibility-mask suppression.

    Supports pretrained `resnet18`, `mobilenet_v3_small`, `mobilenet_v3_large`,
    and `custom_resnet`.

    Input:
        eye_crops: (B, T, 2, C, H_eye, W_eye)
        eye_mask:  (B, T, 2) with values in {0.0, 1.0} for [v_L, v_R]
    Output:
        frame_feats: (B, T, 2 * D + 2) representing [e_L || e_R || v_L || v_R]
    """

    def __init__(self, config: BackboneConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else BackboneConfig()
        self.backbone = build_backbone(self.config)
        self.embedding_dim = self.config.embedding_dim

        proj_layers: list[nn.Module] = []
        if self.config.dropout > 0.0:
            proj_layers.append(nn.Dropout(p=self.config.dropout))
        out_channels = int(getattr(self.backbone, "out_channels"))
        if self.config.embedding_dim != out_channels:
            proj_layers.append(nn.Linear(out_channels, self.config.embedding_dim))

        self.proj: nn.Module = nn.Sequential(*proj_layers) if proj_layers else nn.Identity()

    def _first_conv_weight(self) -> torch.Tensor | None:
        conv = getattr(self.backbone, "_first_conv", None)
        if conv is None:
            conv = getattr(self.backbone, "conv1", None)
        if isinstance(conv, nn.Conv2d):
            return conv.weight
        return None

    def encode_eye(self, flat_crops: torch.Tensor) -> torch.Tensor:
        """Encode a flattened batch of single-eye crops: (N, C, H_eye, W_eye) -> (N, D)."""
        # (N, C, H_eye, W_eye) -> (N, out_channels)
        feats = self.backbone(flat_crops)
        # (N, out_channels) -> (N, D)
        return self.proj(feats)

    def forward(self, eye_crops: torch.Tensor, eye_mask: torch.Tensor) -> torch.Tensor:
        """Encode paired eye crops, zero-out missing eyes via mask, and append [v_L, v_R]."""
        if eye_crops.ndim != 6 or eye_crops.shape[2] != 2:
            raise ValueError(
                f"Expected eye_crops of shape (B, T, 2, C, H, W), got {tuple(eye_crops.shape)}"
            )
        b, t, num_eyes, c, h, w = eye_crops.shape
        if eye_mask.shape != (b, t, num_eyes):
            raise ValueError(
                f"Expected eye_mask of shape {(b, t, num_eyes)}, got {tuple(eye_mask.shape)}"
            )

        # (B, T, 2, C, H_eye, W_eye) -> (B * T * 2, C, H_eye, W_eye)
        flat_crops = eye_crops.reshape(b * t * num_eyes, c, h, w)
        first_w = self._first_conv_weight()
        if (
            first_w is not None
            and first_w.is_contiguous(memory_format=torch.channels_last)
            and not first_w.is_contiguous(memory_format=torch.contiguous_format)
        ):
            flat_crops = flat_crops.contiguous(memory_format=torch.channels_last)
        # (B * T * 2, C, H_eye, W_eye) -> (B * T * 2, D)
        flat_embeds = self.encode_eye(flat_crops)
        # (B * T * 2, D) -> (B, T, 2, D)
        embeds = flat_embeds.reshape(b, t, num_eyes, self.embedding_dim)

        # (B, T, 2) -> (B, T, 2, 1)
        mask_float = eye_mask.to(dtype=embeds.dtype)
        mask_expanded = mask_float.unsqueeze(-1)
        # (B, T, 2, D) * (B, T, 2, 1) -> (B, T, 2, D)
        masked_embeds = embeds * mask_expanded

        # (B, T, 2, D) -> (B, T, 2 * D) as [e_L || e_R]
        concat_embeds = masked_embeds.reshape(b, t, num_eyes * self.embedding_dim)
        # (B, T, 2 * D) + (B, T, 2) -> (B, T, 2 * D + 2) as [e_L || e_R || v_L || v_R]
        frame_feats = torch.cat([concat_embeds, mask_float], dim=-1)
        return frame_feats

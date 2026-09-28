"""Dependency-minimal, provenance-pinned UNeXt author-family 2-channel model.

Ported from the official MIT UNeXt implementation recorded in
``vendor_unext/PROVENANCE.md``.  It intentionally does not infer or recreate
the underspecified Pro_UNeXt fused-MBConv architecture.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


ADAPTATION_MODULES = (
    'encoder1: official hard-coded Conv2d(3,16,3) -> Conv2d(2,16,3)',
    'final: official configurable Conv2d(16,num_classes,1) -> raw Conv2d(16,1,1)',
)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _init_weights(module: nn.Module) -> None:
    """The initialization used by the official token/shift submodules."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
    elif isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            module.bias.data.zero_()


class _DWConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, channels = x.shape
        x = x.transpose(1, 2).view(batch, channels, height, width)
        return self.dwconv(x).flatten(2).transpose(1, 2)


class _ShiftMLP(nn.Module):
    def __init__(self, dim: int, drop: float = 0., shift_size: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.dwconv = _DWConv(dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)
        self.shift_size, self.pad = shift_size, shift_size // 2
        self.apply(_init_weights)

    def _shift(self, x: torch.Tensor, axis: int, height: int, width: int) -> torch.Tensor:
        batch, _, channels = x.shape
        image = x.transpose(1, 2).view(batch, channels, height, width).contiguous()
        image = F.pad(image, (self.pad, self.pad, self.pad, self.pad), 'constant', 0)
        chunks = torch.chunk(image, self.shift_size, 1)
        shifted = [torch.roll(chunk, offset, axis) for chunk, offset in zip(chunks, range(-self.pad, self.pad + 1))]
        image = torch.cat(shifted, 1)
        image = torch.narrow(image, 2, self.pad, height)
        image = torch.narrow(image, 3, self.pad, width)
        return image.reshape(batch, channels, height * width).transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = self.fc1(self._shift(x, 2, height, width))
        x = self.drop(self.act(self.dwconv(x, height, width)))
        return self.drop(self.fc2(self._shift(x, 3, height, width)))


class _ShiftedBlock(nn.Module):
    def __init__(self, dim: int, drop: float = 0., drop_path: float = 0.):
        super().__init__()
        if drop_path != 0.:
            raise ValueError('official default drop_path_rate is fixed at zero for this comparator')
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _ShiftMLP(dim, drop=drop)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return x + self.drop_path(self.mlp(self.norm2(x), height, width))


class _OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (3, 3)
        self.H, self.W = img_size // 3, img_size // 3
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        _, _, height, width = x.shape
        return self.norm(x.flatten(2).transpose(1, 2)), height, width


class _UNeXtCore(nn.Module):
    """Official standard UNeXt topology with an explicit input/output interface."""
    def __init__(self, input_channels: int, num_classes: int, img_size: int = 256):
        super().__init__()
        if img_size != 256:
            raise ValueError('this fixed experiment accepts native 256-pixel patches only')
        self.encoder1 = nn.Conv2d(input_channels, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)
        self.ebn1, self.ebn2, self.ebn3 = nn.BatchNorm2d(16), nn.BatchNorm2d(32), nn.BatchNorm2d(128)
        self.norm3, self.norm4 = nn.LayerNorm(160), nn.LayerNorm(256)
        self.dnorm3, self.dnorm4 = nn.LayerNorm(160), nn.LayerNorm(128)
        self.block1, self.block2 = nn.ModuleList([_ShiftedBlock(160)]), nn.ModuleList([_ShiftedBlock(256)])
        self.dblock1, self.dblock2 = nn.ModuleList([_ShiftedBlock(160)]), nn.ModuleList([_ShiftedBlock(128)])
        self.patch_embed3 = _OverlapPatchEmbed(img_size // 4, 128, 160)
        self.patch_embed4 = _OverlapPatchEmbed(img_size // 8, 160, 256)
        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.dbn1, self.dbn2 = nn.BatchNorm2d(160), nn.BatchNorm2d(128)
        self.dbn3, self.dbn4 = nn.BatchNorm2d(32), nn.BatchNorm2d(16)
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    @staticmethod
    def _tokens_to_image(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch = tokens.shape[0]
        return tokens.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[-2:]) != (256, 256):
            raise AssertionError('UNeXtAuthor2ch expects BxCx256x256')
        batch = x.shape[0]
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2)); t1 = out
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2)); t2 = out
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2)); t3 = out
        out, height, width = self.patch_embed3(out)
        out = self.norm3(self.block1[0](out, height, width)); t4 = self._tokens_to_image(out, height, width)
        out, height, width = self.patch_embed4(t4)
        out = self._tokens_to_image(self.norm4(self.block2[0](out, height, width)), height, width)
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2), mode='bilinear')) + t4
        _, _, height, width = out.shape
        out = self._tokens_to_image(self.dnorm3(self.dblock1[0](out.flatten(2).transpose(1, 2), height, width)), height, width)
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear')) + t3
        _, _, height, width = out.shape
        out = self._tokens_to_image(self.dnorm4(self.dblock2[0](out.flatten(2).transpose(1, 2), height, width)), height, width)
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear')) + t2
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear')) + t1
        return self.final(F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear')))


class UNeXtOfficial3ch(_UNeXtCore):
    """Three-channel semantic counterpart used only by CPU provenance tests."""
    def __init__(self, num_classes: int = 1):
        super().__init__(input_channels=3, num_classes=num_classes, img_size=256)


class UNeXtAuthor2ch(_UNeXtCore):
    """Pinned author UNeXt topology adapted solely to this experiment interface."""
    adaptation_modules = ADAPTATION_MODULES

    def __init__(self, config):
        if int(config['input_channels']) != 2 or int(config['patch_size']) != 256:
            raise AssertionError('UNeXtAuthor2ch requires the fixed 2-channel native256 protocol')
        super().__init__(input_channels=2, num_classes=1, img_size=256)

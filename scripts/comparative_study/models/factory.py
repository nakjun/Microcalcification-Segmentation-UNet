"""Construct the two pending comparative-study backbone variants.

Every returned model accepts ``[B, 2, H, W]`` and returns raw one-channel
logits.  This module only supplies the protocol's fixed upstream arguments;
it does not add activations, adapters, or initialization.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch.nn as nn

try:  # Package import when used by the comparative-study harness.
    from .vendor_attention.compat.unet_2D import unet_2D
    from .vendor_utnet.compat.utnet import UTNet
except ImportError:  # Direct execution of test_models.py from this directory.
    from vendor_attention.compat.unet_2D import unet_2D
    from vendor_utnet.compat.utnet import UTNet


BACKBONES = ("unet2d", "utnet")


def _require_mapping(config: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    return config


def build_model(config: Mapping[str, object]) -> nn.Module:
    """Build a fixed upstream architecture selected by ``config['backbone']``.

    Optional keys mirror only upstream constructor values.  The comparative
    protocol uses the documented defaults below; accepting the keys makes the
    factory explicit and serializable without changing the model operations.
    """
    config = _require_mapping(config)
    backbone = config.get("backbone")
    if backbone == "unet2d":
        return unet_2D(
            feature_scale=int(config.get("feature_scale", 4)),
            in_channels=int(config.get("in_channels", 2)),
            n_classes=int(config.get("num_classes", 1)),
            is_deconv=bool(config.get("is_deconv", True)),
            is_batchnorm=bool(config.get("batchnorm", True)),
        )
    if backbone == "utnet":
        return UTNet(
            in_chan=int(config.get("in_channels", 2)),
            base_chan=int(config.get("base_chan", 32)),
            num_classes=int(config.get("num_classes", 1)),
            reduce_size=int(config.get("reduce_size", 8)),
            block_list=str(config.get("block_list", "1234")),
            num_blocks=list(config.get("num_blocks", [1, 1, 1, 1])),
            num_heads=list(config.get("heads", [4, 4, 4, 4])),
            projection=str(config.get("projection", "interp")),
            attn_drop=float(config.get("attn_drop", 0.1)),
            proj_drop=float(config.get("proj_drop", 0.1)),
            rel_pos=bool(config.get("rel_pos", True)),
            aux_loss=bool(config.get("aux_loss", config.get("aux", False))),
            maxpool=bool(config.get("maxpool", True)),
        )
    raise ValueError(f"unknown backbone {backbone!r}; expected one of {BACKBONES}")


def parameter_count(model: nn.Module) -> int:
    """Return all trainable and non-trainable parameter elements, not buffers."""
    return sum(parameter.numel() for parameter in model.parameters())

"""CPU parity checks for the vendored comparative-study backbones.

Run with:
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python \
        scripts/comparative_study/models/test_models.py

Each upstream class is imported from its supplied source checkout.  A factory
model built after the same seed must have identical tensors, train/eval logits,
parameter gradients, and BatchNorm buffers.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable

import torch

try:
    from .factory import build_model, parameter_count
except ImportError:
    from factory import build_model, parameter_count


ROOT = Path(__file__).resolve().parents[3]
ATTENTION_SOURCE = Path("/tmp/comparative_attention_source")
UTNET_SOURCE = Path("/tmp/comparative_utnet_source")

CONFIGS = {
    "unet2d": {
        "backbone": "unet2d",
        "feature_scale": 4,
        "in_channels": 2,
        "num_classes": 1,
        "is_deconv": True,
        "batchnorm": True,
    },
    "utnet": {
        "backbone": "utnet",
        "in_channels": 2,
        "base_chan": 32,
        "num_classes": 1,
        "reduce_size": 8,
        "block_list": "1234",
        "num_blocks": [1, 1, 1, 1],
        "heads": [4, 4, 4, 4],
        "projection": "interp",
        "attn_drop": 0.1,
        "proj_drop": 0.1,
        "rel_pos": True,
        "aux_loss": False,
        "maxpool": True,
    },
}


def _clear_top_level_package(package: str) -> None:
    for name in tuple(sys.modules):
        if name == package or name.startswith(package + "."):
            del sys.modules[name]


def _attention_original() -> Callable[[], torch.nn.Module]:
    if not ATTENTION_SOURCE.is_dir():
        raise AssertionError(f"missing supplied attention source: {ATTENTION_SOURCE}")
    _clear_top_level_package("models")
    sys.path.insert(0, str(ATTENTION_SOURCE))
    try:
        source_class = importlib.import_module("models.networks.unet_2D").unet_2D
    finally:
        sys.path.pop(0)
        _clear_top_level_package("models")
    return lambda: source_class(
        feature_scale=4,
        in_channels=2,
        n_classes=1,
        is_deconv=True,
        is_batchnorm=True,
    )


def _utnet_original() -> Callable[[], torch.nn.Module]:
    if not UTNET_SOURCE.is_dir():
        raise AssertionError(f"missing supplied UTNet source: {UTNET_SOURCE}")
    _clear_top_level_package("model")
    sys.path.insert(0, str(UTNET_SOURCE))
    try:
        source_class = importlib.import_module("model.utnet").UTNet
    finally:
        sys.path.pop(0)
        _clear_top_level_package("model")
    return lambda: source_class(
        in_chan=2,
        base_chan=32,
        num_classes=1,
        reduce_size=8,
        block_list="1234",
        num_blocks=[1, 1, 1, 1],
        num_heads=[4, 4, 4, 4],
        projection="interp",
        attn_drop=0.1,
        proj_drop=0.1,
        rel_pos=True,
        aux_loss=False,
        maxpool=True,
    )


def _assert_tensor_maps_equal(
    label: str, left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> None:
    if left.keys() != right.keys():
        raise AssertionError(f"{label}: state keys differ")
    for key in left:
        if not torch.equal(left[key], right[key]):
            delta = (left[key].to(torch.float64) - right[key].to(torch.float64)).abs().max()
            raise AssertionError(f"{label}: mismatch at {key}; max_abs={float(delta)}")


def _new_pair(
    original_ctor: Callable[[], torch.nn.Module], config: dict[str, object]
) -> tuple[torch.nn.Module, torch.nn.Module]:
    torch.manual_seed(421)
    original = original_ctor()
    torch.manual_seed(421)
    vendored = build_model(config)
    _assert_tensor_maps_equal("initial state", original.state_dict(), vendored.state_dict())
    return original, vendored


def _forward(model: torch.nn.Module, inputs: torch.Tensor, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)  # makes UTNet's upstream Dropout replay exactly.
    return model(inputs)


def _run_backbone(name: str, original_ctor: Callable[[], torch.nn.Module]) -> int:
    config = CONFIGS[name]
    torch.manual_seed(77)
    inputs = torch.randn(1, 2, 256, 256)

    original, vendored = _new_pair(original_ctor, config)
    original.eval()
    vendored.eval()
    with torch.no_grad():
        original_eval = _forward(original, inputs, 991)
        vendored_eval = _forward(vendored, inputs, 991)
    if original_eval.shape != (1, 1, 256, 256):
        raise AssertionError(f"{name}: invalid raw-logit shape {tuple(original_eval.shape)}")
    if not torch.equal(original_eval, vendored_eval):
        raise AssertionError(f"{name}: eval logits differ")
    _assert_tensor_maps_equal(f"{name} eval buffers", original.state_dict(), vendored.state_dict())

    original, vendored = _new_pair(original_ctor, config)
    original.train()
    vendored.train()
    original_train = _forward(original, inputs, 992)
    vendored_train = _forward(vendored, inputs, 992)
    if not torch.equal(original_train, vendored_train):
        raise AssertionError(f"{name}: train logits differ")
    original_train.square().mean().backward()
    vendored_train.square().mean().backward()
    for (original_name, original_parameter), (vendored_name, vendored_parameter) in zip(
        original.named_parameters(), vendored.named_parameters(), strict=True
    ):
        if original_name != vendored_name:
            raise AssertionError(f"{name}: parameter names differ")
        if original_parameter.grad is None or vendored_parameter.grad is None:
            raise AssertionError(f"{name}: missing gradient for {original_name}")
        if not torch.equal(original_parameter.grad, vendored_parameter.grad):
            raise AssertionError(f"{name}: gradient differs for {original_name}")
    _assert_tensor_maps_equal(f"{name} train state and BN buffers", original.state_dict(), vendored.state_dict())

    reloaded = build_model(config)
    reloaded.load_state_dict(vendored.state_dict(), strict=True)
    _assert_tensor_maps_equal(f"{name} strict reload", vendored.state_dict(), reloaded.state_dict())
    if not torch.isfinite(vendored_train).all():
        raise AssertionError(f"{name}: non-finite train logits")
    return parameter_count(vendored)


def main() -> None:
    torch.set_num_threads(min(2, torch.get_num_threads()))
    results = {
        "unet2d": _run_backbone("unet2d", _attention_original()),
        "utnet": _run_backbone("utnet", _utnet_original()),
    }
    print("CPU upstream parity passed", results)


if __name__ == "__main__":
    main()

"""Load the official DeepMiCa segmentor without altering its source.

One channel preserves the released architecture. Two channels are an explicit
architecture adaptation for this project's raw-intensity/local-contrast input;
it is not a reproduction of the complete DeepMiCa preprocessing/loss pipeline.
"""
from pathlib import Path
import hashlib
import importlib.util
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'external/DeepMiCa/Step2_Segmentation/_03_Train/UNet.py'


def build_model(in_channels=1):
    if in_channels not in (1, 2):
        raise ValueError('in_channels must be 1 (official) or 2 (adapted)')
    spec = importlib.util.spec_from_file_location('deepmica_official_unet', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.build_unet()
    if in_channels == 2:
        # Fresh initialization: no claim of pretrained-weight equivalence.
        model.e1.conv.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
    return model


def source_sha256():
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()

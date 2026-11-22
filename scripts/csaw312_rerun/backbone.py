"""Fixed CSAW312 backbone builds, epoch training and native image inference."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from calcification_refinement_common import PatchUNet, tiled_probability
from calcification_refinement_v5_common import training_loss

INPUT_CONFIG = dict(patch_size=256, tile_stride=192, contrast_kernel=19,
                    contrast_sigma=3., contrast_gain=16., inference_batch_size=16)


def build(model_name, device='cpu'):
    if model_name in ('proposed', 'patchunet', 'PatchUNet'):
        model = PatchUNet(dict(base_channels=16, input_channels=2))
    elif model_name in ('unext', 'UNeXt'):
        spec = importlib.util.spec_from_file_location('_csaw312_unext', SCRIPTS / 'improvement_unext_deterministic/model.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.UNeXtAuthor2ch(dict(input_channels=2, patch_size=256))
    elif model_name in ('unet2d', 'U-Net2D', 'utnet', 'UTNet'):
        from comparative_study.models.factory import build_model
        model = build_model({'backbone': {'U-Net2D': 'unet2d', 'UTNet': 'utnet'}.get(model_name, model_name)})
    else:
        raise ValueError(f'Unknown backbone: {model_name}')
    model = model.to(device)
    return model, torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)


def train_one_epoch(model, optimizer, examples, variant, epoch, device='cpu', on_update=None):
    """examples yields inputs, target, component weights, valid, metadata."""
    if variant not in ('C', 'L', 'H', 'LH'):
        raise ValueError('Unknown loss variant')
    device = torch.device(device)
    config = dict(positive_BCE_weight=20., overlap_loss='dice',
                  component_aux='L' in variant, background_aux='H' in variant,
                  component_coefficient=.1, background_coefficient=.1)
    model.train()
    batch, count, updates, total = [], 0, 0, 0.
    unique_ids, unique_groups, modes = set(), set(), {}
    positive_patches = 0
    def step():
        nonlocal updates, total, positive_patches
        inputs, targets, weights, valid = [torch.as_tensor(np.stack([r[i] for r in batch]), device=device) for i in range(4)]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = model(inputs)
        loss, stats = training_loss(logits.float(), targets.float(), weights.float(), valid.float(), config, epoch)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite gradient')
        optimizer.step()
        updates += 1
        total += float(loss.detach())
        positive_patches += stats['positive_patches']
        batch.clear()
        if on_update is not None:
            on_update(dict(patches=count, updates=updates, loss=float(loss.detach())))
    for item in examples:
        if len(item) != 5:
            raise ValueError('Expected inputs, target, weights, valid, metadata')
        metadata = item[4]
        unique_ids.add(metadata['image_id'])
        unique_groups.add(metadata.get('group_id', metadata['image_id']))
        mode = metadata['mode']; modes[mode] = modes.get(mode, 0) + 1
        batch.append(item); count += 1
        if len(batch) == 16:
            step()
    if batch:
        step()
    if not updates:
        raise ValueError('Epoch contains no patches')
    return dict(patches=count, updates=updates, training_loss=total/updates,
                positive_patches=positive_patches, negative_patches=count-positive_patches,
                unique_images=len(unique_ids), unique_groups=len(unique_groups), modes=modes)


def predict(model, image, device='cpu'):
    """Return ungated full-native probability averaged over 256/192 tiles."""
    return tiled_probability(model, image, INPUT_CONFIG, torch.device(device))[0]

"""Native pixel refinement; image-only tiling and train-only target sampling."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from calcification_gate_sam_common import load_masks, summarize_segmentation


class ConvBlock(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                         nn.GroupNorm(8, outgoing), nn.SiLU(),
                         nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
                         nn.GroupNorm(8, outgoing), nn.SiLU())


class PatchUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        b = config['base_channels']
        self.enc1 = ConvBlock(config['input_channels'], b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.bridge = ConvBlock(b * 4, b * 8)
        self.dec3 = ConvBlock(b * 12, b * 4)
        self.dec2 = ConvBlock(b * 6, b * 2)
        self.dec1 = ConvBlock(b * 3, b)
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.enc3(F.max_pool2d(b, 2))
        d = self.bridge(F.max_pool2d(c, 2))
        def up(x, skip):
            return torch.cat((F.interpolate(x, size=skip.shape[-2:], mode='bilinear',
                                             align_corners=False), skip), 1)
        d = self.dec3(up(d, c))
        d = self.dec2(up(d, b))
        return self.out(self.dec1(up(d, a)))


def reflected_indices(start, count, length):
    if length == 1:
        return np.zeros(count, dtype=np.int64)
    idx = np.arange(start, start + count) % (2 * length - 2)
    return np.where(idx < length, idx, 2 * length - 2 - idx)


def input_patch(image, y, x, config):
    """Local contrast equals full-image Gaussian filtering, including native edges."""
    size, halo = config['patch_size'], config['contrast_kernel'] // 2
    ys = reflected_indices(y - halo, size + 2 * halo, image.shape[0])
    xs = reflected_indices(x - halo, size + 2 * halo, image.shape[1])
    raw = image[np.ix_(ys, xs)].astype(np.float32) / 255.0
    smooth = cv2.GaussianBlur(raw, (config['contrast_kernel'],) * 2,
                            config['contrast_sigma'], borderType=cv2.BORDER_REFLECT_101)
    raw, smooth = raw[halo:halo + size, halo:halo + size], smooth[halo:halo + size, halo:halo + size]
    residual = np.clip((raw - smooth) * config['contrast_gain'], -1, 1)
    return np.stack((raw * 2 - 1, residual))


def tile_starts(length, size, stride):
    assert 0 < stride <= size
    last = max(0, length - size)
    return sorted(set(range(0, last + 1, stride)) | {last})


@torch.inference_mode()
def tiled_probability(model, image, config, device, callback=None):
    """All positions depend solely on image shape, never labels or predicted boxes."""
    assert image.dtype == np.uint8 and image.ndim == 2
    model.eval()
    h, w = image.shape
    size = config['patch_size']
    tiles = [(y, x) for y in tile_starts(h, size, config['tile_stride'])
             for x in tile_starts(w, size, config['tile_stride'])]
    total = np.zeros((h, w), np.float32)
    count = np.zeros((h, w), np.uint8)
    batch = config['inference_batch_size']
    for start in range(0, len(tiles), batch):
        positions = tiles[start:start + batch]
        inputs = torch.from_numpy(np.stack([input_patch(image, y, x, config) for y, x in positions])).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = model(inputs)
        probability = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]
        assert np.isfinite(probability).all()
        for p, (y, x) in zip(probability, positions):
            hh, ww = min(size, h - y), min(size, w - x)
            total[y:y + hh, x:x + ww] += p[:hh, :ww]
            count[y:y + hh, x:x + ww] += 1
        if callback is not None:
            callback(min(start + batch, len(tiles)), len(tiles))
    assert count.min() >= 1 and count.max() <= 9
    total /= count
    return total, len(tiles)


def target_patch(ys, xs, y, x, size):
    target = np.zeros((1, size, size), np.float32)
    keep = (ys >= y) & (ys < y + size) & (xs >= x) & (xs < x + size)
    target[0, ys[keep] - y, xs[keep] - x] = 1
    return target


def baseline_masks(root, arm, image_id):
    path = Path(root) / 'csaw' / arm / 'gated/cases' / f'{image_id}.json'
    case = json.loads(path.read_text())
    sam, fill, boxes = load_masks(case['mask_path'], case['mask_sha256'])
    assert not np.any(sam & ~fill)
    if not case['gate_pass']:
        assert not sam.any() and not fill.any()
    return sam, fill, case


class TrainSampler:
    def __init__(self, rows, config, centers):
        assert rows and all(row['role'] == 'train' for row in rows), 'Held targets cannot enter sampler'
        self.rows, self.config, self.centers = rows, config, centers
        self.images = {r['image_id']: np.load(r['native_cache_path'], mmap_mode='r') for r in rows}
        self.targets = {}
        for row in rows:
            with np.load(row['target_path']) as target:
                self.targets[row['image_id']] = (target['y'], target['x'])
        self.positive = [r for r in rows if r['positive']]
        assert self.positive

    def sample(self, rng):
        c, size = self.config, self.config['patch_size']
        mode = rng.choice(['positive', 'hard_negative', 'random_breast'],
                          p=[c['sampling_probabilities'][k] for k in ['positive', 'hard_negative', 'random_breast']])
        # Hard negatives reject whole-patch GT, not just the proposed center.
        for attempt in range(128):
            row = rng.choice(self.positive if mode == 'positive' else self.rows)
            ident = row['image_id']
            image, (gy, gx) = self.images[ident], self.targets[ident]
            if mode == 'positive':
                point = int(rng.integers(len(gy)))
                cy, cx = int(gy[point]), int(gx[point])
                jitter = c['positive_jitter']
                cy, cx = cy + int(rng.integers(-jitter, jitter + 1)), cx + int(rng.integers(-jitter, jitter + 1))
            else:
                key = ident + ('_hard' if mode == 'hard_negative' and attempt < 64 else '_breast')
                points = self.centers[key]
                if not len(points):
                    continue
                cy, cx = points[int(rng.integers(len(points)))]
            y, x = int(np.clip(cy - size // 2, 0, max(0, image.shape[0] - size))), int(np.clip(cx - size // 2, 0, max(0, image.shape[1] - size)))
            target = target_patch(gy, gx, y, x, size)
            if mode == 'hard_negative' and target.any():
                continue
            if mode == 'positive':
                assert target.any()
            inputs = input_patch(image, y, x, c)
            # Reflections preserve microscopic appearance and foreground geometry.
            if rng.random() < .5:
                inputs, target = inputs[:, :, ::-1], target[:, :, ::-1]
            if rng.random() < .5:
                inputs, target = inputs[:, ::-1, :], target[:, ::-1, :]
            return inputs.copy(), target.copy(), mode
        raise RuntimeError('Could not obtain a valid training-only patch in 128 attempts')


def training_loss(logits, targets, config):
    logits, targets = logits.float(), targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets,
        pos_weight=logits.new_tensor(config['positive_BCE_weight']))
    p = torch.sigmoid(logits)
    axes = (1, 2, 3)
    positive = targets.sum(axes) > 0
    dice_loss = 1 - (2 * (p * targets).sum(axes) + 1) / (p.sum(axes) + targets.sum(axes) + 1)
    dice = dice_loss[positive].mean() if positive.any() else logits.sum() * 0
    return bce + dice, {'BCE': float(bce.detach()), 'positive_soft_Dice_loss': float(dice.detach()),
                        'positive_patches': int(positive.sum())}


def pixel_counts(tp, predicted, gt):
    fp, fn = predicted - tp, gt - tp
    return {'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'gt_pixels': int(gt),
            'predicted_pixels': int(predicted),
            'Dice': 2 * tp / (predicted + gt) if predicted + gt else None,
            'IoU': tp / (predicted + gt - tp) if predicted + gt - tp else None}


def candidate_metrics(probability, sam, fill, gy, gx, config):
    assert probability.shape == sam.shape == fill.shape
    assert probability.dtype == np.float32 and np.isfinite(probability).all()
    gt = len(gy)
    covered = int(fill[gy, gx].sum())
    methods = {'SAM_baseline': pixel_counts(int(sam[gy, gx].sum()), int(sam.sum()), gt),
               'box_fill_control': pixel_counts(covered, int(fill.sum()), gt)}
    for name in config['constraint_candidates']:
        region = fill if name == 'predicted_boxes' else sam
        values = probability[region]
        foreground_values = probability[gy, gx][region[gy, gx]]
        for threshold in config['threshold_grid']:
            methods[f'{name}@{threshold:g}'] = pixel_counts(int((foreground_values >= threshold).sum()),
                                                          int((values >= threshold).sum()), gt)
    return {'box_covered_gt_pixels': covered, 'methods': methods}


def summarize_cases(cases):
    methods = list(cases[0]['methods']) if cases else []
    flat = [dict(c['methods']) for c in cases]
    covered = sum(c['box_covered_gt_pixels'] for c in cases)
    result = {}
    for method in methods:
        stats = summarize_segmentation(flat, method)
        gt = stats['TP'] + stats['FN']
        stats.update(positive_pixel_recall=stats['TP'] / gt if gt else None,
                     GT_box_coverage=covered / gt if gt else None,
                     box_covered_GT_retention=stats['TP'] / covered if covered else None)
        result[method] = stats
    return result


def select_candidate(summaries, config, epoch):
    choices = []
    for constraint in config['constraint_candidates']:
        for threshold in config['threshold_grid']:
            key = f'{constraint}@{threshold:g}'
            dice = float(np.mean([summaries[a][key]['positive_macro_Dice'] for a in config['arms']]))
            fp = float(np.mean([summaries[a][key]['negative_FP_pixels_per_image'] for a in config['arms']]))
            rank = (dice, -fp, threshold, int(constraint == 'predicted_boxes'), -epoch)
            choices.append({'epoch': epoch, 'constraint': constraint, 'threshold': threshold,
                            'method': key, 'mean_positive_macro_Dice': dice,
                            'mean_negative_FP_pixels_per_image': fp, 'rank': list(rank)})
    return max(choices, key=lambda x: tuple(x['rank']))

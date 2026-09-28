"""V5 component supervision and pixel hard-background loss; native GT unchanged."""
import math
import numpy as np
import torch
import torch.nn.functional as F

from calcification_refinement_common import TrainSampler, input_patch, target_patch
from calcification_refinement_v2_common import component_pixel_groups, training_loss as base_loss


def component_weights(gy, gx, identifiers, sizes, y, x, patch_size):
    """Each fully contained original component has equal total weight in a patch."""
    weights = np.zeros((1, patch_size, patch_size), np.float32)
    keep = (gy >= y) & (gy < y + patch_size) & (gx >= x) & (gx < x + patch_size)
    ids, counts = np.unique(identifiers[keep], return_counts=True)
    full = [int(i) for i, n in zip(ids, counts) if int(n) == sizes[int(i)]]
    for i in full:
        points = keep & (identifiers == i)
        weights[0, gy[points] - y, gx[points] - x] = 1.0 / (len(full) * sizes[i])
    return weights


class V5Sampler(TrainSampler):
    def __init__(self, rows, config, centers):
        super().__init__(rows, config, centers)
        assert config['positive_strategy'] == 'pixel_uniform'
        self.component_info = {}
        for row in self.positive:
            ident = row['image_id']
            gy, gx = self.targets[ident]
            groups = component_pixel_groups(gy, gx, self.images[ident].shape)
            identifiers = np.zeros(len(gy), np.int32)
            for i, group in enumerate(groups):
                identifiers[group] = i
            self.component_info[ident] = identifiers, [len(g) for g in groups]

    def sample(self, rng):
        c, size = self.config, self.config['patch_size']
        mode = rng.choice(['positive', 'hard_negative', 'random_breast'],
                          p=[c['sampling_probabilities'][k] for k in ['positive', 'hard_negative', 'random_breast']])
        for attempt in range(128):
            row = rng.choice(self.positive if mode == 'positive' else self.rows)
            ident = row['image_id']
            image, (gy, gx) = self.images[ident], self.targets[ident]
            if mode == 'positive':
                point = int(rng.integers(len(gy)))
                cy, cx = int(gy[point]), int(gx[point])
                jitter = c['positive_jitter']
                cy += int(rng.integers(-jitter, jitter + 1))
                cx += int(rng.integers(-jitter, jitter + 1))
            else:
                key = ident + ('_hard' if mode == 'hard_negative' and attempt < 64 else '_breast')
                points = self.centers[key]
                if not len(points):
                    continue
                cy, cx = points[int(rng.integers(len(points)))]
            y = int(np.clip(cy - size // 2, 0, max(0, image.shape[0] - size)))
            x = int(np.clip(cx - size // 2, 0, max(0, image.shape[1] - size)))
            target = target_patch(gy, gx, y, x, size)
            if mode == 'hard_negative' and target.any():
                continue
            if mode == 'positive':
                assert target.any()
            inputs = input_patch(image, y, x, c)
            weights = np.zeros_like(target)
            if ident in self.component_info:
                ids, sizes = self.component_info[ident]
                weights = component_weights(gy, gx, ids, sizes, y, x, size)
            valid = np.zeros_like(target)
            valid[:, :min(size, image.shape[0] - y), :min(size, image.shape[1] - x)] = 1
            values = [inputs, target, weights, valid]
            if rng.random() < .5:
                values = [v[:, :, ::-1] for v in values]
            if rng.random() < .5:
                values = [v[:, ::-1, :] for v in values]
            inputs, target, weights, valid = [v.copy() for v in values]
            return inputs, target, mode, weights, valid
        raise RuntimeError('Could not obtain a valid training-only patch in 128 attempts')


def component_loss(logits, weights):
    active = weights.sum((1, 2, 3)) > 0
    if not active.any():
        return logits.sum() * 0
    return (F.softplus(-logits) * weights).sum((1, 2, 3))[active].mean()


def background_eligible(targets, valid):
    halo = F.max_pool2d(targets.float(), 3, stride=1, padding=1) > 0
    return (valid > 0) & ~halo


def background_loss(logits, targets, valid, fraction=.01, minimum=32):
    eligible = background_eligible(targets, valid)
    losses = []
    for z, mask in zip(logits, eligible):
        values = F.softplus(z[mask])
        if values.numel():
            k = min(values.numel(), max(minimum, math.ceil(fraction * values.numel())))
            # Stable order also fixes gradient assignment for exact ties.
            losses.append(torch.sort(values, descending=True, stable=True).values[:k].mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0


def warmup(epoch):
    return min(1., max(0., (epoch - 3) / 3))


def training_loss(logits, targets, weights, valid, config, epoch):
    logits, targets = logits.float(), targets.float()
    base, stats = base_loss(logits, targets, config)
    lc = component_loss(logits, weights) if config['component_aux'] else logits.sum() * 0
    lh = background_loss(logits, targets, valid) if config['background_aux'] else logits.sum() * 0
    scale = warmup(epoch)
    result = base + scale * (config['component_coefficient'] * lc + config['background_coefficient'] * lh)
    stats.update(component_loss=float(lc.detach()), background_loss=float(lh.detach()),
                 auxiliary_scale=scale, component_patches=int((weights.sum((1, 2, 3)) > 0).sum()))
    return result, stats


def failures(metrics, v2, v4b):
    result = {}
    for arm, m in metrics.items():
        r, fp = v2[arm], v4b[arm]
        assert m['TP'] + m['FN'] == r['TP'] + r['FN'] == fp['TP'] + fp['FN']
        assert (m['positive_images'], m['negative_images']) == (r['positive_images'], r['negative_images'])
        assert (m['positive_images'], m['negative_images']) == (fp['positive_images'], fp['negative_images'])
        checks = {'recall_below_v2': m['TP'] >= r['TP'],
                  'Dice_not_above_v2': m['positive_macro_Dice'] > r['positive_macro_Dice'],
                  'positive_overlap_below_v2': m['positive_images_with_TP'] >= r['positive_images_with_TP'],
                  'negative_FP_area_above_v4b': m['negative_FP_pixels'] <= fp['negative_FP_pixels'],
                  'negative_FP_images_above_v4b': m['negative_images_with_FP'] <= fp['negative_images_with_FP']}
        for size in ['1-4', '5-16']:
            assert m['components'][size]['total'] == r['components'][size]['total']
            checks[f'component_{size}_below_v2'] = m['components'][size]['any_overlap'] >= r['components'][size]['any_overlap']
        result[arm] = [key for key, ok in checks.items() if not ok]
    return result


def choose(candidates, v2, v4b):
    feasible = [c for c in candidates if not any(failures(c['metrics'], v2, v4b).values())]
    return min(feasible, key=lambda c: (
        -np.mean([m['positive_macro_Dice'] for m in c['metrics'].values()]),
        np.mean([m['negative_FP_pixels'] / m['negative_images'] for m in c['metrics'].values()]),
        c['epoch'], c['candidate_id'])) if feasible else None

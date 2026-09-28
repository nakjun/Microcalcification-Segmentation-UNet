"""V2 training candidates; preserved v1 model, input transform and negative sampling.

These are implemented building blocks, not a trained or selected v2 model.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from calcification_refinement_common import TrainSampler, input_patch, target_patch


def component_pixel_groups(gy, gx, shape):
    """Pixel-index groups for every eight-connected GT component, including singletons."""
    assert len(gy) == len(gx) and len(gy) > 0
    binary = np.zeros(shape, np.uint8)
    binary[gy, gx] = 1
    assert int(binary.sum()) == len(gy), 'GT coordinates must be unique'
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    identifiers = labels[gy, gx]
    groups = [np.flatnonzero(identifiers == value) for value in range(1, count)]
    assert groups and sum(map(len, groups)) == len(gy)
    return groups


def choose_positive_pixel(rng, groups, count, strategy):
    if strategy == 'pixel_uniform':
        return int(rng.integers(count))
    assert strategy == 'component_uniform'
    component = groups[int(rng.integers(len(groups)))]
    return int(component[int(rng.integers(len(component)))])


def mine_negative_centers(row, probability, gy, gx, patch_size=256,
                          threshold=.3, cell_size=128, maximum=4096):
    """Current-model FP peaks on training images; reject any GT in the full patch.

    Spatial cells keep one confident peak each, so a large bright false-positive
    region cannot occupy the entire center pool. Returns native (y,x) centers.
    """
    assert row['role'] == 'train', 'Mining may only use this fold training images'
    assert probability.ndim == 2 and probability.dtype == np.float32
    assert np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1))
    assert 0 < threshold <= 1 and cell_size > 0 and maximum > 0
    h, w = probability.shape
    target = np.zeros((h, w), np.uint8)
    target[gy, gx] = 1
    integral = cv2.integral(target, sdepth=cv2.CV_32S)
    pool = []
    for by in range(0, h, cell_size):
        for bx in range(0, w, cell_size):
            cell = probability[by:by + cell_size, bx:bx + cell_size]
            dy, dx = np.unravel_index(np.argmax(cell), cell.shape)
            score = float(cell[dy, dx])
            if score < float(np.float32(threshold)):
                continue
            cy, cx = by + int(dy), bx + int(dx)
            y = int(np.clip(cy - patch_size // 2, 0, max(0, h - patch_size)))
            x = int(np.clip(cx - patch_size // 2, 0, max(0, w - patch_size)))
            y1, x1 = min(h, y + patch_size), min(w, x + patch_size)
            gt_count = integral[y1, x1] - integral[y, x1] - integral[y1, x] + integral[y, x]
            if gt_count == 0:
                pool.append((score, cy, cx))
    pool.sort(key=lambda item: (-item[0], item[1], item[2]))
    pool = pool[:maximum]
    return (np.asarray([(y, x) for _, y, x in pool], np.int32).reshape(-1, 2),
            np.asarray([score for score, _, _ in pool], np.float32))


class RecallAwareSampler(TrainSampler):
    """Keep image-uniform sampling; optionally sample components uniformly within it."""
    def __init__(self, rows, config, centers):
        super().__init__(rows, config, centers)
        assert config['positive_strategy'] in ['pixel_uniform', 'component_uniform']
        self.components = {}
        if config['positive_strategy'] == 'component_uniform':
            for row in self.positive:
                ident = row['image_id']
                gy, gx = self.targets[ident]
                self.components[ident] = component_pixel_groups(gy, gx, self.images[ident].shape)

    def sample(self, rng):
        c, size = self.config, self.config['patch_size']
        mode = rng.choice(['positive', 'hard_negative', 'random_breast'],
                          p=[c['sampling_probabilities'][k] for k in ['positive', 'hard_negative', 'random_breast']])
        for attempt in range(128):
            row = rng.choice(self.positive if mode == 'positive' else self.rows)
            ident = row['image_id']
            image, (gy, gx) = self.images[ident], self.targets[ident]
            if mode == 'positive':
                point = choose_positive_pixel(rng, self.components.get(ident), len(gy), c['positive_strategy'])
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
            if rng.random() < .5:
                inputs, target = inputs[:, :, ::-1], target[:, :, ::-1]
            if rng.random() < .5:
                inputs, target = inputs[:, ::-1, :], target[:, ::-1, :]
            return inputs.copy(), target.copy(), mode
        raise RuntimeError('Could not obtain a valid training-only patch in 128 attempts')


def positive_overlap_loss(logits, targets, kind='tversky', fp_weight=.3, fn_weight=.7):
    """Positive-patch overlap term; explicit FP/FN names avoid alpha/beta conventions."""
    logits, targets = logits.float(), targets.float()
    probability = torch.sigmoid(logits)
    axes = (1, 2, 3)
    positive = targets.sum(axes) > 0
    if not positive.any():
        return logits.sum() * 0
    tp = (probability * targets).sum(axes)
    if kind == 'dice':
        loss = 1 - (2 * tp + 1) / (probability.sum(axes) + targets.sum(axes) + 1)
    else:
        assert kind == 'tversky' and fp_weight > 0 and fn_weight > 0
        assert abs(fp_weight + fn_weight - 1) < 1e-8
        fp = (probability * (1 - targets)).sum(axes)
        fn = ((1 - probability) * targets).sum(axes)
        # 0.5 makes symmetric Tversky equal to the v1 Dice smoothing convention.
        loss = 1 - (tp + .5) / (tp + fp_weight * fp + fn_weight * fn + .5)
    return loss[positive].mean()


def training_loss(logits, targets, config):
    logits, targets = logits.float(), targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets,
        pos_weight=logits.new_tensor(config['positive_BCE_weight']))
    overlap = positive_overlap_loss(logits, targets, config['overlap_loss'],
                                   config.get('fp_weight', .3), config.get('fn_weight', .7))
    loss = bce + overlap
    return loss, {'BCE': float(bce.detach()), 'positive_overlap_loss': float(overlap.detach()),
                  'positive_patches': int((targets.sum((1, 2, 3)) > 0).sum())}


def choose_recall_constrained(candidates, references):
    """Development candidate only: retain recall and positive-image overlap, bound FP.

    The reference is the SAME cohort's fixed SAM. It is a regression floor, not
    a clinical target or an independent validation certificate. No feasible
    candidate returns None, with no relaxed fallback.
    """
    feasible = []
    for candidate in candidates:
        valid = True
        for arm, reference in references.items():
            value = candidate['metrics'][arm]
            assert value['TP'] + value['FN'] == reference['TP'] + reference['FN']
            assert value['positive_images'] == reference['positive_images']
            assert value['negative_images'] == reference['negative_images']
            valid &= value['TP'] >= reference['TP']
            valid &= value['positive_images_with_TP'] >= reference['positive_images_with_TP']
            valid &= value['negative_FP_pixels'] <= reference['negative_FP_pixels']
            valid &= value['negative_images_with_FP'] <= reference['negative_images_with_FP']
        if valid:
            dice = np.mean([value['positive_macro_Dice'] for value in candidate['metrics'].values()])
            fp = np.mean([value['negative_FP_pixels'] / value['negative_images']
                          for value in candidate['metrics'].values()])
            feasible.append((float(dice), -float(fp), candidate['candidate_id'], candidate))
    return max(feasible, key=lambda item: item[:3])[3] if feasible else None

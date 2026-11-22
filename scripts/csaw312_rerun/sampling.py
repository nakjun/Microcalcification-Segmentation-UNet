"""Training-only fresh patch draws for CSAW312 backbones and specialists."""
from collections import OrderedDict
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from calcification_refinement_common import input_patch, target_patch
from calcification_refinement_v5_common import component_weights

INPUT_CONFIG = dict(patch_size=256, contrast_kernel=19, contrast_sigma=3., contrast_gain=16.)


def _read(path):
    path = Path(path)
    if path.suffix.lower() == '.npy':
        a = np.load(path, mmap_mode='r', allow_pickle=False)
    else:
        with Image.open(path) as f:
            a = np.array(f)
    if a.ndim != 2 or a.dtype != np.uint8 or not a.size:
        raise ValueError(f'Expected nonempty uint8 grayscale: {path}')
    return a


def breast_centers(image, seed=42, stride=16, maximum=4096):
    """Deterministic capped grid of native pixels whose intensity exceeds 40."""
    image = np.asarray(image)
    if image.ndim != 2 or image.dtype != np.uint8 or not image.size or stride < 1 or maximum < 1:
        raise ValueError('Invalid image or breast center settings')
    ys, xs = np.nonzero(image[::stride, ::stride] > 40)
    points = np.column_stack((ys * stride, xs * stride)).astype(np.int32)
    if len(points) > maximum:
        rng = np.random.default_rng(seed)
        points = points[np.sort(rng.choice(len(points), maximum, replace=False))]
    return points


class PatchSampler:
    """Center pools use id+'_hard' and id+'_breast', arrays of (y,x).

    Caller guarantees rows are exactly the current training split. No validation
    or test row is accepted. Optional native_cache_path uses read-only mmap.
    Caller owns and persists numpy Generator state and the current center pools.
    """
    def __init__(self, rows, centers, cache_size=4):
        if not rows or any(r.get('role') != 'train' for r in rows):
            raise ValueError('Only explicit training rows are permitted')
        self.rows = list(rows)
        self.by_id = {r['image_id']: r for r in rows}
        if len(self.by_id) != len(rows) or cache_size < 1:
            raise ValueError('Duplicate ID or invalid cache size')
        self.positive = [r for r in rows if r['positive']]
        if not self.positive:
            raise ValueError('Training partition has no positive images')
        self.cache_size = cache_size
        self.images = OrderedDict()
        self.targets = {}
        self.centers = {}
        self.reset_exposure()
        allowed = {i + suffix for i in self.by_id for suffix in ('_hard', '_breast')}
        if set(centers) != allowed:
            raise ValueError('Center keys must match current training IDs exactly')
        for key in sorted(allowed):
            self.centers[key] = self._center_array(centers[key])

    def reset_exposure(self):
        self.last_metadata = None
        self._exposure_images = set()
        self._exposure_groups = set()
        self._exposure_components = set()
        self._exposure_modes = {}
        self._positive_patches = 0
        self._exposure_draws = 0

    def exposure(self):
        return dict(patches=self._exposure_draws, unique_images=len(self._exposure_images),
                    unique_groups=len(self._exposure_groups), unique_components=len(self._exposure_components),
                    positive_patches=self._positive_patches,
                    negative_patches=self._exposure_draws-self._positive_patches,
                    modes=dict(self._exposure_modes))

    @staticmethod
    def _center_array(points):
        a = np.asarray(points)
        if a.size == 0:
            return np.empty((0, 2), np.int32)
        if a.ndim != 2 or a.shape[1] != 2 or not np.issubdtype(a.dtype, np.integer):
            raise ValueError('Centers must be integer (y,x) array')
        return a.astype(np.int32, copy=True)

    def set_hard_centers(self, centers):
        """Replace all hard pools atomically; mapping keys are training image IDs."""
        if set(centers) != set(self.by_id):
            raise ValueError('Hard center IDs must equal the current training IDs')
        new = {key + '_hard': self._center_array(value) for key, value in centers.items()}
        self.centers.update(new)

    def _image(self, row):
        ident = row['image_id']
        if ident not in self.images:
            self.images[ident] = _read(row.get('native_cache_path') or row['source_image'])
            if len(self.images) > self.cache_size:
                self.images.popitem(last=False)
        else:
            self.images.move_to_end(ident)
        return self.images[ident]

    def _target(self, row, shape):
        ident = row['image_id']
        if ident not in self.targets:
            mask = _read(row['source_mask'])
            if mask.shape != shape:
                raise ValueError('Image and training mask shape mismatch')
            mask = (mask > 0).astype(np.uint8)
            if bool(mask.any()) != bool(row['positive']):
                raise ValueError('Inventory label does not match training mask')
            gy, gx = np.nonzero(mask)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            identifiers = labels[gy, gx] - 1
            sizes = stats[1:, cv2.CC_STAT_AREA].tolist()
            if n - 1 != len(sizes) or sum(sizes) != len(gy):
                raise AssertionError('Invalid component count')
            self.targets[ident] = (gy, gx, identifiers, sizes)
        return self.targets[ident]

    def sample(self, rng, specialist=False):
        mode = rng.choice(['positive', 'uniform'] if specialist else ['positive', 'hard_negative', 'random_breast'],
                          p=[.5, .5] if specialist else [.5, .25, .25])
        size = 256
        for attempt in range(128):
            candidates = self.positive if mode == 'positive' else self.rows
            row = candidates[int(rng.integers(len(candidates)))]
            ident = row['image_id']
            image = self._image(row)
            gy, gx, ids, sizes = self._target(row, image.shape)
            if mode == 'positive':
                point = int(rng.integers(len(gy)))
                cy = int(gy[point]) + int(rng.integers(-64, 65))
                cx = int(gx[point]) + int(rng.integers(-64, 65))
                y = int(np.clip(cy - size//2, 0, max(0, image.shape[0]-size)))
                x = int(np.clip(cx - size//2, 0, max(0, image.shape[1]-size)))
            elif mode == 'uniform':
                y = int(rng.integers(max(0, image.shape[0]-size)+1))
                x = int(rng.integers(max(0, image.shape[1]-size)+1))
            else:
                suffix = '_hard' if mode == 'hard_negative' and attempt < 64 else '_breast'
                points = self.centers[ident + suffix]
                if not len(points):
                    continue
                cy, cx = points[int(rng.integers(len(points)))]
                if cy < 0 or cy >= image.shape[0] or cx < 0 or cx >= image.shape[1]:
                    raise ValueError(f'Center outside training image: {ident}')
                y = int(np.clip(cy - size//2, 0, max(0, image.shape[0]-size)))
                x = int(np.clip(cx - size//2, 0, max(0, image.shape[1]-size)))
            target = target_patch(gy, gx, y, x, size)
            if mode == 'hard_negative' and target.any():
                continue
            if mode == 'positive' and not target.any():
                raise AssertionError('Positive patch lost sampled GT')
            valid = np.zeros_like(target)
            h, w = min(size, image.shape[0]-y), min(size, image.shape[1]-x)
            valid[:, :h, :w] = 1
            if specialist:
                patch = np.zeros((size, size), np.uint8)
                patch[:h, :w] = image[y:y+h, x:x+w]
                values = [patch[None], target]
            else:
                inputs = input_patch(image, y, x, INPUT_CONFIG)
                weights = component_weights(gy, gx, ids, sizes, y, x, size)
                values = [inputs, target, weights, valid]
            flip_h, flip_v = rng.random() < .5, rng.random() < .5
            if flip_h:
                values = [v[:, :, ::-1] for v in values]
            if flip_v:
                values = [v[:, ::-1, :] for v in values]
            values = [v.copy() for v in values]
            metadata = dict(image_id=ident, group_id=row.get('split_group_id', row.get('group_id', ident)),
                            mode=str(mode), y=y, x=x, flip_h=flip_h, flip_v=flip_v)
            self.last_metadata = metadata
            self._exposure_draws += 1
            self._exposure_images.add(ident)
            self._exposure_groups.add(metadata['group_id'])
            self._exposure_modes[str(mode)] = self._exposure_modes.get(str(mode), 0) + 1
            self._positive_patches += int(bool(target.any()))
            inside = (gy >= y) & (gy < y+size) & (gx >= x) & (gx < x+size)
            self._exposure_components.update((ident, int(i)) for i in np.unique(ids[inside]))
            if specialist:
                return values[0][0], values[1][0].astype(np.uint8)
            return (*values, metadata)
        raise RuntimeError('No valid training-only patch in 128 attempts')

    def epoch(self, rng, specialist=False, draws=4096):
        if draws < 1:
            raise ValueError('draws must be positive')
        self.reset_exposure()
        for _ in range(draws):
            yield self.sample(rng, specialist=specialist)

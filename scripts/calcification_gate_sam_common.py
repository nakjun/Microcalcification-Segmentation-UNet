"""Native ROI adapter and fixed predicted-box SAM; no GT drives inference."""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from calcification_gate_common import atomic_json, checksum, pixel_hash
from prepare_retinanet_roi import breast_roi, transform_boxes


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def atomic_copy(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def disk_guard(config, extra=0):
    free = shutil.disk_usage(config['root']).free
    required = int(config['disk_reserve_gib'] * 2**30) + extra
    if free < required:
        raise RuntimeError(f'Disk reserve violated: {free} bytes free, require {required}')


def freeze_detector_pair(config, arm):
    """Snapshot both atomic training outputs without stopping or changing training."""
    source = Path(config['detector_root']) / f'{arm}_seed42'
    out = Path(config['root']) / 'frozen' / arm
    out.mkdir(parents=True, exist_ok=True)
    marker = out / 'snapshot.json'
    if marker.exists():
        info = json.loads(marker.read_text())
        for name, digest in info['files'].items():
            assert checksum(out / name) == digest, f'Frozen detector snapshot changed: {arm}/{name}'
        return info
    disk_guard(config, 600 * 2**20)
    for _ in range(8):
        before = {name: checksum(source / name) for name in ['best.pt', 'val_internal.pt']}
        for name in before:
            atomic_copy(source / name, out / name)
        after = {name: checksum(source / name) for name in before}
        if before != after or any(checksum(out / name) != before[name] for name in before):
            time.sleep(1)
            continue
        checkpoint = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
        validation = torch.load(out / 'val_internal.pt', map_location='cpu', weights_only=False)
        best_json = json.loads((source / 'best.json').read_text())
        if checkpoint['metrics'] != validation['metrics'] or checkpoint['epoch'] != best_json['epoch']:
            time.sleep(1)
            continue
        info = {'arm': arm, 'epoch': checkpoint['epoch'], 'files': before,
                'threshold': float(validation['metrics']['threshold']), 'seed': config['seed'],
                'source': str(source), 'snapshot_at_unix': time.time(),
                'detector_training_complete_at_snapshot': (source / 'complete.json').exists(),
                'architecture': 'train_retinanet_roi.Detector: official RetinaNet, EfficientNet B5, high-resolution ROI',
                'scope': 'fixed intermediate checkpoint; original validation-selected operating point',
                'original_metrics': validation['metrics']}
        atomic_json(marker, info)
        return info
    raise RuntimeError(f'Could not obtain coherent checkpoint/prediction pair for {arm}')


def raw_png(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(f'Expected unchanged uint8 grayscale CSAW PNG: {path}')
    return image


def png_roi(image):
    """Adapt displayed PNG to the unchanged high-resolution detector/gate input."""
    assert image.ndim == 2 and image.dtype == np.uint8 and min(image.shape) > 10
    a = image[5:-5, 5:-5].astype(np.float32)
    a -= a.min()
    if not a.max() > 0:
        raise ValueError('Constant image cannot produce a breast ROI')
    a = (a / a.max() * 255).astype(np.uint8)
    x0, y0, x1, y1 = breast_roi(a)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Empty image-only breast ROI')
    roi = [x0 + 5, y0 + 5, x1 + 5, y1 + 5]
    resized = cv2.resize(a[y0:y1, x0:x1], (912, 1520), interpolation=cv2.INTER_AREA)
    padded = np.zeros((1536, 928), dtype=np.uint8)
    padded[:1520, :912] = resized
    return resized, padded, roi


def native_dicom(path, shape, roi_record=None):
    d = pydicom.dcmread(path)
    raw = d.pixel_array
    assert raw.shape == tuple(shape)
    if roi_record is not None:
        # Rebuild the audited gate/detector ROI from this same decode before SAM display mapping.
        roi_image = raw[5:-5, 5:-5].astype(np.float32)
        if str(d.PhotometricInterpretation) == 'MONOCHROME1':
            roi_image = roi_image.max() - roi_image
        roi_image -= roi_image.min()
        if not roi_image.max() > 0:
            raise ValueError(f'Invalid audited ROI source: {path}')
        roi_image = (roi_image / roi_image.max() * 255).astype(np.uint8)
        x0, y0, x1, y1 = breast_roi(roi_image)
        assert [x0 + 5, y0 + 5, x1 + 5, y1 + 5] == roi_record['roi']
        resized = cv2.resize(roi_image[y0:y1, x0:x1], (912, 1520), interpolation=cv2.INTER_AREA)
        assert pixel_hash(resized) == roi_record['roi_sha256'], 'DICOM differs from audited gate/detector ROI'
    a = raw.astype(np.float32) * float(d.get('RescaleSlope', 1)) + float(d.get('RescaleIntercept', 0))
    valid = np.isfinite(a)
    if 'PixelPaddingValue' in d:
        p = float(d.PixelPaddingValue)
        q = float(d.get('PixelPaddingRangeLimit', p))
        valid &= ~((raw >= min(p, q)) & (raw <= max(p, q)))
    if not valid.any():
        raise ValueError(f'No valid DICOM pixels: {path}')
    lo, hi = np.percentile(a[valid], [.5, 99.5])
    if not hi > lo:
        raise ValueError(f'Degenerate DICOM intensity: {path}')
    a = np.clip((a - lo) / (hi - lo), 0, 1)
    if str(d.PhotometricInterpretation) == 'MONOCHROME1':
        a = 1 - a
    elif str(d.PhotometricInterpretation) != 'MONOCHROME2':
        raise ValueError('Unexpected DICOM photometric interpretation')
    a[~valid] = 0
    return (a * 255).round().astype(np.uint8)


def selected_boxes(predictions, threshold, height, width):
    p = np.asarray(predictions).reshape(-1, 5).copy()
    if not np.isfinite(p).all() or not np.isfinite(threshold):
        raise ValueError('Nonfinite prediction or threshold')
    p = p[p[:, 4] >= threshold]
    p[:, [0, 2]] = p[:, [0, 2]].clip(0, width)
    p[:, [1, 3]] = p[:, [1, 3]].clip(0, height)
    if len(p) and not ((p[:, 2] > p[:, 0]) & (p[:, 3] > p[:, 1])).all():
        raise ValueError('Degenerate detector box after clipping')
    return p


def crop_bounds(box, height, width, config):
    x, y, u, v = [float(z) for z in box[:4]]
    side = max(config['crop_min_side'], int(max(u - x, v - y) * config['crop_context_factor']))
    cx, cy = int((x + u) // 2), int((y + v) // 2)
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    x1, y1 = min(width, cx + (side + 1) // 2), min(height, cy + (side + 1) // 2)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Empty predicted-box crop')
    prompt = np.asarray([x - x0, y - y0, u - x0, v - y0], dtype=np.float64)
    return (x0, y0, x1, y1), prompt


def box_fill_mask(shape, boxes):
    mask = np.zeros(shape, dtype=bool)
    h, w = shape
    for x, y, u, v, _ in boxes:
        mask[max(0, int(np.floor(y))):min(h, int(np.ceil(v))),
             max(0, int(np.floor(x))):min(w, int(np.ceil(u)))] = True
    return mask


@torch.inference_mode()
def infer_sam(predictor, image, boxes, config):
    assert image.ndim == 2 and image.dtype == np.uint8
    union = np.zeros(image.shape, dtype=bool)
    fill = box_fill_mask(image.shape, boxes)
    calls, prompts = 0, []
    for box in boxes:
        bounds, prompt = crop_bounds(box, *image.shape, config)
        x0, y0, x1, y1 = bounds
        predictor.set_image(cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_GRAY2RGB))
        masks, quality, _ = predictor.predict(box=prompt, multimask_output=False)
        assert masks.shape == (1, y1 - y0, x1 - x0) and masks.dtype == bool
        union[y0:y1, x0:x1] |= masks[0]
        calls += 1
        prompts.append({'crop_xyxy': list(bounds), 'box_in_crop_xyxy': prompt.tolist(),
                        'sam_predicted_quality': float(quality[0])})
    union &= fill
    assert not np.any(union & ~fill)
    return union, fill, {'sam_calls': calls, 'prompts': prompts}


def conditional_sam(predictor, image_loader, shape, boxes, passed, config):
    """The gate and predicted boxes determine whether image decoding/SAM can run."""
    if not passed or not len(boxes):
        empty = np.zeros(shape, dtype=bool)
        return empty, empty.copy(), {'sam_calls': 0, 'prompts': [],
                                    'skip_reason': 'gate_rejected' if not passed else 'no_predicted_boxes'}
    image = image_loader()
    assert image.shape == tuple(shape)
    return infer_sam(predictor, image, boxes, config)


def save_masks(path, mask, box_fill, boxes):
    assert mask.dtype == box_fill.dtype == bool and mask.shape == box_fill.shape
    atomic_npz(path, mask=np.packbits(mask), box_fill=np.packbits(box_fill),
               shape=np.asarray(mask.shape, dtype=np.int64), boxes=np.asarray(boxes))
    return checksum(path)


def load_masks(path, digest=None):
    if digest is not None:
        assert checksum(path) == digest, f'Mask changed: {path}'
    with np.load(path, allow_pickle=False) as z:
        shape = tuple(int(n) for n in z['shape'])
        count = int(np.prod(shape))
        assert z['mask'].dtype == z['box_fill'].dtype == np.uint8
        assert len(z['mask']) == len(z['box_fill']) == (count + 7) // 8
        mask = np.unpackbits(z['mask'], count=count).reshape(shape).astype(bool)
        fill = np.unpackbits(z['box_fill'], count=count).reshape(shape).astype(bool)
        boxes = z['boxes'].copy()
    assert not np.any(mask & ~fill)
    return mask, fill, boxes


def segmentation_counts(pred, gt):
    assert pred.dtype == gt.dtype == bool and pred.shape == gt.shape
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    return {'TP': tp, 'FP': fp, 'FN': fn, 'gt_pixels': tp + fn, 'predicted_pixels': tp + fp,
            'Dice': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            'IoU': tp / (tp + fp + fn) if tp + fp + fn else None}


def summarize_segmentation(cases, method):
    records = [c[method] for c in cases]
    positive = [r for r in records if r['gt_pixels'] > 0]
    negative = [r for r in records if r['gt_pixels'] == 0]
    tp, fp, fn = [sum(r[k] for r in records) for k in ['TP', 'FP', 'FN']]
    return {'positive_images': len(positive), 'negative_images': len(negative),
            'positive_macro_Dice': float(np.mean([r['Dice'] for r in positive])) if positive else None,
            'positive_macro_IoU': float(np.mean([r['IoU'] for r in positive])) if positive else None,
            'all_micro_Dice': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            'all_micro_IoU': tp / (tp + fp + fn) if tp + fp + fn else None, 'TP': tp, 'FP': fp, 'FN': fn,
            'negative_image_FP_rate': sum(r['FP'] > 0 for r in negative) / len(negative) if negative else None,
            'negative_FP_pixels_per_image': sum(r['FP'] for r in negative) / len(negative) if negative else None}


def load_sam(config, device):
    sys.path.insert(0, str(Path('external/segment-anything').resolve()))
    from segment_anything import SamPredictor, sam_model_registry
    model = sam_model_registry[config['sam_model']](checkpoint=config['sam_weights'])
    return SamPredictor(model.requires_grad_(False).eval().to(device))


def load_gate(checkpoint_path, device):
    from extract_calcification_features import build_encoder
    from train_calcification_gate_finetune import TailGate
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = saved['model']
    arrays = {'mean': state['feature_mean'].numpy(), 'scale': state['feature_scale'].numpy(),
              'coef': state['head.weight'][0].numpy(), 'intercept': state['head.bias'][0].numpy()}
    encoder = build_encoder(saved['config'], 'cpu')
    gate = TailGate(encoder, arrays, saved['config'])
    gate.load_state_dict(state, strict=True)
    encoder.requires_grad_(False).eval().to(device)
    gate.requires_grad_(False).eval().to(device)
    return encoder, gate, saved['config']


@torch.inference_mode()
def gate_logits(encoder, gate, images, gate_config, device):
    from train_calcification_gate_finetune import prefix_features
    scores = gate(prefix_features(encoder, images, gate_config, device)).double().cpu().numpy()
    assert np.isfinite(scores).all()
    return scores


def load_detector(arm, checkpoint_path, device):
    # Intentionally import the current high-resolution RetinaNet, never the old RT-DETR adapter.
    from train_retinanet_roi import Detector
    model = Detector(arm)
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True)
    return model.requires_grad_(False).eval().to(device)


@torch.inference_mode()
def detector_prediction(model, padded, roi, device):
    return detector_predictions(model, [padded], [roi], device)[0]


@torch.inference_mode()
def detector_predictions(model, padded_batch, rois, device):
    from train_retinanet_roi import normalize
    images = torch.from_numpy(np.stack(padded_batch))[:, None]
    predictions = model.predict(normalize(images, device))
    for pred, roi in zip(predictions, rois):
        # Match the original validator's assignment and FP32 coordinate rounding exactly.
        pred[:, :4] = transform_boxes(pred[:, :4], roi, inverse=True)
        assert pred.dtype == np.float32 and np.isfinite(pred).all()
    return predictions

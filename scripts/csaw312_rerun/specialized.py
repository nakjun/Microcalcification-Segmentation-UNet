"""CSAW312 specialist adapters. Scheduling, sampling and checkpointing belong to caller.

Training patches are (256x256 uint8 grayscale, binary mask). The caller controls RNG,
constructs fresh 4096-patch epochs, applies paired flips, and persists all RNG,
model and optimizer states. Inference accepts image only; no GT prompts.
"""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    path = Path(path).resolve()
    if name in sys.modules:
        if Path(sys.modules[name].__file__).resolve() != path:
            raise RuntimeError(f'Conflicting module {name}; use a separate process')
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules[name]
        raise
    return mod


def _deep():
    return _load('_csaw312_deepmica', ROOT / 'DeepMiCa/pipeline.py')


def _yolo():
    d = ROOT / 'YOLO-HLSAM'
    _load('upstream', d / 'upstream.py').activate()
    _load('decoder', d / 'decoder.py')
    return _load('_csaw312_yolo_model', d / 'model.py')


def _image(image):
    a = np.asarray(image)
    if a.ndim != 2 or a.dtype != np.uint8 or not a.size:
        raise ValueError('Expected nonempty uint8 grayscale image')
    return np.ascontiguousarray(a)


def _mask(mask, shape):
    a = np.asarray(mask)
    if a.shape != shape or not np.isin(a, (0, 1, 255)).all():
        raise ValueError('Expected aligned binary mask')
    return np.ascontiguousarray(a > 0, dtype=np.uint8)


def boxes_from_mask(mask):
    """8-connected component boxes in native xyxy, exclusive upper bounds."""
    mask = _mask(mask, np.asarray(mask).shape)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = np.empty((n - 1, 4), dtype=np.float32)
    for i, (x, y, w, h, _) in enumerate(stats[1:]):
        result[i] = (x, y, x + w, y + h)
    return result


def _rgb(a):
    return np.repeat(a[..., None], 3, axis=2)


def deep_input(image):
    return torch.from_numpy(_image(image).astype(np.float32) / 255.)[None]


class DeterministicGlobalMaxPool(torch.nn.Module):
    """Global max with the first-index tie gradient of AdaptiveMaxPool2d(1)."""
    def forward(self, x):
        return x.flatten(2).max(dim=2).values[:, :, None, None]


def _deterministic_global_max(model):
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.AdaptiveMaxPool2d) and child.output_size in (1, (1, 1)):
                setattr(parent, name, DeterministicGlobalMaxPool())
    return model


def build(stage, device='cpu', sam_checkpoint=None, model_type='vit_b'):
    """Return model, optimizer, criterion; stage is deepmica/detector/segmentor."""
    if stage == 'deepmica':
        model = _deep().build_model(1).to(device)
        return model, torch.optim.SGD(model.parameters(), lr=.01, momentum=.99), _deep().topk_loss
    yolo = _yolo()
    if stage == 'detector':
        import yaml
        from utils.loss import ComputeLoss
        model = _deterministic_global_max(yolo.build_detector()).to(device)
        model.hyp = yaml.safe_load((_load('upstream', ROOT / 'YOLO-HLSAM/upstream.py').SOURCE / 'data/hyp.scratch.p5.yaml').read_text())
        model.gr = 1.0
        criterion = ComputeLoss(model)
    elif stage == 'segmentor':
        if sam_checkpoint is None or not Path(sam_checkpoint).is_file():
            raise ValueError('A SAM checkpoint is required for segmentor')
        model = yolo.Segmentor(model_type, sam_checkpoint).to(device)
        criterion = None
    else:
        raise ValueError(f'Unknown stage: {stage}')
    return model, torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4), criterion


def _detector_sample(image, boxes, device, size=640):
    from utils.datasets import letterbox
    image, ratio, pad = letterbox(_rgb(image), new_shape=size, auto=False)
    x = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).to(device).float()[None] / 255
    b = torch.as_tensor(boxes, device=device).clone()
    if len(b):
        b = b * b.new_tensor([ratio[0], ratio[1], ratio[0], ratio[1]]) + b.new_tensor([pad[0], pad[1], pad[0], pad[1]])
        targets = torch.cat([b.new_zeros((len(b), 2)), (b[:, :2] + b[:, 2:]) / (2 * size), (b[:, 2:] - b[:, :2]) / size], 1)
    else:
        targets = x.new_zeros((0, 6))
    return x, targets


def _step(loss, model, optimizer):
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise FloatingPointError('Nonfinite gradient')
    optimizer.step()


def train_one_epoch(stage, model, optimizer, criterion, patches, device='cpu', on_update=None):
    """Consume caller-provided fresh augmented patches; return update statistics."""
    model.train()
    count = updates = skipped = 0
    total = 0.
    batch_x, batch_y = [], []
    def deep_step():
        nonlocal updates, total
        x = torch.stack(batch_x).to(device)
        y = torch.stack(batch_y).to(device)
        loss = criterion(model(x), y)
        _step(loss, model, optimizer)
        total += float(loss.detach()); updates += 1
        batch_x.clear(); batch_y.clear()
        if on_update is not None:
            on_update(dict(patches=count, updates=updates, skipped=skipped, loss=float(loss.detach())))
    for image, mask in patches:
        image = _image(image)
        mask = _mask(mask, (256, 256))
        count += 1
        if stage == 'deepmica':
            if image.shape != (256, 256):
                raise ValueError('DeepMiCa training patches must be 256x256')
            batch_x.append(deep_input(image))
            batch_y.append(torch.from_numpy(mask.astype(np.float32))[None])
            if len(batch_x) == 16:
                deep_step()
            continue
        if image.shape != (256, 256):
            raise ValueError('YOLO training patches must be 256x256')
        boxes = boxes_from_mask(mask)
        if stage == 'detector':
            x, targets = _detector_sample(image, boxes, device)
            loss, _ = criterion(model(x), targets)
        elif stage == 'segmentor':
            if not len(boxes):
                skipped += 1
                continue
            box = boxes[np.random.randint(len(boxes))]
            logits = model.logits(_rgb(image), box[None])
            x1, y1, x2, y2 = box.astype(int)
            target = np.zeros_like(mask, dtype=np.float32)
            target[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            target = torch.as_tensor(target, device=device)[None, None]
            probability = logits.sigmoid()
            loss = F.binary_cross_entropy_with_logits(logits, target) + 1 - (2 * (probability * target).sum() + 1) / (probability.sum() + target.sum() + 1)
        else:
            raise ValueError(f'Unknown stage: {stage}')
        _step(loss, model, optimizer)
        total += float(loss.detach()); updates += 1
        if on_update is not None and count % 64 == 0:
            on_update(dict(patches=count, updates=updates, skipped=skipped, loss=float(loss.detach())))
    if batch_x:
        deep_step()
    if not count or not updates:
        raise ValueError('Epoch has no training updates')
    return {'patches': count, 'updates': updates, 'negative_patches_skipped': skipped, 'training_loss': total / updates}


@torch.inference_mode()
def detect(model, image, device='cpu', confidence=.25, iou=.45, max_detections=100, size=640):
    """Image-only YOLO detection, returning native-coordinate boxes and scores."""
    from utils.datasets import letterbox
    from torchvision.ops import nms
    image = _image(image)
    h, w = image.shape
    resized, ratio, pad = letterbox(_rgb(image), new_shape=size, auto=False)
    x = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1))).to(device).float()[None] / 255
    model.eval()
    pred = model(x)[0][0]
    scores = pred[:, 4] * pred[:, 5]
    keep = scores >= confidence
    pred, scores = pred[keep], scores[keep]
    boxes = torch.cat((pred[:, :2] - pred[:, 2:4] / 2, pred[:, :2] + pred[:, 2:4] / 2), 1)
    idx = nms(boxes, scores, iou)[:max_detections]
    boxes, scores = boxes[idx], scores[idx]
    boxes -= boxes.new_tensor([pad[0], pad[1], pad[0], pad[1]])
    boxes /= boxes.new_tensor([ratio[0], ratio[1], ratio[0], ratio[1]])
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[valid].cpu().numpy(), scores[valid].cpu().numpy()


@torch.inference_mode()
def predict_boxes(model, image, device='cpu', confidence=.001, iou=.45, max_detections=100):
    """Tile native image and NMS overlapping proposals, for detector AP only."""
    from torchvision.ops import nms
    image = _image(image)
    h, w = image.shape
    padded = np.pad(image, ((0, max(0, 256-h)), (0, max(0, 256-w))))
    all_boxes, all_scores = [], []
    for y in _deep().positions(padded.shape[0], 256, 192):
        for x in _deep().positions(padded.shape[1], 256, 192):
            boxes, scores = detect(model, padded[y:y+256, x:x+256], device, confidence, iou, max_detections)
            boxes += np.array([x, y, x, y], np.float32)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            all_boxes.append(boxes[valid]); all_scores.append(scores[valid])
    boxes, scores = np.concatenate(all_boxes), np.concatenate(all_scores)
    if not len(boxes):
        return boxes, scores
    keep = nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou).numpy()
    return boxes[keep], scores[keep]


@torch.inference_mode()
def predict(stage, model, image, device='cpu', detector=None, confidence=.25):
    """Full-native 256/192 tiles. YOLO uses only detector-generated prompts."""
    image = _image(image)
    model.eval()
    if stage == 'deepmica':
        return _deep().tiled_predict(model, deep_input(image), size=256, stride=192, device=device)
    if stage != 'segmentor' or detector is None:
        raise ValueError('Segmentation inference requires a detector')
    model.eval(); detector.eval()
    h, w = image.shape
    padded = np.pad(image, ((0, max(0, 256-h)), (0, max(0, 256-w))))
    sums = np.zeros(padded.shape, np.float32); counts = np.zeros(padded.shape, np.float32)
    for y in _deep().positions(padded.shape[0], 256, 192):
        for x in _deep().positions(padded.shape[1], 256, 192):
            patch = padded[y:y+256, x:x+256]
            boxes, _ = detect(detector, patch, device=device, confidence=confidence)
            prob = np.zeros((256, 256), np.float32)
            if len(boxes):
                encoded = model.encode(_rgb(patch))
                for box in boxes:
                    out = model.decode(encoded, box[None]).sigmoid()[0, 0].cpu().numpy()
                    np.maximum(prob, out, out=prob)
            sums[y:y+256, x:x+256] += prob
            counts[y:y+256, x:x+256] += 1
    return (sums / counts)[:h, :w]

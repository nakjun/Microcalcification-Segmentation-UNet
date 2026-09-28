"""Project adapter for the pinned official DeepMiCa segmentation network."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'external/DeepMiCa/Step2_Segmentation/_03_Train/UNet.py'
SOURCE_HASH = '166055cbc2c0966a104bfc158783525820f8756941ea55329c2d32c6db05250a'
COMMIT = 'c2ce72cdf3265897827b8ab38fc9e1c7965ff812'


def build_model(channels=1):
    if channels not in (1, 2):
        raise ValueError('channels must be 1 or 2')
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_HASH:
        raise ValueError('Official model source hash mismatch')
    spec = importlib.util.spec_from_file_location('deepmica_pinned_unet', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.build_unet()
    if channels == 2:
        model.e1.conv.conv1 = nn.Conv2d(2, 64, 3, padding=1)
    return model


def read_array(path):
    path = Path(path)
    if path.suffix.lower() == '.npy':
        a = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as im:
            a = np.array(im)
    if a.ndim != 2 or not np.isfinite(a).all() or min(a.shape) < 1:
        raise ValueError(f'Expected a finite nonempty grayscale HxW array: {path}')
    return a.astype(np.float32)


def image_tensor(path, channels, scale):
    a = read_array(path)
    if not np.isfinite(scale) or scale <= 0 or a.min() < 0 or a.max() > scale:
        raise ValueError(f'Image outside declared [0, {scale}] range: {path}')
    raw = a / scale
    if channels == 2:
        if scale != 255. or not np.equal(a, np.floor(a)).all():
            raise ValueError('Paper 2-channel representation requires integer uint8-range input and scale=255')
        smooth = cv2.GaussianBlur(raw, (19, 19), 3.0, borderType=cv2.BORDER_REFLECT_101)
        return torch.from_numpy(np.stack((raw * 2 - 1, np.clip((raw - smooth) * 16., -1, 1))))
    return torch.from_numpy(raw)[None]


def read_manifest(path, require_training=True):
    path = Path(path).resolve()
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'image', 'mask', 'patient_id', 'split'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Manifest requires {sorted(required)}')
        rows = list(reader)
    patients, files, file_splits = {}, set(), {}
    for row in rows:
        if row['split'] not in ('train', 'val', 'test') or not row['patient_id'].strip():
            raise ValueError('Require train/val/test split and nonempty patient_id')
        previous = patients.setdefault(row['patient_id'], row['split'])
        if previous != row['split']:
            raise ValueError('Patient crosses dataset splits')
        for key in ('image', 'mask'):
            row[key] = (path.parent / row[key]).resolve()
            if not row[key].is_file():
                raise ValueError(f'Missing {key}: {row[key]}')
            old_split = file_splits.setdefault(row[key], row['split'])
            if old_split != row['split']:
                raise ValueError('Image/mask file crosses dataset splits')
        if row['image'] == row['mask'] or row['image'] in files:
            raise ValueError('Duplicate image or image used as its own mask')
        files.add(row['image'])
    if not rows or (require_training and not all(any(r['split'] == s for r in rows) for s in ('train', 'val'))):
        raise ValueError('Separate nonempty train and val partitions required')
    return rows


class Patches(torch.utils.data.Dataset):
    def __init__(self, rows, channels, scale, size):
        self.rows, self.channels, self.scale, self.size = rows, channels, scale, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        x = image_tensor(row['image'], self.channels, self.scale)
        mask = read_array(row['mask'])
        expected = self.size + (18 if self.channels == 2 else 0)
        if tuple(x.shape[-2:]) != (expected, expected) or mask.shape != (self.size, self.size):
            raise ValueError('Require square image/mask patches; 2-channel image needs a 9-pixel native-context halo on every edge')
        if self.channels == 2:
            x = x[:, 9:-9, 9:-9]
        values = set(np.unique(mask).tolist())
        if not (values <= {0., 1.} or values <= {0., 255.}):
            raise ValueError('Mask must be binary 0/1 or 0/255')
        return x, torch.from_numpy((mask > 0).astype(np.float32))[None]


def topk_loss(logits, target):
    """Official 3:1 hard-negative BCE, with explicit all-negative fallback."""
    losses = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    terms = []
    for loss, truth in zip(losses, target):
        positive, negative = loss[truth > 0], loss[truth == 0]
        # Original loss is NaN for a wholly negative batch. Retain every
        # negative pixel for empty patches, explicitly differing in this case.
        count = min(negative.numel(), 3 * positive.numel()) if positive.numel() else negative.numel()
        terms.extend((positive, negative.topk(count).values))
    return torch.cat(terms).mean()


def metadata(channels, scale, size):
    return dict(format_version=1, architecture='DeepMiCa-official-UNet', channels=channels,
                source_sha256=SOURCE_HASH, upstream_commit=COMMIT, scale=scale, patch_size=size,
                input_transform='raw/scale' if channels == 1 else 'uint8/255*2-1 + clip((uint8/255-Gaussian19x19-sigma3-reflect101)*16,-1,1); training-native-halo9',
                loss='topk-BCE-3-to-1; empty-target uses all negatives', selection='minimum validation loss')


def save_checkpoint(path, model, meta, epoch):
    torch.save(dict(metadata=meta, epoch=epoch, model_state_dict=model.state_dict()), path)


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    meta = checkpoint['metadata']
    expected = metadata(meta['channels'], meta['scale'], meta['patch_size'])
    if meta != expected or not np.isfinite(meta['scale']) or meta['scale'] <= 0 or (meta['channels'] == 2 and meta['scale'] != 255.) or meta['patch_size'] < 32 or meta['patch_size'] % 16:
        raise ValueError('Unsupported or inconsistent checkpoint metadata')
    model = build_model(meta['channels'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model.to(device).eval(), meta


def positions(length, size, stride):
    starts = list(range(0, max(length - size, 0) + 1, stride))
    if starts[-1] != max(length - size, 0):
        starts.append(max(length - size, 0))
    return starts


@torch.inference_mode()
def tiled_predict(model, x, size=256, stride=128, device='cpu'):
    if size < 32 or size % 16 or not 1 <= stride <= size:
        raise ValueError('Require patch size >=32 divisible by16 and 1<=stride<=size')
    model.eval()
    h, w = x.shape[-2:]
    if x.shape[0] == 2:
        def reflected(length, count):
            if length == 1:
                return torch.zeros(count, dtype=torch.long)
            idx = torch.arange(count) % (2 * length - 2)
            return torch.where(idx < length, idx, 2 * length - 2 - idx)
        padded = x[:, reflected(h, max(h, size))][:, :, reflected(w, max(w, size))]
    else:
        padded = F.pad(x, (0, max(0, size-w), 0, max(0, size-h)))
    sums, counts = torch.zeros(padded.shape[-2:]), torch.zeros(padded.shape[-2:])
    for y in positions(padded.shape[-2], size, stride):
        for z in positions(padded.shape[-1], size, stride):
            prob = model(padded[:, y:y+size, z:z+size][None].to(device)).sigmoid()[0, 0].cpu()
            if not torch.isfinite(prob).all():
                raise ValueError('Nonfinite model prediction')
            sums[y:y+size, z:z+size] += prob
            counts[y:y+size, z:z+size] += 1
    return (sums / counts)[:h, :w].numpy()


def train(args):
    if args.patch_size < 32 or args.patch_size % 16 or args.epochs < 1 or args.batch_size < 1 or not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.scale) or args.scale <= 0:
        raise ValueError('Invalid training settings')
    if args.channels == 2 and args.scale != 255.:
        raise ValueError('Paper 2-channel mode requires scale=255')
    rows = read_manifest(args.manifest)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    meta = metadata(args.channels, args.scale, args.patch_size)
    config = vars(args).copy()
    config['manifest_sha256'] = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    config['metadata'] = meta
    (out / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    loaders = {s: torch.utils.data.DataLoader(Patches([r for r in rows if r['split'] == s], args.channels, args.scale, args.patch_size), batch_size=args.batch_size, shuffle=(s == 'train')) for s in ('train', 'val')}
    model = build_model(args.channels).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.99)
    best, history = float('inf'), []
    for epoch in range(1, args.epochs + 1):
        record = {'epoch': epoch}
        for split, loader in loaders.items():
            model.train(split == 'train')
            total, count = 0., 0
            for x, target in loader:
                with torch.set_grad_enabled(split == 'train'):
                    loss = topk_loss(model(x.to(args.device)), target.to(args.device))
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite loss')
                    if split == 'train':
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                total += loss.item() * len(x)
                count += len(x)
            record[split + '_loss'] = total / count
        save_checkpoint(out / 'last.pt', model, meta, epoch)
        if record['val_loss'] < best:
            best = record['val_loss']
            save_checkpoint(out / 'best.pt', model, meta, epoch)
        history.append(record)
        (out / 'history.json').write_text(json.dumps(history, indent=2)+'\n')
        print(json.dumps(record), flush=True)
    (out / 'training_complete.json').write_text(json.dumps({'epochs': args.epochs, 'scope': 'train and validation only; test not evaluated'})+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--manifest', required=True)
    t.add_argument('--output', required=True)
    t.add_argument('--channels', type=int, choices=(1, 2), default=1)
    t.add_argument('--scale', type=float, default=255.)
    t.add_argument('--patch-size', type=int, default=256)
    t.add_argument('--batch-size', type=int, default=4)
    t.add_argument('--epochs', type=int, default=16)
    t.add_argument('--lr', type=float, default=.01)
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--device', default='cpu')
    i = sub.add_parser('infer')
    i.add_argument('--checkpoint', required=True)
    i.add_argument('--image', required=True)
    i.add_argument('--output', required=True)
    i.add_argument('--stride', type=int, default=128)
    i.add_argument('--threshold', type=float, default=.5)
    i.add_argument('--device', default='cpu')
    args = p.parse_args()
    if args.command == 'train':
        train(args)
    else:
        if not 0 <= args.threshold <= 1:
            raise ValueError('threshold must be in [0,1]')
        model, meta = load_checkpoint(args.checkpoint, args.device)
        x = image_tensor(args.image, meta['channels'], meta['scale'])
        prob = tiled_predict(model, x, meta['patch_size'], args.stride, args.device)
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=False)
        np.save(out / 'probability.npy', prob)
        Image.fromarray((prob >= args.threshold).astype(np.uint8)*255).save(out / 'mask.png')
        (out / 'inference.json').write_text(json.dumps({'image': str(Path(args.image).resolve()), 'checkpoint': str(Path(args.checkpoint).resolve()), 'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(), 'metadata': meta, 'threshold': args.threshold, 'stride': args.stride, 'shape': list(prob.shape), 'scope': 'native image inference without GT'}, indent=2)+'\n')


if __name__ == '__main__':
    main()

"""Resumable frozen Mammo-CLIP features for cleaned train/validation only."""
import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from calcification_gate_common import atomic_json, checksum, pixel_hash, validate_shard


def signature(config_path, config):
    paths = [Path(config_path), Path(config['manifest']), Path(config['encoder_weights']),
             Path(__file__), Path('scripts/calcification_gate_common.py'),
             Path('scripts/prepare_calcification_gate_data.py')]
    paths += sorted(Path('external/Mammo-CLIP/src/codebase/Detectors/retinanet').glob('*.py'))
    return {str(p): checksum(p) for p in paths}


class Images(Dataset):
    def __init__(self, rows, config):
        self.rows, self.config = rows, config

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        cv2.setNumThreads(1)
        row, c = self.rows[index], self.config
        im = cv2.imread(row['roi_cache_path'], cv2.IMREAD_UNCHANGED)
        if im is None or im.dtype != np.uint8 or im.shape != (c['input_height'], c['input_width']):
            raise ValueError(f"Invalid ROI cache: {row['image_id']}")
        if pixel_hash(im) != row['roi_sha256']:
            raise ValueError(f"ROI pixels changed since audit: {row['image_id']}")
        padded = np.zeros((c['padded_height'], c['padded_width']), np.uint8)
        padded[:c['input_height'], :c['input_width']] = im
        return torch.from_numpy(padded)[None]


def build_encoder(config, device):
    sys.path.insert(0, str(Path('external/Mammo-CLIP/src/codebase').resolve()))
    from Detectors.retinanet.efficient_net import EfficientNet
    expected = json.loads(Path('runs/classifier_preflight/report.json').read_text())['encoder']['encoder_sha256']
    assert checksum(config['encoder_weights']) == expected
    model = EfficientNet.from_name('efficientnet-b5', override_params={'image_size': None})
    state = torch.load(config['encoder_weights'], map_location='cpu', weights_only=True)
    loaded = model.load_state_dict(state, strict=False)
    assert set(loaded.missing_keys) <= {'_fc.weight', '_fc.bias'} and not loaded.unexpected_keys
    model._fc = nn.Identity()
    model.source_layer_indexes = [26, 37]
    return model.requires_grad_(False).eval().to(device)


@torch.inference_mode()
def pooled_features(model, images, config, device):
    x = (images.to(device, non_blocking=True).float().div(255).expand(-1, 3, -1, -1)
         - config['mean']) / config['std']
    with torch.autocast('cuda', dtype=torch.bfloat16):
        feat = model(x)[0]
    feat = feat.float()
    height, width = feat.shape[-2:]
    yy = (config['input_height'] - torch.arange(height, device=device) * 32).clamp(0, 32) / 32
    xx = (config['input_width'] - torch.arange(width, device=device) * 32).clamp(0, 32) / 32
    mask = yy[:, None] * xx[None, :]
    pooled = (feat * mask).sum((-2, -1)) / mask.sum()
    assert pooled.shape == (len(images), 2048) and torch.isfinite(pooled).all()
    return pooled.cpu().numpy()


def run(args):
    config = json.loads(Path(args.config).read_text())
    root = Path(config['root']) / ('smoke_features' if args.smoke else 'features')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ready = json.loads((Path(config['manifest']).parent / 'READY.json').read_text())
    assert ready['checks_passed'] and checksum(config['manifest']) == ready['manifest_sha256']
    all_rows = json.loads(Path(config['manifest']).read_text())
    rows = sorted([r for r in all_rows if r['split'] in ('train', 'val')],
                  key=lambda r: (r['split'] != 'train', r['image_id']))
    if args.smoke:
        rows = ([r for r in rows if r['split'] == 'train' and r['label']][:8]
                + [r for r in rows if r['split'] == 'train' and not r['label']][:8])
    assert len({r['image_id'] for r in rows}) == len(rows)
    sig = signature(args.config, config)
    if (root / 'signature.json').exists():
        assert json.loads((root / 'signature.json').read_text()) == sig, 'Feature protocol/source changed'
    else:
        atomic_json(root / 'signature.json', sig)
    torch.set_num_threads(config['cpu_threads'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; feature extraction has not started')
    torch.cuda.set_device(config['gpu'])
    torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'], config['gpu'])
    torch.backends.cudnn.benchmark = False
    device = f"cuda:{config['gpu']}"
    model = build_encoder(config, device)
    atomic_json(root / 'initialization.json', {'encoder_frozen': all(not p.requires_grad for p in model.parameters()),
                'encoder_eval': not model.training, 'gpu': torch.cuda.get_device_name(device), 'pid': os.getpid(),
                'rows': len(rows), 'scope': 'smoke only' if args.smoke else 'clean train/validation; no test'})
    start, reused, batches = time.monotonic(), 0, 0
    files = []
    shard_size = config['feature_shard_size']
    for offset in range(0, len(rows), shard_size):
        subset = rows[offset:offset + shard_size]
        path = root / f'shard_{offset // shard_size:04d}.npz'
        if path.exists() and path.with_suffix('.json').exists():
            validate_shard(path, subset, sig)
            reused += len(subset)
        else:
            loader = DataLoader(Images(subset, config), batch_size=config['feature_batch_size'],
                                num_workers=config['feature_workers'], pin_memory=True)
            values, processed = [], 0
            for images in loader:
                values.append(pooled_features(model, images, config, device))
                processed += len(images)
                batches += 1
                atomic_json(root / 'status.json', {'state': 'extracting', 'processed_images': offset + processed,
                            'committed_images': offset, 'total': len(rows), 'reused_images': reused,
                            'elapsed_seconds': time.monotonic() - start, 'pid': os.getpid(), 'updated_at_unix': time.time()})
            features = np.concatenate(values).astype(np.float32, copy=False)
            temporary = path.with_suffix('.npz.tmp')
            with temporary.open('wb') as f:
                np.savez(f, features=features, image_ids=np.asarray([r['image_id'] for r in subset]),
                         labels=np.asarray([r['label'] for r in subset], dtype=np.int64))
            temporary.replace(path)
            atomic_json(path.with_suffix('.json'), {'signature': sig, 'sha256': checksum(path)})
            validate_shard(path, subset, sig)
        files.append({'path': str(path), 'offset': offset, 'images': len(subset), 'sha256': checksum(path)})
        elapsed = time.monotonic() - start
        print(f'Features committed {offset + len(subset)}/{len(rows)}; reused={reused}; elapsed={elapsed:.1f}s', flush=True)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters()) and not model.training
    result = {'complete': True, 'smoke': args.smoke, 'rows': len(rows), 'image_ids': [r['image_id'] for r in rows],
              'shards': files, 'signature': sig, 'reused_images': reused, 'new_batches': batches,
              'elapsed_seconds': time.monotonic() - start,
              'gpu_peak_allocated_MiB': torch.cuda.max_memory_allocated(device) / 2**20,
              'encoder_frozen': True, 'encoder_optimizer_steps': 0}
    atomic_json(root / 'complete.json', result)
    atomic_json(root / 'status.json', {'state': 'complete', 'processed_images': len(rows), 'total': len(rows),
                'updated_at_unix': time.time(), 'pid': os.getpid()})
    print(json.dumps({k: v for k, v in result.items() if k not in ('shards', 'signature', 'image_ids')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/calcification_gate.json')
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())

"""Last-stage Mammo-CLIP fine-tuning with train-only CV and exact prefix caching.

Existing frozen-gate and RetinaNet source/configuration/artifacts are read-only.
The outer development validation is scored once after epoch selection/refitting.
"""
import argparse
import copy
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from calcification_gate_common import atomic_json, checksum, classification_metrics, validate_shard
from extract_calcification_features import Images, build_encoder, pooled_features
from train_calcification_gate import fit_head, model_arrays, score_arrays


def event(root, state, **kwargs):
    value = dict(state=state, pid=os.getpid(), updated_at_unix=time.time(), **kwargs)
    atomic_json(root / 'status.json', value)
    with (root / 'events.jsonl').open('a') as handle:
        handle.write(json.dumps(value, allow_nan=False) + '\n')
    print(json.dumps(value, allow_nan=False), flush=True)


def atomic_torch(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)
    atomic_json(path.with_suffix('.sha256.json'), {'sha256': checksum(path)})


def verified_load(path, device='cpu'):
    assert checksum(path) == json.loads(path.with_suffix('.sha256.json').read_text())['sha256']
    return torch.load(path, map_location=device, weights_only=False)


def disk_guard(config, additional=0):
    free = shutil.disk_usage(config['root']).free
    needed = int(config['disk_reserve_gib'] * 2**30) + additional
    if free < needed:
        raise RuntimeError(f'Insufficient disk: free={free}, required including reserve={needed}')


def protocol_signature(args, config):
    paths = [args.config, config['manifest'], config['encoder_weights'], __file__,
             'scripts/calcification_gate_common.py', 'scripts/train_calcification_gate.py',
             'scripts/extract_calcification_features.py', 'scripts/run_calcification_gate_finetune.py',
             'runs/classifier_preflight/report.json',
             str(Path(config['baseline_root']) / 'features/complete.json')]
    paths += [str(p) for p in sorted(Path('external/Mammo-CLIP/src/codebase/Detectors/retinanet').glob('*.py'))]
    return {'files': {str(p): checksum(p) for p in paths}, 'torch_version': str(torch.__version__),
            'numpy_version': np.__version__, 'smoke': args.stage == 'smoke'}


def read_rows(config):
    path = Path(config['manifest'])
    ready = json.loads((path.parent / 'READY.json').read_text())
    assert ready['checks_passed'] and checksum(path) == ready['manifest_sha256']
    all_rows = json.loads(path.read_text())
    rows = sorted([r for r in all_rows if r['split'] in ('train', 'val')],
                  key=lambda r: (r['split'] != 'train', r['image_id']))
    assert len(rows) == len({r['image_id'] for r in rows})
    for fold in range(config['inner_folds']):
        fit = [r for r in rows if r['split'] == 'train' and r['inner_fold'] != fold]
        held = [r for r in rows if r['split'] == 'train' and r['inner_fold'] == fold]
        for key in ['study_id', 'component_id', 'roi_sha256']:
            assert not {r[key] for r in fit} & {r[key] for r in held}
    for key in ['study_id', 'component_id', 'roi_sha256']:
        assert not {r[key] for r in rows if r['split'] == 'train'} & {
            r[key] for r in rows if r['split'] == 'val'}
    return rows


def baseline_train_features(config, rows):
    done = json.loads((Path(config['baseline_root']) / 'features/complete.json').read_text())
    assert done['complete'] and not done['smoke']
    for path, digest in done['signature'].items():
        assert checksum(path) == digest, f'Frozen baseline source changed: {path}'
    assert done['image_ids'] == [r['image_id'] for r in rows]
    chunks = []
    train_count = sum(r['split'] == 'train' for r in rows)
    for shard in done['shards']:
        if shard['offset'] >= train_count:
            break
        assert checksum(shard['path']) == shard['sha256']
        subset = rows[shard['offset']:shard['offset'] + shard['images']]
        values = validate_shard(shard['path'], subset, done['signature'])
        chunks.append(values[:max(0, train_count - shard['offset'])])
    values = np.concatenate(chunks).astype(np.float64)
    assert values.shape == (train_count, config['features'])
    return values


def normalized(images, config, device):
    return (images.to(device, non_blocking=True).float().div(255).expand(-1, 3, -1, -1)
            - config['mean']) / config['std']


@torch.no_grad()
def prefix_features(encoder, images, config, device):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        x = normalized(images, config, device)
        x = encoder._bn0(encoder._conv_stem(x))
        x = x * torch.sigmoid(x)  # Exact upstream relu_fn, including BF16 rounding.
        for block in encoder._blocks[:config['tail_start_block']]:
            x = block(x, drop_connect_rate=None)
    assert x.shape[1:] == (304, 48, 29) and x.dtype == torch.bfloat16
    assert torch.isfinite(x).all()
    return x


class TailGate(nn.Module):
    def __init__(self, encoder, arrays, config):
        super().__init__()
        assert len(encoder._blocks) == 39 and config['tail_start_block'] == 36
        self.blocks = nn.ModuleList(list(encoder._blocks[36:]))
        self.conv_head, self.bn_head = encoder._conv_head, encoder._bn1
        self.config = config
        self.register_buffer('feature_mean', torch.tensor(arrays['mean'], dtype=torch.float32))
        self.register_buffer('feature_scale', torch.tensor(arrays['scale'], dtype=torch.float32))
        self.head = nn.Linear(config['features'], 1)
        with torch.no_grad():
            self.head.weight.copy_(torch.as_tensor(arrays['coef'], dtype=torch.float32)[None])
            self.head.bias.copy_(torch.as_tensor(arrays['intercept'], dtype=torch.float32).reshape(1))
        self.requires_grad_(True)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Deliberately keep BN statistics and all stochastic layers in eval mode.
        super().train(False)
        return self

    def pooled(self, x):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, drop_connect_rate=None)
            x = self.bn_head(self.conv_head(x))
            x = x * torch.sigmoid(x)  # Exact upstream relu_fn.
        x = x.float()
        h, w = x.shape[-2:]
        yy = (self.config['input_height'] - torch.arange(h, device=x.device) * 32).clamp(0, 32) / 32
        xx = (self.config['input_width'] - torch.arange(w, device=x.device) * 32).clamp(0, 32) / 32
        mask = yy[:, None] * xx[None, :]
        return (x * mask).sum((-2, -1)) / mask.sum()

    def forward(self, x):
        z = (self.pooled(x) - self.feature_mean) / self.feature_scale
        return self.head(z).squeeze(1)


def initialize(config, x_fit, y_fit, device):
    with threadpool_limits(limits=config['cpu_threads']):
        scaler, head, meta = fit_head(x_fit, y_fit, config['C'], config)
    arrays = model_arrays(scaler, head)
    encoder = build_encoder(config, 'cpu')
    gate = TailGate(encoder, arrays, config).to(device)
    params = [(n, p) for n, p in gate.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam([
        {'params': [p for n, p in params if not n.startswith('head.')], 'lr': config['encoder_lr']},
        {'params': list(gate.head.parameters()), 'lr': config['head_lr']}], weight_decay=0)
    meta.update(trainable_parameters=sum(p.numel() for _, p in params),
                tail_parameter_names=[n for n, _ in params],
                scaler_scope='fit rows only; fixed thereafter', batchnorm_frozen=True,
                drop_connect_enabled=False, feature_mean_sha256=hashlib.sha256(arrays['mean'].tobytes()).hexdigest())
    return gate, optimizer, arrays, meta


def validate_prefix_shard(path, subset, sig):
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['signature'] == sig and meta['sha256'] == checksum(path)
    assert meta['image_ids'] == [r['image_id'] for r in subset]
    assert meta['labels'] == [r['label'] for r in subset]
    arr = np.load(path, mmap_mode='r', allow_pickle=False)
    assert arr.shape == (len(subset), 304, 48, 29) and arr.dtype == np.uint16
    # Exponent bits all ones denote BF16 NaN or infinity.
    for start in range(0, len(arr), 8):
        assert not np.any((arr[start:start + 8] & 0x7F80) == 0x7F80)
    return meta


def cache_prefix(config, rows, root, sig, device):
    out = root / 'prefix_cache'
    out.mkdir(exist_ok=True)
    train_rows = [r for r in rows if r['split'] == 'train']
    size = config['cache_shard_size']
    missing_rows = sum(len(train_rows[o:o + size]) for o in range(0, len(train_rows), size)
                       if not (out / f'shard_{o // size:04d}.npy').exists())
    disk_guard(config, missing_rows * 304 * 48 * 29 * 2 + int(config['checkpoint_budget_gib'] * 2**30))
    model = None
    shards, started, reused = [], time.monotonic(), 0
    for offset in range(0, len(train_rows), size):
        subset = train_rows[offset:offset + size]
        path = out / f'shard_{offset // size:04d}.npy'
        if path.exists() and path.with_suffix('.json').exists():
            validate_prefix_shard(path, subset, sig)
            reused += len(subset)
        else:
            if model is None:
                model = build_encoder(config, device)
            loader = DataLoader(Images(subset, config), batch_size=config['cache_batch_size'],
                                num_workers=config['workers'], pin_memory=True)
            values = []
            for images in loader:
                values.append(prefix_features(model, images, config, device).cpu().view(torch.uint16).numpy())
                event(root, 'cache_extracting', images=offset + sum(len(v) for v in values),
                      total=len(train_rows), elapsed_seconds=time.monotonic() - started)
            temporary = path.with_suffix('.npy.tmp')
            with temporary.open('wb') as handle:
                np.save(handle, np.concatenate(values), allow_pickle=False)
            temporary.replace(path)
            atomic_json(path.with_suffix('.json'), {'signature': sig, 'sha256': checksum(path),
                        'image_ids': [r['image_id'] for r in subset], 'labels': [r['label'] for r in subset]})
            validate_prefix_shard(path, subset, sig)
        shards.append({'path': str(path), 'offset': offset, 'images': len(subset), 'sha256': checksum(path)})
        event(root, 'cache_shard_committed', images=offset + len(subset), total=len(train_rows),
              reused_images=reused, elapsed_seconds=time.monotonic() - started)
    done = {'complete': True, 'signature': sig, 'rows': len(train_rows), 'shards': shards,
            'image_ids': [r['image_id'] for r in train_rows], 'scope': 'train only; no validation or test cache',
            'elapsed_seconds': time.monotonic() - started, 'reused_images': reused,
            'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20}
    atomic_json(out / 'complete.json', done)
    event(root, 'cache_complete', rows=len(train_rows), elapsed_seconds=done['elapsed_seconds'])


class PrefixDataset(Dataset):
    def __init__(self, rows, indices, shards):
        self.rows, self.indices = rows, np.asarray(indices)
        self.mapping = []
        for shard in shards:
            assert shard['offset'] == len(self.mapping)
            self.mapping.extend((shard['path'], i) for i in range(shard['images']))
        assert len(self.mapping) == len(rows)
        self.arrays = {}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        path, offset = self.mapping[index]
        if path not in self.arrays:
            self.arrays[path] = np.load(path, mmap_mode='r', allow_pickle=False)
        x = torch.from_numpy(np.array(self.arrays[path][offset], copy=True)).view(torch.bfloat16)
        return x, torch.tensor(self.rows[index]['label'], dtype=torch.float32)


def make_loader(config, rows, indices, shards, epoch=None, job_seed=42):
    generator = torch.Generator().manual_seed(job_seed + (epoch or 0) * 1009)
    return DataLoader(PrefixDataset(rows, indices, shards), batch_size=config['batch_size'],
                      shuffle=epoch is not None, generator=generator, num_workers=config['workers'],
                      pin_memory=True, persistent_workers=False)


def regularized_loss(logits, labels, gate, ratio, n_fit, config):
    bce = F.binary_cross_entropy_with_logits(logits, labels,
                                           pos_weight=logits.new_tensor(ratio), reduction='sum')
    l2 = gate.head.weight.square().sum() / (2 * config['C'] * n_fit)
    return bce, l2


def train_epoch(gate, optimizer, loader, ratio, n_fit, config, root, job, epoch, device):
    gate.train()
    optimizer.zero_grad(set_to_none=True)
    effective = config['batch_size'] * config['accumulate_batches']
    running, seen, steps = 0., 0, 0
    started = time.monotonic()
    frozen_before = {n: t.detach().clone() for n, t in gate.state_dict().items()
                     if n not in {name for name, p in gate.named_parameters() if p.requires_grad}}
    # Each accumulation window has its own exact denominator, including the last partial window.
    for batch_index, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = gate(x)
        bce, l2 = regularized_loss(logits, y, gate, ratio, n_fit, config)
        window_start = (batch_index // config['accumulate_batches']) * effective
        window_images = min(effective, n_fit - window_start)
        loss = bce / window_images + l2 * (len(y) / window_images)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss in {job}, epoch {epoch}, batch {batch_index}')
        loss.backward()
        running += float(bce.detach())
        seen += len(y)
        if (batch_index + 1) % config['accumulate_batches'] == 0 or batch_index + 1 == len(loader):
            norm = nn.utils.clip_grad_norm_([p for p in gate.parameters() if p.requires_grad],
                                           config['gradient_clip_norm'], error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
        if (batch_index + 1) % 50 == 0 or batch_index + 1 == len(loader):
            event(root, 'training', job=job, epoch=epoch, images=seen, total=n_fit,
                  optimizer_steps=steps, mean_weighted_bce=running / seen,
                  elapsed_seconds=time.monotonic() - started)
    for name, expected in frozen_before.items():
        assert torch.equal(gate.state_dict()[name], expected), f'Frozen state changed: {name}'
    assert seen == n_fit
    return {'epoch': epoch, 'fit_images': n_fit, 'optimizer_steps': steps,
            'mean_weighted_bce': running / seen, 'head_l2': float(l2.detach()),
            'gradient_norm_last': float(norm), 'elapsed_seconds': time.monotonic() - started}


@torch.no_grad()
def predict_cache(gate, loader, device):
    gate.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(gate(x.to(device, non_blocking=True)).double().cpu().numpy())
        labels.append(y.numpy())
    logits, labels = np.concatenate(logits), np.concatenate(labels).astype(np.int64)
    assert np.isfinite(logits).all()
    return labels, logits


def train_job(config, root, job, rows, x, fit_indices, held_indices, shards, epochs, sig, device):
    out = root / job
    out.mkdir(exist_ok=True)
    job_seed = config['seed']
    torch.manual_seed(job_seed)
    np.random.seed(job_seed)
    labels = np.asarray([r['label'] for r in rows])
    gate, optimizer, arrays, meta = initialize(config, x[fit_indices], labels[fit_indices], device)
    atomic_json(out / 'initialization.json', meta | {
        'fit_image_ids': [rows[i]['image_id'] for i in fit_indices],
        'held_image_ids': [rows[i]['image_id'] for i in held_indices], 'seed': job_seed})
    history, start_epoch = [], 1
    checkpoint = out / 'last.pt'
    if checkpoint.exists():
        state = verified_load(checkpoint, device)
        assert state['signature'] == sig and state['job'] == job and state['epoch'] <= epochs
        gate.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        history = state['history']
        start_epoch = state['epoch'] + 1
    if len(held_indices):
        held_loader = make_loader(config, rows, held_indices, shards)
        if not history:
            y_held, logits = predict_cache(gate, held_loader, device)
            expected = score_arrays(arrays, x[held_indices])
            max_diff = float(np.max(np.abs(logits - expected)))
            # Same BF16 convolutions may have batch-size-dependent rounding.
            assert max_diff < .02, f'Initial cached logits differ from frozen baseline: {max_diff}'
            history = [{'epoch': 0, 'held_metrics': classification_metrics(y_held, logits),
                        'initial_logit_max_abs_difference_from_baseline': max_diff}]
            atomic_json(out / 'epoch_00.json', history[0])
    for epoch in range(start_epoch, epochs + 1):
        loader = make_loader(config, rows, fit_indices, shards, epoch=epoch, job_seed=job_seed)
        meta_epoch = train_epoch(gate, optimizer, loader, meta['positive_weight'], len(fit_indices),
                                 config, root, job, epoch, device)
        if len(held_indices):
            y_held, logits = predict_cache(gate, held_loader, device)
            meta_epoch['held_metrics'] = classification_metrics(y_held, logits)
            with (out / f'held_epoch_{epoch:02d}.npz').open('wb') as handle:
                np.savez(handle, image_ids=np.asarray([rows[i]['image_id'] for i in held_indices]),
                         labels=y_held, logits=logits)
        history.append(meta_epoch)
        disk_guard(config, 512 * 2**20)
        atomic_torch(checkpoint, {'model': gate.state_dict(), 'optimizer': optimizer.state_dict(),
                                 'epoch': epoch, 'job': job, 'signature': sig, 'history': history})
        atomic_json(out / f'epoch_{epoch:02d}.json', meta_epoch)
        event(root, 'epoch_complete', job=job, **meta_epoch)
    atomic_json(out / 'complete.json', {'complete': True, 'epochs': epochs, 'signature': sig,
                'checkpoint_sha256': checksum(checkpoint), 'history': history,
                'scope': 'train-only fold' if len(held_indices) else 'all cleaned train refit'})
    return gate, history


@torch.no_grad()
def validation(gate, config, root, rows, selected_epoch, sig, device):
    out = root / 'final_seed42'
    val_rows = [r for r in rows if r['split'] == 'val']
    encoder = build_encoder(config, device)
    loader = DataLoader(Images(val_rows, config), batch_size=config['batch_size'],
                        num_workers=config['workers'], pin_memory=True)
    logits = []
    for images in loader:
        prefix = prefix_features(encoder, images, config, device)
        logits.append(gate(prefix).double().cpu().numpy())
        event(root, 'development_validation', images=sum(len(x) for x in logits), total=len(val_rows))
    logits = np.concatenate(logits)
    labels = np.asarray([r['label'] for r in val_rows])
    metrics = classification_metrics(labels, logits)
    metrics.update(scope=config['threshold_scope'], selected_epoch=selected_epoch,
                   images=len(val_rows), positive_images=int(labels.sum()), seed=config['seed'],
                   test_evaluated=False, cascade_evaluated=False)
    atomic_json(out / 'validation_metrics.json', metrics)
    primary = metrics['operating_points'][0]
    atomic_json(out / 'thresholds.json', {'primary': primary, 'secondary': metrics['operating_points'][1:],
                'comparison': 'logit >= threshold_logit; ties included', 'scope': config['threshold_scope'],
                'score_interpretation': 'class-weighted score, not calibrated disease probability'})
    with (out / 'validation_predictions.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['image_id', 'study_id', 'label', 'logit', 'score', 'gate_pass'])
        writer.writeheader()
        for row, logit in zip(val_rows, logits):
            writer.writerow({key: row[key] for key in ['image_id', 'study_id', 'label']} | {
                'logit': repr(float(logit)), 'score': repr(float(expit(logit))),
                'gate_pass': int(logit >= primary['threshold_logit'])})
    with (out / 'validation_scores.npz').open('wb') as handle:
        np.savez(handle, image_ids=np.asarray([r['image_id'] for r in val_rows]), labels=labels, logits=logits)
    model_path = out / 'classifier.pt'
    atomic_torch(model_path, {'model': gate.state_dict(), 'signature': sig, 'config': config,
                             'selected_epoch': selected_epoch, 'encoder_source': config['encoder_weights']})
    reloaded = verified_load(model_path, device)
    gate.load_state_dict(reloaded['model'])
    # Export must reproduce the last evaluated batch exactly.
    assert np.array_equal(gate(prefix).double().cpu().numpy(), logits[-len(prefix):])
    done = {'complete': True, 'smoke': False, 'finished_at_unix': time.time(), 'signature': sig,
            'classifier_sha256': checksum(model_path), 'selected_epoch': selected_epoch,
            'manifest_sha256': checksum(config['manifest']), 'validation': metrics,
            'test_evaluated': False, 'cascade_evaluated': False}
    atomic_json(out / 'evaluation_complete.json', done)
    event(root, 'fit_and_validation_complete', selected_epoch=selected_epoch, validation=metrics)


def fit(config, root, rows, sig, device):
    done = json.loads((root / 'prefix_cache/complete.json').read_text())
    assert done['complete'] and done['signature'] == sig
    train_rows = [r for r in rows if r['split'] == 'train']
    assert done['image_ids'] == [r['image_id'] for r in train_rows]
    for shard in done['shards']:
        subset = train_rows[shard['offset']:shard['offset'] + shard['images']]
        validate_prefix_shard(Path(shard['path']), subset, sig)
    x = baseline_train_features(config, rows)
    folds = np.asarray([r['inner_fold'] for r in train_rows])
    histories = []
    for fold in range(config['inner_folds']):
        event(root, 'fold_starting', fold=fold)
        gate, history = train_job(config, root, f'fold_{fold}', train_rows, x,
                                  np.flatnonzero(folds != fold), np.flatnonzero(folds == fold),
                                  done['shards'], config['epochs'], sig, device)
        histories.append(history)
        del gate
        gc.collect()
        torch.cuda.empty_cache()
    candidates = []
    for epoch in range(config['epochs'] + 1):
        records = [next(r for r in history if r['epoch'] == epoch) for history in histories]
        candidates.append({'epoch': epoch, 'eligible_for_finetuned_selection': epoch > 0,
            'mean_negative_rejection_at_full_sensitivity': float(np.mean([
                r['held_metrics']['operating_points'][0]['negative_rejection'] for r in records])),
            'mean_average_precision': float(np.mean([r['held_metrics']['average_precision'] for r in records])),
            'mean_roc_auc': float(np.mean([r['held_metrics']['roc_auc'] for r in records]))})
    selected = max(candidates[1:], key=lambda r: (r['mean_negative_rejection_at_full_sensitivity'],
                                                 r['mean_average_precision'], -r['epoch']))
    atomic_json(root / 'cv_results.json', {'candidates': candidates, 'fold_histories': histories})
    atomic_json(root / 'selection.json', {'selected_epoch': selected['epoch'], 'criterion': config['selection'],
                'scope': 'clean train grouped five-fold only', 'validation_used_for_selection': False,
                'epoch_zero_reference': candidates[0], 'selected': selected})
    gate, _ = train_job(config, root, 'final_seed42', train_rows, x, np.arange(len(train_rows)),
                        np.asarray([], dtype=np.int64), done['shards'], selected['epoch'], sig, device)
    validation(gate, config, root, rows, selected['epoch'], sig, device)


def smoke(config, root, rows, device):
    train_rows = [r for r in rows if r['split'] == 'train']
    x = baseline_train_features(config, rows)
    # Only training images; this path is never used for model selection or performance reporting.
    positives = [i for i, r in enumerate(train_rows) if r['label'] == 1][:8]
    negatives = [i for i, r in enumerate(train_rows) if r['label'] == 0][:8]
    indices = [index for pair in zip(positives, negatives) for index in pair]
    subset = [train_rows[i] for i in indices]
    labels = np.asarray([r['label'] for r in subset])
    gate, optimizer, arrays, meta = initialize(config, x[indices], labels, device)
    encoder = build_encoder(config, device)
    before_encoder = {n: p.detach().clone() for n, p in encoder.named_parameters()}
    images = next(iter(DataLoader(Images(subset, config), batch_size=config['batch_size'], num_workers=0)))
    started = time.monotonic()
    prefix = prefix_features(encoder, images, config, device)
    reconstructed = prefix.cpu().view(torch.uint16).numpy().copy()
    restored = torch.from_numpy(reconstructed).view(torch.bfloat16).to(device)
    assert torch.equal(prefix, restored)
    original_pool = pooled_features(encoder, images, config, device)
    with torch.no_grad():
        split_pool = gate.pooled(restored).cpu().numpy()
        logits = gate(restored).cpu().numpy()
    np.testing.assert_allclose(split_pool, original_pool, rtol=0, atol=1e-6)
    np.testing.assert_allclose(logits, score_arrays(arrays, original_pool), rtol=1e-4, atol=1e-4)
    before_tail = {n: p.detach().clone() for n, p in gate.named_parameters()}
    before_buffers = {n: t.detach().clone() for n, t in gate.named_buffers()}
    gate.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        y = torch.as_tensor(labels[:len(images)], dtype=torch.float32, device=device)
        bce, l2 = regularized_loss(gate(restored), y, gate, meta['positive_weight'], len(subset), config)
        loss = bce / len(images) + l2
        loss.backward()
        assert all(p.grad is None for n, p in gate.named_parameters() if not p.requires_grad)
        tail_grads = [p.grad for n, p in gate.named_parameters() if p.requires_grad and not n.startswith('head.')]
        assert all(g is not None and torch.isfinite(g).all() for g in tail_grads)
        assert sum(float(g.abs().sum()) for g in tail_grads) > 0
        optimizer.step()
    changed = [n for n, p in gate.named_parameters() if not torch.equal(p, before_tail[n])]
    assert any(n.startswith('blocks.') for n in changed) and 'conv_head.weight' in changed
    assert 'head.weight' in changed
    assert all(torch.equal(t, before_buffers[n]) for n, t in gate.named_buffers())
    assert all(torch.equal(p, before_encoder[n]) and p.grad is None for n, p in encoder.named_parameters())
    checkpoint = root / 'smoke_checkpoint.pt'
    atomic_torch(checkpoint, {'model': gate.state_dict(), 'optimizer': optimizer.state_dict()})
    with torch.no_grad():
        expected = gate(restored).detach().clone()
    loaded = verified_load(checkpoint, device)
    gate.load_state_dict(loaded['model'])
    optimizer.load_state_dict(loaded['optimizer'])
    with torch.no_grad():
        assert torch.equal(expected, gate(restored))
    result = {'complete': True, 'smoke': True, 'scope': 'training-path checks only, no performance estimate',
              'batch_size': len(images), 'optimizer_steps': 3, 'tail_parameters_changed': changed,
              'prefix_cache_bitwise_roundtrip': True, 'split_encoder_pooling_matches_original': True,
              'frozen_prefix_unchanged': True, 'all_bn_parameters_and_buffers_unchanged': True,
              'checkpoint_reload_reproduces_logits': True, 'elapsed_seconds': time.monotonic() - started,
              'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
              'initialization': meta}
    atomic_json(root / 'complete.json', result)
    event(root, 'smoke_complete', **{k: v for k, v in result.items() if k not in ('initialization', 'tail_parameters_changed')})


def run(args):
    config = json.loads(Path(args.config).read_text())
    root = Path(config['root']) / 'smoke' if args.stage == 'smoke' else Path(config['root'])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sig = protocol_signature(args, config)
    sig_path = root / 'signature.json'
    if sig_path.exists() and args.stage != 'smoke':
        assert json.loads(sig_path.read_text()) == sig, 'Configuration/source/provenance changed; use a new experiment version'
    else:
        atomic_json(sig_path, sig)
    rows = read_rows(config)
    torch.set_num_threads(config['cpu_threads'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no training started')
    torch.cuda.set_device(config['gpu'])
    torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'], config['gpu'])
    torch.backends.cudnn.benchmark = False
    device = f"cuda:{config['gpu']}"
    try:
        if args.stage == 'smoke':
            smoke(config, root, rows, device)
        elif args.stage == 'cache':
            cache_prefix(config, rows, root, sig, device)
        else:
            fit(config, root, rows, sig, device)
    except BaseException as exc:
        event(root, 'failed', stage=args.stage, error=repr(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/calcification_gate_finetune.json')
    parser.add_argument('--stage', choices=['smoke', 'cache', 'fit'], required=True)
    run(parser.parse_args())

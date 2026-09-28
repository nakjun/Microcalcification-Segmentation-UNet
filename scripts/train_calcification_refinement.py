"""Fixed-budget native pixel U-Net adaptation, internal selection, then locked evaluation."""
import argparse
import fcntl
import hashlib
import json
import os
import time
import traceback
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from calcification_gate_common import atomic_json, checksum
from calcification_gate_sam_common import atomic_npz, disk_guard
from calcification_refinement_common import (PatchUNet, TrainSampler, baseline_masks,
    tiled_probability, training_loss)
from evaluate_calcification_refinement import evaluate
from run_calcification_gate_sam import Progress


def atomic_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def reserved_rows(config):
    root = Path(config['root'])
    path = Path(config['frozen_pipeline_root']) / 'inputs/csaw_manifest.json'
    original = json.loads(path.read_text())
    assert {r['split'] for r in original} == {'CSAW-S reserved'}, 'Unexpected original evaluation cohort'
    audit = {r['image_id']: r for r in json.loads((root / 'data/audit_rows.json').read_text()) if r['split'] == 'reserved'}
    assert len(original) == len(audit) == 156
    assert {r['image_id'] for r in original} == set(audit)
    normalized = []
    for row in original:
        checked = audit[row['image_id']]
        for key in ['study_id', 'source_path', 'source_sha256', 'height', 'width']:
            assert row[key] == checked[key], f'Reserved audit mismatch: {row["image_id"]}/{key}'
        assert checked['gt_read'] is False
        normalized.append(dict(row, original_split=row['split'], split='reserved'))
    return normalized


def source_signature(config_path, config, rows):
    root, baseline = Path(config['root']), Path(config['development_baseline_root'])
    complete = json.loads((baseline / 'complete.json').read_text())
    assert complete['complete'] and not complete['GT_masks_read']
    for path, expected in complete['output_files'].items():
        assert checksum(path) == expected, f'Development baseline changed: {path}'
    data = json.loads((root / 'data/complete.json').read_text())
    assert data['complete'] and not data['reserved_GT_read']
    assert checksum(root / 'data/manifest.json') == data['manifest_sha256']
    files = [config_path, __file__, 'scripts/calcification_refinement_common.py',
             'scripts/evaluate_calcification_refinement.py', 'scripts/prepare_calcification_refinement.py',
             'scripts/test_calcification_refinement.py', 'scripts/test_calcification_refinement_evaluation.py',
             'scripts/calcification_gate_common.py',
             'scripts/calcification_gate_sam_common.py', 'scripts/run_calcification_gate_sam.py',
             'scripts/prepare_calcification_refinement_baselines.py', root / 'data/complete.json',
             root / 'data/manifest.json', root / 'data/split.json', root / 'data/decisions.json',
             root / 'data/audit_rows.json', baseline / 'complete.json', baseline / 'signature.json',
             config['source_manifest'], config['original_group_split'],
             Path(config['frozen_pipeline_root']) / 'complete.json',
             Path(config['frozen_pipeline_root']) / 'inputs/csaw_manifest.json']
    hashes = {str(p): checksum(p) for p in files}
    for row in rows:
        for pathkey, hashkey in [('native_cache_path', 'native_cache_sha256'), ('target_path', 'target_sha256'),
                                 ('source_path', 'source_sha256'), ('gt_source_path', 'gt_source_sha256')]:
            assert checksum(row[pathkey]) == row[hashkey], f'Development input changed: {row[pathkey]}'
            hashes[row[pathkey]] = row[hashkey]
    hashes.update(complete['output_files'])
    # Read only the prior run's artifact hash map, never its scores to choose this model.
    # Reserved baseline case/mask checksums and image hashes require no GT mask access.
    previous = Path(config['frozen_pipeline_root'])
    expected_outputs = json.loads((previous / 'complete.json').read_text())['output_files']
    for row in reserved_rows(config):
        assert checksum(row['source_path']) == row['source_sha256']
        hashes[row['source_path']] = row['source_sha256']
        for arm in config['arms']:
            for relative in [f'gated/cases/{row["image_id"]}.json', f'gated/masks/{row["image_id"]}.npz']:
                path = str(previous / 'csaw' / arm / relative)
                assert checksum(path) == expected_outputs[path], f'Reserved frozen baseline changed: {path}'
                hashes[path] = expected_outputs[path]
    return {'files': hashes, 'torch': str(torch.__version__), 'numpy': np.__version__, 'opencv': cv2.__version__,
            'reserved_GT_read_for_signature': False}


def sampling_centers(rows, config, signature, progress):
    root = Path(config['root']) / 'data'
    path, marker = root / 'sampling_centers.npz', root / 'sampling_centers.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        assert saved['signature_digest'] == signature and checksum(path) == saved['sha256']
        with np.load(path) as value:
            return {k: value[k] for k in value.files}
    rng = np.random.default_rng(config['seed'])
    centers = {}
    for index, row in enumerate(rows):
        assert row['role'] == 'train'
        ident = row['image_id']
        image = np.load(row['native_cache_path'], mmap_mode='r')
        y, x = np.where(image[::16, ::16] > 40)
        breast = np.stack((y * 16, x * 16), 1).astype(np.int32)
        union = np.zeros(image.shape, bool)
        for arm in config['arms']:
            sam, _, _ = baseline_masks(config['development_baseline_root'], arm, ident)
            union |= sam
        flat = np.flatnonzero(union)
        if len(flat) > 4096:
            flat = rng.choice(flat, 4096, replace=False)
        hard = np.stack(np.unravel_index(flat, image.shape), 1).astype(np.int32)
        if len(breast) > 4096:
            breast = breast[rng.choice(len(breast), 4096, replace=False)]
        assert len(breast)
        centers[ident + '_hard'], centers[ident + '_breast'] = hard, breast
        if (index + 1) % 20 == 0:
            progress('training_sampling_centers', images=index + 1, total=len(rows))
    atomic_npz(path, **centers)
    atomic_json(marker, {'complete': True, 'signature_digest': signature, 'sha256': checksum(path),
                         'training_images': len(rows), 'held_targets_used': False,
                         'hard_negative_whole_patch_GT_rejection_at_sampling': True})
    return centers


def train_epoch(model, optimizer, sampler, config, epoch, device, progress):
    model.train()
    rng = np.random.default_rng(config['seed'] + epoch * 1009)
    count, batch = config['training_patches_per_epoch'], config['training_batch_size']
    modes, sums = Counter(), Counter()
    start, steps = time.monotonic(), 0
    for offset in range(0, count, batch):
        examples = [sampler.sample(rng) for _ in range(min(batch, count - offset))]
        x = torch.from_numpy(np.stack([v[0] for v in examples])).to(device)
        y = torch.from_numpy(np.stack([v[1] for v in examples])).to(device)
        modes.update(v[2] for v in examples)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
        loss, stats = training_loss(logits, y, config)
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
        assert torch.isfinite(norm)
        optimizer.step()
        steps += 1
        sums.update({k: v for k, v in stats.items()})
        sums['loss'] += float(loss.detach())
        if steps % 16 == 0 or offset + batch >= count:
            progress('training', epoch=epoch, patches=min(offset + batch, count), total=count,
                     optimizer_steps=steps, mean_loss=sums['loss'] / steps,
                     elapsed_seconds=time.monotonic() - start)
    return {'epoch': epoch, 'optimizer_steps': steps, 'patches': count, 'sampling_counts': dict(modes),
            'mean_loss': sums['loss'] / steps, 'mean_BCE': sums['BCE'] / steps,
            'mean_positive_soft_Dice_loss': sums['positive_soft_Dice_loss'] / steps,
            'positive_patches': int(sums['positive_patches']), 'elapsed_seconds': time.monotonic() - start}


def smoke(rows, config, device, signature, progress):
    root = Path(config['root']) / 'smoke'
    root.mkdir(parents=True, exist_ok=True)
    train = [r for r in rows if r['role'] == 'train']
    centers = sampling_centers(train, config, signature, progress)
    sampler = TrainSampler(train, config, centers)
    model = PatchUNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    rng = np.random.default_rng(142)
    examples = []
    while len(examples) < 4:
        item = sampler.sample(rng)
        want_positive = len(examples) % 2 == 0
        if bool(item[1].any()) == want_positive:
            examples.append(item)
    x, y = [torch.from_numpy(np.stack([item[i] for item in examples])).to(device) for i in [0, 1]]
    t0 = time.monotonic()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            prediction = model(x)
        loss, stats = training_loss(prediction, y, config)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
        assert torch.isfinite(norm)
        optimizer.step()
    changed = [k for k, value in model.state_dict().items() if not torch.equal(value, before[k])]
    assert any(k.startswith('enc1') for k in changed) and any(k.startswith('out') for k in changed)
    path = root / 'checkpoint.pt'
    atomic_torch(path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict()})
    restored = PatchUNet(config).to(device)
    restored.load_state_dict(torch.load(path, weights_only=False, map_location=device)['model'], strict=True)
    model.eval(), restored.eval()
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        assert torch.equal(model(x), restored(x))
    update_seconds = time.monotonic() - t0
    held = next(r for r in rows if r['role'] == 'validation')
    image = np.load(held['native_cache_path'], mmap_mode='r')
    progress('smoke_full_native_inference', image_id=held['image_id'], shape=list(image.shape))
    t0 = time.monotonic()
    probability, tiles = tiled_probability(model, image, config, device)
    elapsed = time.monotonic() - t0
    assert probability.shape == image.shape and np.isfinite(probability).all()
    result = {'complete': True, 'signature_digest': signature, 'pid': os.getpid(),
              'optimizer_steps': 3, 'both_positive_negative_patches': True,
              'encoder_and_output_updated': True, 'checkpoint_reload_exact': True,
              'optimizer_update_seconds': update_seconds, 'parameters': sum(p.numel() for p in model.parameters()),
              'held_image_only_inference': {'image_id': held['image_id'], 'shape': list(image.shape),
                  'tiles': tiles, 'seconds': elapsed, 'tiles_per_second': tiles / elapsed, 'GT_read': False},
              'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated() / 2**20}
    atomic_json(root / 'complete.json', result)
    progress('smoke_complete', result=result)


def run(rows, config, device, signature, progress):
    root = Path(config['root'])
    smoke_complete = json.loads((root / 'smoke/complete.json').read_text())
    assert smoke_complete['complete'] and smoke_complete['signature_digest'] == signature
    if (root / 'complete.json').exists():
        completed = json.loads((root / 'complete.json').read_text())
        assert completed['signature_digest'] == signature
        for path, sha in completed['output_files'].items():
            assert checksum(path) == sha
        progress('already_complete', complete_path=str(root / 'complete.json'))
        return
    train, held = [[r for r in rows if r['role'] == role] for role in ['train', 'validation']]
    centers = sampling_centers(train, config, signature, progress)
    sampler = TrainSampler(train, config, centers)
    model = PatchUNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    best, epoch_outputs = None, []
    for epoch in range(1, config['epochs'] + 1):
        disk_guard(config, 256 * 2**20)
        checkpoint_path = root / 'checkpoints' / f'epoch_{epoch:02d}.pt'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            assert checkpoint['signature_digest'] == signature and checkpoint['epoch'] == epoch
            model.load_state_dict(checkpoint['model'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            training = checkpoint['training']
        else:
            training = train_epoch(model, optimizer, sampler, config, epoch, device, progress)
            atomic_torch(checkpoint_path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'signature_digest': signature, 'config': config, 'epoch': epoch, 'training': training})
        checkpoint_sha = checksum(checkpoint_path)
        evaluation_signature = digest({'source': signature, 'checkpoint': checkpoint_sha})
        result = evaluate(model, held, config, device, root / 'epochs' / f'{epoch:02d}',
                          evaluation_signature, epoch, progress)
        candidate = result['selected_candidate'] | {'checkpoint_path': str(checkpoint_path),
                                                   'checkpoint_sha256': checkpoint_sha}
        if best is None or tuple(candidate['rank']) > tuple(best['rank']):
            best = candidate
        epoch_record = {'complete': True, 'signature_digest': signature, 'training': training,
                        'held_candidate': candidate, 'best_to_date': best,
                        'held_complete_path': str(root / 'epochs' / f'{epoch:02d}' / 'complete.json')}
        record_path = root / 'epochs' / f'epoch_{epoch:02d}.json'
        atomic_json(record_path, epoch_record)
        epoch_outputs += [checkpoint_path, record_path, root / 'epochs' / f'{epoch:02d}' / 'complete.json']
        progress('epoch_complete', epoch=epoch, best=best, training=training)
    selected_path = root / 'selection.json'
    selection = best | {'locked_before_reserved_GT': True, 'signature_digest': signature,
        'training_groups': sorted({r['study_id'] for r in train}),
        'validation_groups': sorted({r['study_id'] for r in held}),
        'development_groups': sorted({r['study_id'] for r in rows}),
        'selection_source': 'development held groups only, mean two-arm positive macro Dice',
        'refit': False, 'epochs_considered': config['epochs']}
    if selected_path.exists():
        assert json.loads(selected_path.read_text()) == selection
    else:
        atomic_json(selected_path, selection)
    assert checksum(selection['checkpoint_path']) == selection['checkpoint_sha256']
    checkpoint = torch.load(selection['checkpoint_path'], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    locked_signature = digest({'source': signature, 'selection': checksum(selected_path)})
    final_held = evaluate(model, held, config, device, root / 'final_held', locked_signature,
                          selection['epoch'], progress, selection=selection)
    # The first reserved mask read can only occur beyond the persisted selection lock above.
    reserved = reserved_rows(config)
    external = evaluate(model, reserved, config, device, root / 'additional_reserved', locked_signature,
                        selection['epoch'], progress, selection=selection, reserved=True)
    files = {str(p): checksum(p) for p in epoch_outputs + [selected_path, root / 'final_held/complete.json',
                root / 'additional_reserved/complete.json', root / 'data/sampling_centers.json', root / 'data/sampling_centers.npz']}
    files.update(final_held['output_files'])
    files.update(external['output_files'])
    complete = {'complete': True, 'signature_digest': signature, 'finished_at_unix': time.time(),
                'seed': config['seed'], 'trained_epochs': config['epochs'], 'selection': selection,
                'final_held_metrics': final_held['metrics'], 'additional_reserved_metrics': external['metrics'],
                'reserved_used_for_selection': False, 'refit': False,
                'interpretation': config['interpretation'], 'output_files': files}
    atomic_json(root / 'complete.json', complete)
    progress('complete', complete_path=str(root / 'complete.json'), selected=selection)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/calcification_refinement.json')
    parser.add_argument('--mode', choices=['smoke', 'run'], required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config['root'])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    progress = Progress(root, config['heartbeat_seconds'])
    try:
        cv2.setNumThreads(1)
        torch.set_num_threads(config['cpu_threads'])
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        assert torch.cuda.is_available(), 'CUDA requires the authorized host runner'
        torch.cuda.set_device(config['gpu'])
        device = torch.device('cuda', config['gpu'])
        torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'], device)
        free, _ = torch.cuda.mem_get_info(device)
        assert free / 2**20 >= config['gpu_min_free_mib'], 'Wait for sufficient free GPU memory before launch'
        disk_guard(config, 512 * 2**20)
        rows = json.loads((root / 'data/manifest.json').read_text())
        progress('verifying_source_signature', mode=args.mode)
        signature = source_signature(args.config, config, rows)
        signature_path = root / 'signature.json'
        if signature_path.exists():
            assert json.loads(signature_path.read_text()) == signature, 'Source or inputs changed; do not silently resume'
        else:
            atomic_json(signature_path, signature)
        signature_hash = digest(signature)
        if (root / 'failure.json').exists():
            history = root / 'failure_history'
            history.mkdir(exist_ok=True)
            (root / 'failure.json').replace(history / f'failure_{time.time_ns()}.json')
        progress('initialized', mode=args.mode, signature_digest=signature_hash,
                 gpu=config['gpu'], gpu_free_mib=free / 2**20)
        if args.mode == 'smoke':
            smoke(rows, config, device, signature_hash, progress)
        else:
            run(rows, config, device, signature_hash, progress)
    except BaseException as error:
        atomic_json(root / 'failure.json', {'time': time.time(), 'pid': os.getpid(),
            'mode': args.mode, 'error': repr(error), 'traceback': traceback.format_exc()})
        progress('failed', error=repr(error))
        raise
    finally:
        progress.close()


if __name__ == '__main__':
    main()

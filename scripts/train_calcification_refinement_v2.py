"""Paired train-only grouped OOF experiment; immutable v1 inputs, no reserved access."""
import argparse
import copy
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
from calcification_gate_sam_common import atomic_npz, disk_guard, summarize_segmentation
from calcification_refinement_common import PatchUNet, baseline_masks, tiled_probability
from calcification_refinement_coverage_common import candidates, measure_probability, reconstructed_mask
from calcification_refinement_v2_common import (RecallAwareSampler, choose_recall_constrained,
    component_pixel_groups, mine_negative_centers, training_loss)
from run_calcification_gate_sam import Progress
from train_calcification_refinement import atomic_torch, digest, sampling_centers


PLAN = 'configs/calcification_refinement_v2_plan.json'
ROOT = Path('runs/calcification_refinement_v2_seed42')


def read(path):
    return json.loads(Path(path).read_text())


def checked_marker(path, signature):
    if not path.exists():
        return None
    record = read(path)
    assert record['complete'] and record['signature'] == signature, str(path)
    for output, sha in record.get('output_files', {}).items():
        assert checksum(output) == sha, output
    return record


def validate_folds(rows, folds):
    train = {r['image_id']: r for r in rows if r['role'] == 'train'}
    held = {r['image_id'] for r in rows if r['role'] != 'train'}
    assert len(train) == 124 and len(held) == 32 and len(folds) == 3
    seen = []
    for fold in folds:
        a, b = set(fold['train_ids']), set(fold['validation_ids'])
        assert not a & b and a | b == set(train) and not (a | b) & held
        assert not {train[i]['study_id'] for i in a} & {train[i]['study_id'] for i in b}
        assert not {train[i]['component_id'] for i in a} & {train[i]['component_id'] for i in b}
        assert not {train[i]['pixel_sha256'] for i in a} & {train[i]['pixel_sha256'] for i in b}
        seen.extend(b)
    assert len(seen) == len(set(seen)) == 124
    return train


def prepare():
    plan = read(PLAN)
    assert checksum(plan['base_config']) == plan['base_config_sha256']
    assert checksum(plan['folds']) == plan['folds_sha256']
    config = read(plan['base_config'])
    config.update(root=str(ROOT), epochs=plan['fixed_training']['epochs'])
    folds = read(plan['folds'])
    assert checksum(folds['source_manifest']) == folds['source_manifest_sha256']
    rows = read(folds['source_manifest'])
    validate_folds(rows, folds['folds'])
    ROOT.mkdir(parents=True, exist_ok=True)
    paths = [PLAN, plan['base_config'], plan['folds'], folds['source_manifest'], __file__,
        'scripts/calcification_refinement_v2_common.py', 'scripts/test_calcification_refinement_v2.py',
        'scripts/test_calcification_refinement_v2_runner.py', 'scripts/calcification_refinement_common.py',
        'scripts/calcification_refinement_coverage_common.py', 'scripts/calcification_gate_common.py',
        'scripts/calcification_gate_sam_common.py', 'scripts/run_calcification_gate_sam.py',
        'scripts/train_calcification_refinement.py']
    hashes = {str(p): checksum(p) for p in paths}
    # Only development156 files. In particular do not read reserved masks or metrics.
    for row in rows:
        for key in ['native_cache', 'target', 'source', 'gt_source']:
            path = row[key + '_path']
            assert checksum(path) == row[key + '_sha256'], path
            hashes[path] = row[key + '_sha256']
    baseline = Path(config['development_baseline_root'])
    complete = read(baseline / 'complete.json')
    assert complete['complete'] and not complete['GT_masks_read']
    hashes[str(baseline / 'complete.json')] = checksum(baseline / 'complete.json')
    for path, sha in complete['output_files'].items():
        assert checksum(path) == sha, path
        hashes[path] = sha
    signature_record = {'files': hashes, 'config': config, 'seed': 42,
        'torch': str(torch.__version__), 'numpy': np.__version__, 'opencv': cv2.__version__,
        'scope': 'train124 grouped OOF; held32 only after selection; no reserved reads'}
    signature = digest(signature_record)
    path = ROOT / 'signature.json'
    if path.exists():
        assert read(path) == signature_record, 'Immutable experiment inputs or source changed'
    else:
        atomic_json(path, signature_record)
        atomic_json(ROOT / 'config.json', config)
        atomic_json(ROOT / 'plan.json', plan)
        atomic_json(ROOT / 'folds.json', folds)
    return plan, config, rows, folds['folds'], signature


def fresh_model(config, device):
    torch.manual_seed(config['seed'])
    model = PatchUNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    return model, optimizer


def state_digest(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def epoch_train(model, optimizer, sampler, config, epoch, device, progress, context):
    model.train()
    rng = np.random.default_rng(config['seed'] + epoch * 1009)
    total, batch = config['training_patches_per_epoch'], config['training_batch_size']
    modes, sums = Counter(), Counter()
    start, steps = time.monotonic(), 0
    for offset in range(0, total, batch):
        examples = [sampler.sample(rng) for _ in range(min(batch, total - offset))]
        x, y = [torch.from_numpy(np.stack([v[k] for v in examples])).to(device) for k in [0, 1]]
        modes.update(v[2] for v in examples)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = model(x)
        loss, stats = training_loss(logits, y, config)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm']))
        optimizer.step()
        steps += 1
        sums.update(stats)
        sums['loss'] += float(loss.detach())
        if steps % 16 == 0 or offset + batch >= total:
            progress('training', **context, epoch=epoch, optimizer_steps=steps,
                     patches=min(offset + batch, total), total=total,
                     mean_loss=sums['loss'] / steps, elapsed_seconds=time.monotonic() - start)
    return {'epoch': epoch, 'optimizer_steps': steps, 'patches': total, 'sampling_counts': dict(modes),
        'mean_loss': sums['loss'] / steps, 'mean_BCE': sums['BCE'] / steps,
        'mean_positive_overlap_loss': sums['positive_overlap_loss'] / steps,
        'positive_patches': int(sums['positive_patches']), 'elapsed_seconds': time.monotonic() - start}


def component_stats(hit, groups):
    result = {name: {'total': 0, 'any_overlap': 0, 'half_overlap': 0}
              for name in ['1-4', '5-16', '17-64', '>64']}
    for indices in groups:
        size = len(indices)
        name = '1-4' if size <= 4 else '5-16' if size <= 16 else '17-64' if size <= 64 else '>64'
        overlap = int(hit[indices].sum())
        result[name]['total'] += 1
        result[name]['any_overlap'] += int(overlap > 0)
        result[name]['half_overlap'] += int(overlap * 2 >= size)
    return result


def aggregate(records, policy):
    result = summarize_segmentation(records, policy)
    values = [r[policy] for r in records]
    result.update(pixel_recall=result['TP'] / (result['TP'] + result['FN']) if result['TP'] + result['FN'] else None,
        pixel_precision=result['TP'] / (result['TP'] + result['FP']) if result['TP'] + result['FP'] else None,
        negative_FP_pixels=sum(v['FP'] for v in values if v['gt_pixels'] == 0),
        negative_images_with_FP=sum(v['FP'] > 0 for v in values if v['gt_pixels'] == 0),
        positive_images_with_TP=sum(v['TP'] > 0 for v in values if v['gt_pixels'] > 0),
        TP_inside_boxes=sum(v['TP_inside_boxes'] for v in values),
        TP_outside_boxes=sum(v['TP_outside_boxes'] for v in values))
    result['components'] = {size: {key: sum(v['components'][size][key] for v in values)
        for key in ['total', 'any_overlap', 'half_overlap']} for size in ['1-4', '5-16', '17-64', '>64']}
    return result


def evaluate(model, rows, config, inference, device, out, signature, progress, context, selected=None):
    marker = out / 'complete.json'
    if (saved := checked_marker(marker, signature)) is not None:
        return saved
    out.mkdir(parents=True, exist_ok=True)
    policies = candidates(inference)
    if selected is not None:
        policies = [p for p in policies if p['id'] == selected]
        assert len(policies) == 1
    names = [p['id'] for p in policies] + ['SAM_baseline', 'box_fill_control']
    all_cases, outputs = [], {}
    start = time.monotonic()
    for index, row in enumerate(rows):
        assert row['role'] == 'validation'
        path = out / 'cases' / (row['image_id'] + '.json')
        if path.exists():
            case = read(path)
            assert case['signature'] == signature and case['image_id'] == row['image_id']
            for p, sha in case.get('output_files', {}).items():
                assert checksum(p) == sha
        else:
            image = np.load(row['native_cache_path'], mmap_mode='r')
            probability, info = tiled_probability(model, image, config, device)
            with np.load(row['target_path']) as target:
                gy, gx = target['y'], target['x']
            groups = component_pixel_groups(gy, gx, image.shape) if len(gy) else []
            case = {'signature': signature, 'image_id': row['image_id'], 'study_id': row['study_id'],
                'source_sha256': row['source_sha256'], 'GT_sha256': row['target_sha256'],
                'inference': info, 'arms': {}, 'output_files': {}}
            for arm in config['arms']:
                sam, fill, baseline = baseline_masks(config['development_baseline_root'], arm, row['image_id'])
                passed = bool(baseline['gate_pass'])
                measured = measure_probability(probability, fill, sam, gy, gx, passed, inference)
                case['arms'][arm] = {key: measured[key] for key in names}
                inside = fill[gy, gx]
                scores = probability[gy, gx]
                for policy in policies:
                    hit = (scores >= np.float32(policy['inside_threshold'])) & inside
                    if policy['outside_threshold'] is not None:
                        hit |= (scores >= np.float32(policy['outside_threshold'])) & ~inside
                    hit &= passed
                    case['arms'][arm][policy['id']]['components'] = component_stats(hit, groups)
                for key, mask in [('SAM_baseline', sam), ('box_fill_control', fill)]:
                    case['arms'][arm][key]['components'] = component_stats(mask[gy, gx], groups)
                if selected is not None:
                    mask = reconstructed_mask(probability, fill, passed, policies[0])
                    save = out / 'masks' / arm / (row['image_id'] + '.npz')
                    atomic_npz(save, mask=np.packbits(mask), shape=np.array(mask.shape))
                    case['output_files'][str(save)] = checksum(save)
            atomic_json(path, case)
        outputs[str(path)] = checksum(path)
        outputs.update(case['output_files'])
        all_cases.append(case)
        if (index + 1) % 8 == 0 or index + 1 == len(rows):
            progress('evaluation', **context, images=index + 1, total=len(rows), elapsed_seconds=time.monotonic() - start)
    result = {'complete': True, 'signature': signature, 'context': context,
        'image_ids': [r['image_id'] for r in rows], 'elapsed_seconds': time.monotonic() - start,
        'metrics': {arm: {name: aggregate([r['arms'][arm] for r in all_cases], name) for name in names}
                    for arm in config['arms']}, 'output_files': outputs}
    atomic_json(marker, result)
    return result


def refresh_mining(model, train, centers, config, plan, device, out, signature, checkpoint_sha, progress, context):
    marker = out / 'complete.json'
    mining_signature = digest({'source': signature, 'checkpoint': checkpoint_sha,
        'train_ids': [r['image_id'] for r in train], 'settings': plan['model_FP_mining']})
    cached = checked_marker(marker, mining_signature)
    path = out / 'centers.npz'
    if cached is None:
        out.mkdir(parents=True, exist_ok=True)
        mined, provenance = {}, []
        for index, row in enumerate(train):
            assert row['role'] == 'train'
            image = np.load(row['native_cache_path'], mmap_mode='r')
            probability, _ = tiled_probability(model, image, config, device)
            with np.load(row['target_path']) as target:
                gy, gx = target['y'], target['x']
            settings = plan['model_FP_mining']
            points, scores = mine_negative_centers(row, probability, gy, gx,
                patch_size=config['patch_size'], threshold=settings['threshold'],
                cell_size=settings['spatial_cell_size'], maximum=settings['maximum_centers_per_image'])
            mined[row['image_id'] + '_hard'] = points
            mined[row['image_id'] + '_scores'] = scores
            provenance.append({'image_id': row['image_id'], 'image_sha256': row['native_cache_sha256'],
                'target_sha256': row['target_sha256'], 'centers': len(points)})
            if (index + 1) % 8 == 0 or index + 1 == len(train):
                progress('FP_mining', **context, images=index + 1, total=len(train))
        atomic_npz(path, **mined)
        atomic_json(marker, {'complete': True, 'signature': mining_signature, 'model_sha256': checkpoint_sha,
            'training_only': True, 'inputs': provenance, 'settings': plan['model_FP_mining'],
            'output_files': {str(path): checksum(path)}})
    with np.load(path) as saved:
        return {key: saved[key] if key.endswith('_hard') else value for key, value in centers.items()}


def train_job(train, validation, variant, job, config, plan, device, signature, progress,
              epochs=None, selected=None):
    job.mkdir(parents=True, exist_ok=True)
    c = dict(config, **variant, root=str(job))
    model, optimizer = fresh_model(c, device)
    init_sha = state_digest(model)
    meta = {'signature': signature, 'initial_model_sha256': init_sha, 'variant': variant,
            'train_ids': [r['image_id'] for r in train], 'validation_ids': [r['image_id'] for r in validation]}
    if (job / 'job.json').exists():
        assert read(job / 'job.json') == meta
    else:
        atomic_json(job / 'job.json', meta)
    # Shared seed and row order give identical fixed center pools for paired variants.
    centers = sampling_centers(train, c, signature, progress)
    epochs = epochs or c['epochs']
    for epoch in range(1, epochs + 1):
        context = {'job': str(job), 'variant': variant['id'], 'epoch': epoch}
        disk_guard(c, 32 * 2**20)
        checkpoint = job / 'checkpoints' / f'epoch_{epoch:02d}.pt'
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location=device, weights_only=False)
            assert saved['signature'] == signature and saved['epoch'] == epoch and saved['job'] == meta
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            stats = saved['training']
        else:
            sampler = RecallAwareSampler(train, c, centers)
            stats = epoch_train(model, optimizer, sampler, c, epoch, device, progress,
                                {'job': str(job), 'variant': variant['id']})
            del sampler
            atomic_torch(checkpoint, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'epoch': epoch, 'signature': signature, 'job': meta, 'config': c, 'training': stats})
        sha = checksum(checkpoint)
        epoch_signature = digest({'source': signature, 'job': meta, 'checkpoint': sha})
        if validation and (selected is None or epoch == epochs):
            evaluate(model, validation, c, plan['candidate_inference'], device,
                job / 'epochs' / f'{epoch:02d}', epoch_signature, progress, context, selected=selected)
        atomic_json(job / 'epochs' / f'epoch_{epoch:02d}.json',
            {'complete': True, 'signature': signature, 'epoch': epoch, 'checkpoint': str(checkpoint),
             'checkpoint_sha256': sha, 'training': stats})
        if variant['negative_strategy'] == 'current_model_FP' and epoch in plan['model_FP_mining']['refresh_after_epochs'] and epoch < epochs:
            centers = refresh_mining(model, train, centers, c, plan, device,
                job / 'mining' / f'after_{epoch:02d}', signature, sha, progress, context)
    outputs = {str(p): checksum(p) for p in sorted(job.rglob('*')) if p.is_file() and p.name != 'complete.json'}
    outputs.update({str(p): checksum(p) for p in sorted(job.rglob('complete.json')) if p != job / 'complete.json'})
    result = {'complete': True, 'signature': signature, 'job': meta, 'epochs': epochs,
              'output_files': outputs, 'completed_at_unix': time.time()}
    atomic_json(job / 'complete.json', result)
    return result


def pool_oof(rows, folds, plan, config, signature, progress):
    all_candidates, references = [], None
    names = [p['id'] for p in candidates(plan['candidate_inference'])]
    expected = {r['image_id'] for r in rows if r['role'] == 'train'}
    for variant in plan['ablation_variants']:
        for epoch in range(1, config['epochs'] + 1):
            cases = []
            for fold in folds:
                directory = ROOT / 'jobs' / variant['id'] / f'fold_{fold["fold"]}' / 'epochs' / f'{epoch:02d}'
                cases.extend(read(directory / 'cases' / (i + '.json')) for i in fold['validation_ids'])
            assert {r['image_id'] for r in cases} == expected and len(cases) == 124
            ref = {arm: aggregate([r['arms'][arm] for r in cases], 'SAM_baseline') for arm in config['arms']}
            if references is None:
                references = ref
            else:
                assert ref == references
            for name in names:
                value = {'candidate_id': f'{variant["id"]}/epoch_{epoch:02d}/{name}',
                    'variant': variant['id'], 'epoch': epoch, 'policy': name,
                    'metrics': {arm: aggregate([r['arms'][arm] for r in cases], name) for arm in config['arms']}}
                failures = {}
                for arm in config['arms']:
                    m, r = value['metrics'][arm], references[arm]
                    failures[arm] = [key for key, ok in {
                        'pixel_TP': m['TP'] >= r['TP'],
                        'positive_images_with_TP': m['positive_images_with_TP'] >= r['positive_images_with_TP'],
                        'negative_FP_pixels': m['negative_FP_pixels'] <= r['negative_FP_pixels'],
                        'negative_images_with_FP': m['negative_images_with_FP'] <= r['negative_images_with_FP']}.items() if not ok]
                value['constraint_failures'] = failures
                all_candidates.append(value)
            progress('OOF_aggregation', variant=variant['id'], epoch=epoch, images=124)
    winner = choose_recall_constrained(all_candidates, references)
    atomic_json(ROOT / 'oof_candidates.json', {'complete': True, 'signature': signature,
        'references': references, 'candidates': all_candidates, 'images': 124, 'seed': 42})
    selection = {'complete': True, 'signature': signature, 'selected': winner,
        'feasible_candidates': sum(not any(v['constraint_failures'].values()) for v in all_candidates),
        'total_candidates': len(all_candidates), 'selection_scope': 'pooled train124 OOF only',
        'locked_before_held32_confirmation': True, 'reserved_read': False,
        'candidate_file_sha256': checksum(ROOT / 'oof_candidates.json')}
    path = ROOT / 'selection.json'
    if path.exists():
        assert read(path) == selection
    else:
        atomic_json(path, selection)
    return selection


def smoke(rows, folds, plan, config, device, signature, progress):
    directory = ROOT / 'smoke'
    directory.mkdir(exist_ok=True)
    fold = folds[0]
    lookup = {r['image_id']: r for r in rows}
    train = [dict(lookup[i], role='train') for i in fold['train_ids']]
    validation = [dict(lookup[fold['validation_ids'][0]], role='validation')]
    c = dict(config, root=str(directory), training_patches_per_epoch=32)
    centers = sampling_centers(train, c, signature, progress)
    before_sha, runs = None, []
    for variant in [plan['ablation_variants'][0], plan['ablation_variants'][-1]]:
        v = dict(c, **variant)
        model, optimizer = fresh_model(v, device)
        initial = state_digest(model)
        if before_sha is None:
            before_sha = initial
        assert initial == before_sha
        stats = epoch_train(model, optimizer, RecallAwareSampler(train, v, centers), v, 1, device,
                            progress, {'mode': 'smoke', 'variant': variant['id']})
        assert state_digest(model) != initial
        checkpoint = directory / (variant['id'] + '.pt')
        atomic_torch(checkpoint, {'model': model.state_dict(), 'optimizer': optimizer.state_dict()})
        restored, _ = fresh_model(v, device)
        restored.load_state_dict(torch.load(checkpoint, weights_only=False, map_location=device)['model'])
        assert state_digest(restored) == state_digest(model)
        runs.append({'variant': variant['id'], 'training': stats, 'reload_exact': True})
    result = evaluate(model, validation, c, plan['candidate_inference'], device,
        directory / 'evaluation', digest({'source': signature, 'smoke_model': state_digest(model)}),
        progress, {'mode': 'smoke'}, selected=None)
    mining = refresh_mining(model, train[:1], {k: val for k, val in centers.items() if k.startswith(train[0]['image_id'] + '_')},
        c, plan, device, directory / 'mining', signature, checksum(checkpoint), progress, {'mode': 'smoke'})
    atomic_json(directory / 'complete.json', {'complete': True, 'signature': signature,
        'training_runs': runs, 'initial_model_sha256': before_sha, 'full_image_evaluation_seconds': result['elapsed_seconds'],
        'mining_training_images': 1, 'peak_allocated_MiB': torch.cuda.max_memory_allocated(device) / 2**20,
        'output_files': {str(p): checksum(p) for p in directory.rglob('*') if p.is_file() and p != directory / 'complete.json'}})


def run(rows, folds, plan, config, device, signature, progress):
    assert checked_marker(ROOT / 'smoke/complete.json', signature)
    if checked_marker(ROOT / 'complete.json', signature):
        return
    lookup = {r['image_id']: r for r in rows}
    for variant in plan['ablation_variants']:
        for fold in folds:
            job = ROOT / 'jobs' / variant['id'] / f'fold_{fold["fold"]}'
            if checked_marker(job / 'complete.json', signature):
                continue
            train = [dict(lookup[i], role='train') for i in fold['train_ids']]
            validation = [dict(lookup[i], role='validation') for i in fold['validation_ids']]
            train_job(train, validation, variant, job, config, plan, device, signature, progress)
    selection = pool_oof(rows, folds, plan, config, signature, progress)
    if selection['selected'] is not None:
        selected = selection['selected']
        variant = next(v for v in plan['ablation_variants'] if v['id'] == selected['variant'])
        train_job([r for r in rows if r['role'] == 'train'],
            [dict(r, role='validation') for r in rows if r['role'] != 'train'], variant,
            ROOT / 'final_fit', config, plan, device, signature, progress,
            epochs=selected['epoch'], selected=selected['policy'])
    outputs = {str(p): checksum(p) for p in ROOT.rglob('complete.json') if p != ROOT / 'complete.json'}
    outputs.update({str(ROOT / name): checksum(ROOT / name) for name in ['selection.json', 'oof_candidates.json', 'signature.json']})
    atomic_json(ROOT / 'complete.json', {'complete': True, 'signature': signature, 'seed': 42,
        'selection': selection, 'original_held_confirmation_performed': selection['selected'] is not None,
        'reserved_read': False, 'output_files': outputs, 'completed_at_unix': time.time()})
    progress('complete', selected=selection['selected']['candidate_id'] if selection['selected'] else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'run'], required=True)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    progress = Progress(ROOT, 30)
    try:
        plan, config, rows, folds, signature = prepare()
        torch.set_num_threads(config['cpu_threads'])
        cv2.setNumThreads(config['cpu_threads'])
        np.random.seed(config['seed'])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        assert torch.cuda.is_available()
        device = torch.device(f'cuda:{config["gpu"]}')
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'], device)
        assert torch.cuda.mem_get_info(device)[0] / 2**20 >= config['gpu_min_free_mib']
        disk_guard(config)
        if (ROOT / 'failure.json').exists():
            (ROOT / 'failure.json').rename(ROOT / f'failure_{int(time.time())}.json')
        progress('prepared', mode=args.mode, signature=signature, GPU=config['gpu'])
        (smoke if args.mode == 'smoke' else run)(rows, folds, plan, config, device, signature, progress)
    except Exception:
        atomic_json(ROOT / 'failure.json', {'pid': os.getpid(), 'time': time.time(), 'traceback': traceback.format_exc()})
        raise
    finally:
        progress.close()


if __name__ == '__main__':
    main()

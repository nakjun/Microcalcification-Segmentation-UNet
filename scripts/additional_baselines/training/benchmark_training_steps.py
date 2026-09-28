"""Fixed training-step microbenchmark; disposable updates, never retraining/results.

Run from repository root. Each backbone/fold starts every loss arm from the same
frozen LH epoch16 state and uses the same real train-only batch. Mining, sampler,
input transfer, checkpoint IO and validation costs are outside timing.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'scripts/comparative_study'))
from calcification_refinement_v5_common import V5Sampler, training_loss, background_eligible
from model import StudyModel
NAMES = ('proposed', 'unext', 'unet2d', 'utnet')
VARIANTS = ('C', 'L', 'H', 'LH')
DEFAULT_OUT = ROOT/'reports/additional_baselines/training_steps_20260927'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def run_root(name):
    return ROOT/('runs/improvement_unext_deterministic/unext_v5_losses_deterministic_seed42' if name == 'unext' else f'runs/comparative_study/{name}_seed42')


def factory(name, config):
    if name != 'unext':
        return StudyModel(config)
    spec = importlib.util.spec_from_file_location('step_benchmark_unext', ROOT/'scripts/improvement_unext_deterministic/model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UNeXtAuthor2ch(config)


def fixed_batch(fold):
    """Common U-Net LH sampler/config controls the input stream for all models."""
    baseline = run_root('unet2d')
    protocol = read(baseline/'protocol.json')
    cfg = dict(protocol['config'], **next(v for v in protocol['plan']['variants'] if v['id'] == 'LH'))
    group = next(f for f in protocol['folds'] if f['fold'] == fold)
    fold_record = read(ROOT/'runs/calcification_refinement_v5_seed42/folds.json')
    manifest = Path(fold_record['source_manifest'])
    assert sha(manifest) == fold_record['source_manifest_sha256']
    lookup = {r['image_id']: r for r in read(manifest)}
    rows = [lookup[i] for i in group['train_ids']]
    assert not set(group['train_ids']) & set(group['validation_ids'])
    sources = {str(manifest): sha(manifest)}
    for row in rows:
        assert row['role'] == 'train'
        for key in ('native_cache', 'target'):
            p = row[key+'_path']
            assert sha(p) == row[key+'_sha256']
            sources[str(p)] = row[key+'_sha256']
    pool = protocol['initial_pools'][str(fold)]
    assert pool['training_ids'] == group['train_ids'] and sha(pool['path']) == pool['sha256']
    sources[pool['path']] = pool['sha256']
    with np.load(pool['path']) as z:
        centers = {k: z[k] for k in z.files}
    sampler = V5Sampler(rows, cfg, centers)
    rng = np.random.default_rng(42+6*1009)
    examples = [sampler.sample(rng) for _ in range(16)]
    arrays = [np.stack([e[k] for e in examples]) for k in (0, 1, 3, 4)]
    h = hashlib.sha256()
    for a in arrays:
        h.update(a.tobytes())
    assert arrays[2].sum() > 0, 'Fixed batch must activate component term'
    assert arrays[1].sum() > 0
    return arrays, dict(input_target_geometry_sha256=h.hexdigest(), train_ids=group['train_ids'],
        validation_ids=group['validation_ids'], sampling_seed=42+6*1009,
        modes=[e[2] for e in examples], source_files=sources,
        positive_pixels=int(arrays[1].sum()), component_active_patches=int((arrays[2].sum((1,2,3))>0).sum()),
        shape=list(arrays[0].shape), sampling_pool='fixed initial pool; current-model mining not timed or reproduced')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=NAMES, required=True)
    p.add_argument('--fold', type=int, choices=(0,1,2), required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--output', type=Path, default=DEFAULT_OUT)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    os.chdir(ROOT)
    out = a.output/('smoke' if a.smoke else 'measurements')
    out.mkdir(parents=True, exist_ok=True)
    dest = out/f'{a.model}_fold{a.fold}.json'
    assert not dest.exists(), 'Preserve completed measurement'
    torch.set_num_threads(4)
    torch.cuda.set_device(a.gpu)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(f'cuda:{a.gpu}')
    cp = run_root(a.model)/f'jobs/LH/fold_{a.fold}/checkpoint.pt'
    before = sha(cp)
    done = read(cp.parent/'complete.json')
    key = str(cp.relative_to(ROOT))
    assert done['complete'] and not done['smoke'] and done['epochs'] == 16
    assert done['output_files'].get(key, done['output_files'].get(str(cp))) == before
    state = torch.load(cp, map_location='cpu', weights_only=False)
    assert state['epoch'] == 16
    arrays, batch_meta = fixed_batch(a.fold)
    assert state['job']['train_ids'] == batch_meta['train_ids']
    tensors = [torch.from_numpy(x).to(device) for x in arrays]
    x, y, w, valid = tensors
    batch_meta['eligible_background_pixels'] = int(background_eligible(y, valid).sum())
    assert batch_meta['eligible_background_pixels'] > 0
    protocol = read(run_root(a.model)/'protocol.json')
    warmup, iterations = (1,1) if a.smoke else (3,10)
    results = {}
    for variant in VARIANTS:
        torch.manual_seed(42)
        cfg = dict(state['config'], **next(v for v in protocol['plan']['variants'] if v['id'] == variant))
        model = factory(a.model, cfg)
        model.load_state_dict(state['model'], strict=True)
        model = model.to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        def step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
            loss, stats = training_loss(logits, y, w, valid, cfg, 6)
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            assert torch.isfinite(norm)
            optimizer.step()
            return float(loss.detach()), stats
        for _ in range(warmup):
            step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        times, losses = [], []
        for _ in range(iterations):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            loss, stats = step()
            torch.cuda.synchronize(device)
            times.append(time.perf_counter()-start)
            losses.append(loss)
        peak = torch.cuda.max_memory_allocated(device)
        assert all(torch.isfinite(q).all() for q in model.parameters())
        results[variant] = dict(seconds_runs=times, mean_seconds=float(np.mean(times)), median_seconds=float(np.median(times)),
            sample_sd_seconds=float(np.std(times, ddof=1)) if iterations > 1 else None,
            baseline_allocated_bytes=base, peak_allocated_bytes=peak, incremental_peak_allocated_bytes=peak-base,
            losses=losses, final_stats=stats, component_aux=cfg['component_aux'], background_aux=cfg['background_aux'])
        print(a.model, a.fold, variant, results[variant]['mean_seconds'], flush=True)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    assert sha(cp) == before
    for variant in VARIANTS:
        results[variant]['mean_seconds_minus_C'] = results[variant]['mean_seconds']-results['C']['mean_seconds']
        results[variant]['mean_seconds_ratio_to_C'] = results[variant]['mean_seconds']/results['C']['mean_seconds']
    write(dest, dict(complete=True, smoke=a.smoke, model=a.model, fold=a.fold,
        scope='disposable fixed-batch optimizer-step microbenchmark; not full training cost, not retraining or performance',
        included='zero_grad, BF16 forward, FP32 original loss and scalar stats, backward, clip5, AdamW update; synchronized wall time',
        excluded='sampling, host/device input transfer, component precomputation, current mining, validation, checkpoint IO',
        warmup_steps=warmup, measured_steps=iterations, loss_epoch=6, batch=batch_meta,
        checkpoint=str(cp), checkpoint_sha256=before, checkpoint_unchanged=True,
        same_initial_model_and_optimizer_for_all_loss_arms=True, results=results,
        gpu=a.gpu, gpu_name=torch.cuda.get_device_name(device), torch=str(torch.__version__), cuda=torch.version.cuda,
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),
            **{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/calcification_refinement_v5_common.py', ROOT/'scripts/calcification_refinement_v2_common.py', ROOT/'scripts/calcification_refinement_common.py', ROOT/'scripts/comparative_study/model.py', ROOT/'scripts/improvement_unext_deterministic/model.py']}}))

if __name__ == '__main__':
    main()

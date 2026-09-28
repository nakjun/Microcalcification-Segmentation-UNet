"""Read-only, single-batch diagnosis for the Study C/L/H/LH warmup invariant.

This never opens an experiment job for writing and never runs an epoch.  It
replays fold-0's first training batch, evaluates the four existing v5 loss
graphs at epoch 1 (where warmup is zero), and compares logits, gradients, and
one AdamW update from one shared Study initial state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

# Must be set before the first CUDA context.  The caller may explicitly set a
# different value to compare environments; the recorded report makes it clear.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch


HERE, SCRIPTS = Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]
for path in (HERE, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calcification_refinement_v5_common import V5Sampler, training_loss
from common import OUT, VARIANTS, context, initial_pool, spec, checksum
from model import StudyModel


SELECTED_GRADIENTS = None


def _digest_tensors(items) -> str:
    digest = hashlib.sha256()
    for name, value in items:
        digest.update(name.encode())
        if value is None:
            digest.update(b'<none>')
        else:
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _model_digest(model: torch.nn.Module) -> str:
    return _digest_tensors(model.state_dict().items())


def _gradient_digest(model: torch.nn.Module) -> str:
    return _digest_tensors((name, parameter.grad) for name, parameter in model.named_parameters())


def _one_batch(base_context, device):
    fold = spec(base_context, 0)
    lookup = {row['image_id']: row for row in base_context['rows']}
    rows = [lookup[image_id] for image_id in fold['train_ids']]
    config = dict(base_context['learning_config'])
    sampler = V5Sampler(rows, config, initial_pool(base_context, 0))
    rng = np.random.default_rng(42 + 1009)  # fixed first batch of epoch 1
    examples = [sampler.sample(rng) for _ in range(16)]
    arrays = [np.stack([example[index] for example in examples]) for index in (0, 1, 3, 4)]
    return tuple(torch.from_numpy(array).to(device) for array in arrays), {
        'modes': [example[2] for example in examples],
        'input_target_sha256': hashlib.sha256(arrays[0].tobytes() + arrays[1].tobytes()).hexdigest(),
        'geometry_sha256': hashlib.sha256(arrays[2].tobytes() + arrays[3].tobytes()).hexdigest(),
    }


def _run_variant(config, initial_state, batch, device):
    x, y, weights, valid = batch
    model = StudyModel(config).to(device)
    model.load_state_dict(initial_state, strict=True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    torch.cuda.manual_seed_all(42)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = model(x)
    loss, stats = training_loss(logits, y, weights, valid, config, epoch=1)
    if not torch.isfinite(loss):
        raise AssertionError('non-finite loss')
    loss.backward()
    selected = dict(model.named_parameters())
    selected_digest = _digest_tensors((name, selected[name].grad) for name in (list(selected)[:1] + list(selected)[-1:]))
    gradient_digest = _gradient_digest(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    torch.cuda.synchronize(device)
    return {
        'logits_sha256': _digest_tensors([('logits', logits)]),
        'loss_hex': float(loss.detach()).hex(),
        'gradient_sha256': gradient_digest,
        'selected_gradient_sha256': selected_digest,
        'post_step_model_sha256': _model_digest(model),
        'stats': stats,
    }


def _same_record(left, right):
    keys = ('logits_sha256', 'loss_hex', 'gradient_sha256', 'selected_gradient_sha256', 'post_step_model_sha256')
    return {key: left[key] == right[key] for key in keys}


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write('\n')
        temporary = Path(handle.name)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(args.gpu)
    torch.cuda.set_per_process_memory_fraction(.4, args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    base = context('C')
    batch, batch_info = _one_batch(base, device)
    torch.manual_seed(base['learning_config']['seed'])
    canonical = StudyModel(base['learning_config'])
    initial_state = {name: tensor.detach().cpu().clone() for name, tensor in canonical.state_dict().items()}
    report = {
        'complete': True,
        'diagnostic_only': True,
        'signature': base['signature'],
        'backbone': base['learning_config']['backbone'],
        'verifier_sha256': checksum(__file__),
        'gpu': args.gpu,
        'deterministic_algorithms': bool(args.deterministic),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'torch': str(torch.__version__),
        'cuda': torch.version.cuda,
        'batch': batch_info,
        'initial_model_sha256': _digest_tensors(initial_state.items()),
        'variants': {},
        'errors': {},
    }
    for variant in VARIANTS:
        candidate = context(variant)
        try:
            report['variants'][variant] = _run_variant(candidate['learning_config'], initial_state, batch, device)
        except RuntimeError as error:
            report['errors'][variant] = str(error)
    if 'C' in report['variants']:
        report['vs_C'] = {variant: _same_record(report['variants']['C'], record)
                           for variant, record in report['variants'].items() if variant != 'C'}
        try:
            replay = _run_variant(base['learning_config'], initial_state, batch, device)
            report['same_gpu_C_replay'] = _same_record(report['variants']['C'], replay)
        except RuntimeError as error:
            report['errors']['C_replay'] = str(error)
    if args.compare is not None:
        other = json.loads(args.compare.read_text())
        report['cross_gpu_vs'] = str(args.compare)
        report['cross_gpu_C'] = _same_record(report['variants']['C'], other['variants']['C'])
    output = args.output or OUT / f'determinism_gpu{args.gpu}.json'
    _write_json(output, report)
    print(json.dumps({'output': str(output), 'errors': report['errors'], 'vs_C': report.get('vs_C'),
                      'same_gpu_C_replay': report.get('same_gpu_C_replay')}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()

"""Fixed logical union: retain every v2 pixel, add v3; OOF before held confirmation."""
import fcntl
import os
from pathlib import Path
import time
import traceback
import cv2
import numpy as np
import torch
from calcification_gate_common import atomic_json, checksum
from calcification_gate_sam_common import atomic_npz, disk_guard
from calcification_refinement_common import PatchUNet, baseline_masks, pixel_counts, tiled_probability
from calcification_refinement_coverage_common import reconstructed_mask, candidates
from calcification_refinement_v2_common import component_pixel_groups
from run_calcification_gate_sam import Progress
from train_calcification_refinement_v2 import read, aggregate, component_stats, validate_folds, digest

ROOT = Path('runs/calcification_refinement_v3_union_seed42')
V2 = Path('runs/calcification_refinement_v2_seed42')
V3 = Path('runs/calcification_refinement_v3_seed42')


def union_masks(old, new):
    assert old.shape == new.shape and old.dtype == new.dtype == bool
    union = old | new
    assert np.all(union[old])
    return union


def counts(mask, gy, gx, fill, groups):
    hits = mask[gy, gx]
    tp = int(hits.sum())
    inside = fill[gy, gx]
    tin = int(hits[inside].sum())
    pin = int(mask[fill].sum())
    m = pixel_counts(tp, int(mask.sum()), len(gy))
    m.update(TP_inside_boxes=tin, TP_outside_boxes=tp - tin, FP_inside_boxes=pin - tin,
             FP_outside_boxes=m['FP'] - pin + tin,
             components=component_stats(hits, groups))
    return m


def checked(path, hashes):
    assert checksum(path) == hashes[str(path)], path
    return read(path)


def finalize(records, directory, signature):
    assert len(records) == len({r['image_id'] for r in records})
    metrics = {a: {name: aggregate([r['arms'][a] for r in records], name)
                  for name in ['v2', 'v3', 'union']} for a in ['imagenet', 'mammo']}
    passed = all(m['union']['TP'] >= m['v2']['TP'] and
                 m['union']['positive_macro_Dice'] > m['v2']['positive_macro_Dice'] for m in metrics.values())
    result = {'complete': True, 'signature': signature, 'images': len(records),
        'metrics': metrics, 'upgrade_objective_met': passed,
        'output_files': {str(p): checksum(p) for p in directory.rglob('*') if p.is_file() and p.name != 'complete.json'}}
    atomic_json(directory / 'complete.json', result)
    return result


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    progress = Progress(ROOT, 30)
    try:
        source_complete = read(V2 / 'complete.json')
        final_complete = read(V3 / 'complete.json')
        assert source_complete['complete'] and final_complete['complete']
        s2 = checked(V2 / 'selection.json', source_complete['output_files'])['selected']
        s3 = checked(V3 / 'selection.json', final_complete['output_files'])['selected']
        signatures = read(V2 / 'signature.json')
        for path, sha in signatures['files'].items():
            assert checksum(path) == sha, path
        config = read(V3 / 'config.json')
        folds = read(V2 / 'folds.json')
        rows = read(folds['source_manifest'])
        lookup = {r['image_id']: r for r in rows}
        validate_folds(rows, folds['folds'])
        policies = candidates(read(V2 / 'plan.json')['candidate_inference'])
        p2, p3 = [next(p for p in policies if p['id'] == s['policy']) for s in [s2, s3]]
        protocol = {'method': 'pixelwise logical union of frozen v2 and frozen v3 predictions',
            'v2': s2['candidate_id'], 'v3': s3['candidate_id'], 'seed': 42,
            'OOF_gate': 'Both arms Dice strictly above v2; TP >= v2; no threshold search',
            'negative_FP_area_and_image_count': 'Diagnostic, may increase: union retains original false positives',
            'held32': 'Development confirmation only after OOF passes; no reselection', 'reserved_read': False,
            'files': {str(p): checksum(p) for p in [Path(__file__), Path('scripts/test_calcification_refinement_v3_union.py'),
                       V2 / 'signature.json', V2 / 'selection.json', V3 / 'signature.json', V3 / 'selection.json', V3 / 'complete.json']}}
        signature = digest(protocol)
        if (ROOT / 'protocol.json').exists():
            assert read(ROOT / 'protocol.json') == protocol
        else:
            atomic_json(ROOT / 'protocol.json', protocol)
        torch.set_num_threads(4)
        cv2.setNumThreads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        assert torch.cuda.is_available()
        device = torch.device('cuda:1')
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(.10, device)
        assert torch.cuda.mem_get_info(device)[0] / 2**20 >= 5500
        disk_guard(config)
        if (ROOT / 'complete.json').exists():
            done = read(ROOT / 'complete.json')
            assert done['signature'] == signature
            return
        records = []
        oof_dir = ROOT / 'oof'
        for fold in folds['folds']:
            models, jobs, job_done, prob = [], [], [], []
            for selected in [s2, s3]:
                job = V2 / 'jobs' / selected['variant'] / f'fold_{fold["fold"]}'
                done = checked(job / 'complete.json', source_complete['output_files'])
                checkpoint = job / 'checkpoints' / f'epoch_{selected["epoch"]:02d}.pt'
                assert checksum(checkpoint) == done['output_files'][str(checkpoint)]
                state = torch.load(checkpoint, map_location='cpu', weights_only=False)
                assert state['job']['train_ids'] == fold['train_ids']
                assert state['job']['validation_ids'] == fold['validation_ids']
                model = PatchUNet(config).to(device)
                model.load_state_dict(state['model'], strict=True)
                models.append(model.eval())
                jobs.append(job)
                job_done.append(done)
            for ident in fold['validation_ids']:
                case_path = oof_dir / 'cases' / (ident + '.json')
                if case_path.exists():
                    case = read(case_path)
                    assert case['signature'] == signature
                    records.append(case)
                    continue
                row = lookup[ident]
                image = np.load(row['native_cache_path'], mmap_mode='r')
                prob = [tiled_probability(model, image, config, device)[0] for model in models]
                with np.load(row['target_path']) as target:
                    gy, gx = target['y'], target['x']
                groups = component_pixel_groups(gy, gx, image.shape) if len(gy) else []
                original_cases = [checked(job / 'epochs' / f'{s["epoch"]:02d}' / 'cases' / (ident + '.json'), done['output_files'])
                                  for job, s, done in zip(jobs, [s2, s3], job_done)]
                case = {'signature': signature, 'image_id': ident, 'fold': fold['fold'], 'arms': {}}
                for arm in ['imagenet', 'mammo']:
                    _, fill, base = baseline_masks(config['development_baseline_root'], arm, ident)
                    masks = [reconstructed_mask(probability, fill, bool(base['gate_pass']), policy)
                             for probability, policy in zip(prob, [p2, p3])]
                    values = {name: counts(mask, gy, gx, fill, groups) for name, mask in
                              zip(['v2', 'v3', 'union'], masks + [union_masks(*masks)])}
                    for name, s, original in zip(['v2', 'v3'], [s2, s3], original_cases):
                        for key in ['TP', 'FP', 'FN', 'components']:
                            assert values[name][key] == original['arms'][arm][s['policy']][key], (ident, arm, name, key)
                    case['arms'][arm] = values
                atomic_json(case_path, case)
                records.append(case)
                progress('OOF_union', images=len(records), total=124, fold=fold['fold'])
            del models, model, prob
            torch.cuda.empty_cache()
        oof = finalize(records, oof_dir, signature)
        for arm in ['imagenet', 'mammo']:
            for name, selected in [('v2', s2), ('v3', s3)]:
                for key in ['TP', 'FP', 'FN']:
                    assert oof['metrics'][arm][name][key] == selected['metrics'][arm][key]
        if not oof['upgrade_objective_met']:
            atomic_json(ROOT / 'complete.json', {'complete': True, 'signature': signature,
                'upgrade_objective_met': False, 'held_evaluated': False, 'reserved_read': False})
            progress('complete', upgrade_objective_met=False, held_evaluated=False)
            return
        # Only now inspect held outputs. This fixed union has no tunable parameters.
        directories = [V2 / 'final_fit/epochs/02', V3 / 'final_fit/epochs' / f'{s3["epoch"]:02d}']
        final2 = checked(V2 / 'final_fit/complete.json', source_complete['output_files'])
        final3 = checked(V3 / 'final_fit/complete.json', final_complete['output_files'])
        evaluations = [checked(d / 'complete.json', f['output_files']) for d, f in zip(directories, [final2, final3])]
        held_records = []
        held_dir = ROOT / 'held'
        for row in rows:
            if row['role'] == 'train':
                continue
            ident = row['image_id']
            with np.load(row['target_path']) as target:
                gy, gx = target['y'], target['x']
            shape = (row['height'], row['width'])
            groups = component_pixel_groups(gy, gx, shape) if len(gy) else []
            case = {'signature': signature, 'image_id': ident, 'arms': {}}
            for arm in ['imagenet', 'mammo']:
                _, fill, _ = baseline_masks(config['development_baseline_root'], arm, ident)
                masks = []
                for d, evaluation in zip(directories, evaluations):
                    path = d / 'masks' / arm / (ident + '.npz')
                    assert checksum(path) == evaluation['output_files'][str(path)]
                    with np.load(path) as z:
                        masks.append(np.unpackbits(z['mask'], count=int(np.prod(z['shape']))).reshape(tuple(z['shape'])).astype(bool))
                union = union_masks(*masks)
                case['arms'][arm] = {name: counts(mask, gy, gx, fill, groups) for name, mask in
                                    zip(['v2', 'v3', 'union'], masks + [union])}
                atomic_npz(held_dir / 'masks' / arm / (ident + '.npz'), mask=np.packbits(union), shape=np.array(union.shape))
            atomic_json(held_dir / 'cases' / (ident + '.json'), case)
            held_records.append(case)
        held = finalize(held_records, held_dir, signature)
        atomic_json(ROOT / 'complete.json', {'complete': True, 'signature': signature,
            'upgrade_objective_met': held['upgrade_objective_met'], 'held_evaluated': True,
            'scope': 'seed42, reused OOF and held32 development confirmation; no independent test',
            'reserved_read': False, 'completed_at_unix': time.time(),
            'output_files': {str(p): checksum(p) for p in [ROOT / 'protocol.json', oof_dir / 'complete.json', held_dir / 'complete.json']}})
        progress('complete', upgrade_objective_met=held['upgrade_objective_met'])
    except Exception:
        atomic_json(ROOT / f'failure_{int(time.time())}.json', {'pid': os.getpid(), 'traceback': traceback.format_exc()})
        raise
    finally:
        progress.close()


if __name__ == '__main__':
    main()

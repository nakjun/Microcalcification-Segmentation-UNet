"""Complete native-image validation and locked additional CSAW evaluation."""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from calcification_gate_common import atomic_json, checksum
from calcification_gate_sam_common import atomic_npz, disk_guard, raw_png, save_masks
from calcification_refinement_common import (baseline_masks, candidate_metrics,
    select_candidate, summarize_cases, tiled_probability)


def read_target(row, reserved=False):
    if reserved:
        path = Path(row['source_path']).with_name(Path(row['source_path']).stem + '_calcifications.png')
        target = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert target is not None and target.ndim == 2
        assert target.shape == (row['height'], row['width'])
        gy, gx = np.where(target > 0)
        return gy, gx, {'target_source_path': str(path), 'target_sha256': checksum(path)}
    assert checksum(row['target_path']) == row['target_sha256']
    with np.load(row['target_path']) as target:
        gy, gx = target['y'], target['x']
    return gy, gx, {'target_sha256': row['target_sha256']}


def evaluate(model, rows, config, device, output, signature, epoch, progress,
             selection=None, reserved=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if reserved:
        assert selection is not None and selection['locked_before_reserved_GT']
        assert all(r['split'] == 'reserved' for r in rows)
        assert not {r['study_id'] for r in rows} & set(selection['development_groups'])
    else:
        assert all(r['role'] == 'validation' for r in rows)
    complete_path = output / 'complete.json'
    if complete_path.exists():
        complete = json.loads(complete_path.read_text())
        assert complete['signature_digest'] == signature and complete['epoch'] == epoch
        for path, sha in complete['output_files'].items():
            assert checksum(path) == sha
        return complete
    baseline = config['frozen_pipeline_root'] if reserved else config['development_baseline_root']
    records = {arm: [] for arm in config['arms']}
    start, total_tiles = time.monotonic(), 0
    output_files = {}
    for index, row in enumerate(rows):
        ident = row['image_id']
        case_path = output / 'cases' / f'{ident}.json'
        if case_path.exists():
            case = json.loads(case_path.read_text())
            assert case['signature_digest'] == signature and case['epoch'] == epoch
            assert case['image_id'] == ident
            for path, sha in case.get('output_files', {}).items():
                assert checksum(path) == sha
        else:
            assert checksum(row['source_path']) == row['source_sha256']
            image = raw_png(row['source_path']) if reserved else np.load(row['native_cache_path'], mmap_mode='r')
            assert image.shape == (row['height'], row['width'])
            t0 = time.monotonic()
            probability, ntiles = tiled_probability(model, image, config, device)
            elapsed = time.monotonic() - t0
            masks = {arm: baseline_masks(baseline, arm, ident) for arm in config['arms']}
            case_outputs = {}
            if selection is not None:
                # Preserve raw FP32 values for every pixel eligible under either fixed arm.
                # The model still evaluated the entire native image, including outside boxes.
                eligible = np.logical_or.reduce([m[1] for m in masks.values()])
                positions = np.flatnonzero(eligible).astype(np.uint32)
                disk_guard(config, int(positions.nbytes + positions.size * 4 + 20 * 2**20))
                path = output / 'probabilities' / f'{ident}.npz'
                atomic_npz(path, flat_index=positions, probability=probability.ravel()[positions],
                           shape=np.asarray(image.shape, np.int64))
                case_outputs[str(path)] = checksum(path)
                for arm, (sam, fill, baseline_case) in masks.items():
                    region = fill if selection['constraint'] == 'predicted_boxes' else sam
                    chosen = (probability >= selection['threshold']) & region
                    path = output / arm / 'masks' / f'{ident}.npz'
                    boxes = np.asarray(baseline_case['prediction_boxes'], np.float32).reshape(-1, 5)
                    if not baseline_case['gate_pass']:
                        boxes = np.empty((0, 5), np.float32)
                    sha = save_masks(path, chosen, fill, boxes)
                    case_outputs[str(path)] = sha
            # Both gate/box/SAM constraints and any locked final masks are determined before GT read.
            gy, gx, target_info = read_target(row, reserved=reserved)
            arm_metrics = {}
            for arm, (sam, fill, baseline_case) in masks.items():
                measure_config = config if selection is None else dict(config,
                    threshold_grid=[selection['threshold']], constraint_candidates=[selection['constraint']])
                measured = candidate_metrics(probability, sam, fill, gy, gx, measure_config)
                if selection is not None:
                    keep = ['SAM_baseline', 'box_fill_control', selection['method']]
                    measured['methods'] = {k: measured['methods'][k] for k in keep}
                arm_metrics[arm] = measured | {'gate_pass': baseline_case['gate_pass'],
                    'gate_logit': baseline_case['gate_logit'], 'baseline_case_path': str(Path(baseline) / 'csaw' / arm / 'gated/cases' / f'{ident}.json')}
            case = {'signature_digest': signature, 'epoch': epoch, 'image_id': ident,
                    'study_id': row['study_id'], 'source_sha256': row['source_sha256'],
                    'native_shape': list(image.shape), 'inference_tiles': ntiles,
                    'full_native_inference_seconds': elapsed,
                    'inference_used_GT': False, 'GT_read_after_inference_and_locked_mask': True,
                    'gt_pixels': len(gy), 'arms': arm_metrics, **target_info,
                    'output_files': case_outputs}
            atomic_json(case_path, case)
        for arm in config['arms']:
            records[arm].append(case['arms'][arm] | {'image_id': ident, 'study_id': row['study_id']})
        total_tiles += case['inference_tiles']
        output_files.update(case['output_files'])
        output_files[str(case_path)] = checksum(case_path)
        progress('reserved_evaluation' if reserved else 'held_evaluation', epoch=epoch,
                 images=index + 1, total=len(rows), tiles=total_tiles,
                 elapsed_seconds=time.monotonic() - start, output=str(output))
    summaries = {arm: summarize_cases(records[arm]) for arm in config['arms']}
    choice = selection if selection is not None else select_candidate(summaries, config, epoch)
    result = {'complete': True, 'signature_digest': signature, 'epoch': epoch,
              'cohort': 'previously evaluated CSAW reserved; supervised in-domain diagnostic' if reserved else 'CSAW development internal held groups',
              'images': len(rows), 'groups': len({r['study_id'] for r in rows}),
              'selected_candidate': choice, 'metrics': summaries,
              'gate_positive_losses': {a: sum(c['methods']['SAM_baseline']['gt_pixels'] > 0 and not c['gate_pass'] for c in records[a]) for a in config['arms']},
              'tile_count': total_tiles, 'elapsed_seconds': time.monotonic() - start,
              'output_files': output_files, 'GT_drives_inference': False}
    atomic_json(complete_path, result)
    return result

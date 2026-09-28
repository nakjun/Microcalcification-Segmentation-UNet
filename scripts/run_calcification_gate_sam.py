"""Run the frozen gate -> native RetinaNet predicted boxes -> fixed SAM pipeline."""
import argparse
import csv
import fcntl
import gc
import hashlib
import json
import os
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

from calcification_gate_common import atomic_json, checksum, pixel_hash
from calcification_gate_sam_common import (
    atomic_npz, conditional_sam, crop_bounds, detector_predictions, disk_guard,
    gate_logits, load_detector, load_gate, load_masks, load_sam, native_dicom,
    raw_png, save_masks, segmentation_counts, selected_boxes, summarize_segmentation,
)
from prepare_calcification_gate_sam import prepare


class Progress:
    def __init__(self, root, seconds):
        self.root, self.seconds = root, seconds
        self.latest = {'state': 'starting', 'pid': os.getpid(), 'updated_at_unix': time.time()}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.thread.start()

    def __call__(self, state, **data):
        self.latest = {'state': state, 'pid': os.getpid(), 'updated_at_unix': time.time(), **data}
        atomic_json(self.root / 'status.json', self.latest)
        print(json.dumps(self.latest, allow_nan=False), flush=True)

    def heartbeat(self):
        while not self.stop.is_set():
            value = dict(self.latest, heartbeat_at_unix=time.time())
            atomic_json(self.root / 'pipeline_status.json', value)
            with (self.root / 'heartbeat.jsonl').open('a') as handle:
                handle.write(json.dumps(value, allow_nan=False) + '\n')
            self.stop.wait(self.seconds)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
        atomic_json(self.root / 'pipeline_status.json', dict(self.latest, heartbeat_at_unix=time.time()))


def source_signature(config_path, prepared):
    paths = [config_path, __file__, 'scripts/calcification_gate_sam_common.py',
             'scripts/prepare_calcification_gate_sam.py', 'scripts/calcification_gate_common.py',
             'scripts/train_calcification_gate_finetune.py', 'scripts/extract_calcification_features.py',
             'scripts/train_retinanet_roi.py', 'scripts/prepare_retinanet_roi.py',
             'src/mammo_baselines/metrics.py', 'src/mammo_baselines/runtime.py',
             'runs/retinanet_roi_pair/data/manifest.json']
    paths += [str(p) for p in sorted(Path('external/Mammo-CLIP/src/codebase/Detectors').rglob('*.py'))]
    paths += [str(p) for p in sorted(Path('external/segment-anything/segment_anything').rglob('*.py'))]
    config = json.loads(Path(config_path).read_text())
    paths += [config['sam_weights'], config['clean_manifest'], config['csaw_manifest'], config['csaw_split']]
    files = {str(p): checksum(p) for p in paths}
    # The exported gate contains only its trained tail; bind its frozen prefix and original provenance too.
    saved_gate = torch.load(Path(config['root']) / 'frozen/gate/classifier.pt', map_location='cpu', weights_only=False)
    for path, expected in saved_gate['signature']['files'].items():
        assert checksum(path) == expected, f'Frozen gate prefix/training dependency changed: {path}'
        files[path] = expected
    files.update(prepared['files'])
    return {'files': files, 'torch_version': str(torch.__version__), 'numpy_version': np.__version__,
            'opencv_version': cv2.__version__}


def signature_digest(signature):
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()


def image_loader(row, dataset):
    if dataset == 'vindr':
        return native_dicom(row['source_path'], (row['height'], row['width']), roi_record=row)
    assert checksum(row['source_path']) == row['source_sha256']
    return raw_png(row['source_path'])


def infer_case(config, dataset, arm, route, row, gate, boxes, predictor, out, digest):
    case_dir = out / route / 'cases'
    mask_dir = out / route / 'masks'
    case_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    case_path = case_dir / f"{row['image_id']}.json"
    mask_path = mask_dir / f"{row['image_id']}.npz"
    passed = gate['passed'] if route == 'gated' else True
    if case_path.exists():
        case = json.loads(case_path.read_text())
        assert case['signature_digest'] == digest and case['image_id'] == row['image_id']
        assert case['gate_logit'] == gate['logit'] and case['gate_pass'] == gate['passed']
        assert case['prediction_boxes'] == boxes.tolist()
        mask, fill, saved_boxes = load_masks(mask_path, case['mask_sha256'])
        assert mask.shape == (row['height'], row['width'])
        assert np.array_equal(saved_boxes, boxes if passed else boxes[:0])
        return case
    started = time.monotonic()
    mask, fill, meta = conditional_sam(predictor, lambda: image_loader(row, dataset),
                                     (row['height'], row['width']), boxes, passed, config)
    disk_guard(config, 64 * 2**20)
    mask_sha = save_masks(mask_path, mask, fill, boxes if passed else boxes[:0])
    case = {'image_id': row['image_id'], 'study_id': row['study_id'], 'dataset': dataset,
            'arm': arm, 'route': route, 'gate_logit': gate['logit'], 'gate_pass': gate['passed'],
            'prediction_boxes': boxes.tolist(), 'selected_box_count': len(boxes),
            'sam_calls': meta['sam_calls'], 'prompts': meta['prompts'],
            'skip_reason': meta.get('skip_reason'), 'mask_path': str(mask_path), 'mask_sha256': mask_sha,
            'predicted_pixels': int(mask.sum()), 'box_fill_pixels': int(fill.sum()),
            'elapsed_seconds': time.monotonic() - started, 'signature_digest': digest,
            'gt_read_for_inference': False}
    atomic_json(case_path, case)
    return case


def baseline_case(row, gates, gated, extras):
    key = row['image_id']
    return gated[key] if gates[key]['passed'] or key not in extras else extras[key]


def run_masks(config, dataset, arm, rows, gates, predictions, threshold, predictor, out, digest, progress):
    ids = {r['image_id'] for r in rows}
    assert len(ids) == len(rows) and ids <= set(gates) and ids <= set(predictions), 'Missing gate/detector prediction ID'
    filtered = {r['image_id']: selected_boxes(predictions.get(r['image_id'], np.empty((0, 5), np.float32)),
                                               threshold, r['height'], r['width']) for r in rows}
    gated, extras = {}, {}
    started = time.monotonic()
    for index, row in enumerate(rows):
        key = row['image_id']
        gated[key] = infer_case(config, dataset, arm, 'gated', row, gates[key], filtered[key],
                                predictor, out, digest)
        if (index + 1) % 20 == 0 or index + 1 == len(rows):
            progress('gated_sam', dataset=dataset, arm=arm, images=index + 1, total=len(rows),
                     sam_calls=sum(c['sam_calls'] for c in gated.values()),
                     elapsed_seconds=time.monotonic() - started)
    # Only after all primary gated outputs are written, obtain rejected-image baseline masks.
    rejected = [r for r in rows if not gates[r['image_id']]['passed'] and len(filtered[r['image_id']])]
    for index, row in enumerate(rejected):
        key = row['image_id']
        extras[key] = infer_case(config, dataset, arm, 'no_gate_extra', row, gates[key], filtered[key],
                                 predictor, out, digest)
        if (index + 1) % 10 == 0 or index + 1 == len(rejected):
            progress('baseline_extra_sam', dataset=dataset, arm=arm, images=index + 1,
                     total=len(rejected), sam_calls=sum(c['sam_calls'] for c in extras.values()))
    assert len(gated) == len(rows)
    baseline = {r['image_id']: baseline_case(r, gates, gated, extras) for r in rows}
    for row in rows:
        key = row['image_id']
        assert gated[key]['sam_calls'] == (len(filtered[key]) if gates[key]['passed'] else 0)
        assert baseline[key]['sam_calls'] == len(filtered[key])
    summary = {'gated': summarize_masks(rows, gates, gated),
               'no_gate': summarize_masks(rows, gates, baseline),
               'actual_baseline_extra_calls': sum(c['sam_calls'] for c in extras.values()),
               'actual_baseline_extra_images': len(extras),
               'runtime_including_IO_and_resume_verification_seconds': time.monotonic() - started,
               'runtime_scope': 'not a controlled latency benchmark; shared GPU and mask reuse',
               'saved_gated_union_masks': len(gated), 'saved_baseline_extra_union_masks': len(extras)}
    atomic_json(out / 'mask_inference_complete.json', {'complete': True, 'signature_digest': digest,
                'images': len(rows), 'summary': summary, 'gt_read_for_inference': False})
    return gated, baseline, filtered, summary


def summarize_masks(rows, gates, cases):
    return {'population_images': len(rows), 'gate_pass_images': sum(gates[r['image_id']]['passed'] for r in rows),
            'images_with_sam_calls': sum(c['sam_calls'] > 0 for c in cases.values()),
            'images_without_sam_calls': sum(c['sam_calls'] == 0 for c in cases.values()),
            'sam_calls': sum(c['sam_calls'] for c in cases.values()),
            'nonempty_union_masks': sum(c['predicted_pixels'] > 0 for c in cases.values()),
            'predicted_pixels': sum(c['predicted_pixels'] for c in cases.values()),
            'box_fill_pixels': sum(c['box_fill_pixels'] for c in cases.values())}


def overlays(config, dataset, arm, rows, gates, gated, out):
    candidates = [r for r in rows if gates[r['image_id']]['passed'] and gated[r['image_id']]['sam_calls']]
    candidates.sort(key=lambda r: hashlib.sha256(('42:' + r['image_id']).encode()).hexdigest())
    selected = candidates[:config['overlay_count_per_arm_dataset']]
    dest = out / 'overlays'
    dest.mkdir(exist_ok=True)
    artifacts = []
    for row in selected:
        case = gated[row['image_id']]
        mask, _, boxes = load_masks(case['mask_path'], case['mask_sha256'])
        image = image_loader(row, dataset)
        bounds, _ = crop_bounds(boxes[0], row['height'], row['width'], config)
        x0, y0, x1, y1 = bounds
        view = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
        predicted = mask[y0:y1, x0:x1]
        view[predicted] = (.45 * view[predicted] + .55 * np.asarray([255, 100, 0])).astype(np.uint8)
        if dataset == 'csaw':
            # Prediction/crop is fixed before GT is loaded solely for visualization.
            gt = read_csaw_gt(row)[y0:y1, x0:x1]
            contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(view, contours, -1, (0, 255, 255), 1)
        for box in boxes:
            a, b, c, d = box[:4]
            cv2.rectangle(view, (int(a - x0), int(b - y0)), (int(c - x0), int(d - y0)), (0, 220, 0), 1)
        path = dest / f"{row['image_id']}.png"
        assert cv2.imwrite(str(path), view)
        artifacts.append({'image_id': row['image_id'], 'path': str(path), 'crop_xyxy': list(bounds),
                          'selection': 'first6 deterministic SHA256(seed:image_id) among passed images with boxes',
                          'legend': 'green predicted box; blue SAM union; yellow external GT contour when available'})
    atomic_json(dest / 'manifest.json', artifacts)
    return artifacts


def read_csaw_gt(row):
    path = Path(row['source_path'])
    mask_path = path.with_name(path.stem + '_calcifications.png')
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.ndim != 2 or mask.shape != (row['height'], row['width']):
        raise RuntimeError(f'Missing or mismatched external label: {mask_path}')
    return mask > 0


def evaluate_vindr(config, arm, rows, gates, predictions, threshold, gated, baseline, summary, out):
    from mammo_baselines.metrics import fixed_metrics
    # Box labels are accessed for evaluation only after all image masks are committed.
    gated_predictions = {r['image_id']: predictions[r['image_id']] if gates[r['image_id']]['passed']
                         else np.empty((0, 5), np.float32) for r in rows}
    result = {'complete': True, 'dataset': 'clean VinDr development validation', 'arm': arm,
              'seed': config['seed'], 'images': len(rows), 'threshold': threshold, 'mask_summary': summary,
              'detection': {'no_gate': fixed_metrics(rows, predictions, threshold),
                            'gated': fixed_metrics(rows, gated_predictions, threshold)},
              'gate_positive_images': sum(bool(r['label']) for r in rows),
              'gate_positive_images_rejected': sum(bool(r['label']) and not gates[r['image_id']]['passed'] for r in rows),
              'pixel_Dice': None, 'pixel_IoU': None,
              'pixel_metric_scope': 'VinDr supplies boxes, not pixel GT masks; pixel Dice/IoU unmeasured',
              'detector_training_exposure': config['detector_training_exposure'],
              'scope': config['primary_scope'], 'official_test_evaluated': False}
    result['overlays'] = overlays(config, 'vindr', arm, rows, gates, gated, out)
    atomic_json(out / 'complete.json', result)
    return result


def evaluate_csaw(config, arm, rows, gates, gated, baseline, summary, out, progress):
    cases = {'gated': [], 'no_gate': []}
    for index, row in enumerate(rows):
        key = row['image_id']
        # All predicted masks are committed; GT cannot influence any gate, box, crop or mask.
        gt = read_csaw_gt(row)
        for mode, predictions in [('gated', gated), ('no_gate', baseline)]:
            case = predictions[key]
            mask, fill, _ = load_masks(case['mask_path'], case['mask_sha256'])
            cases[mode].append({'image_id': key, 'study_id': row['study_id'], 'gate_pass': gates[key]['passed'],
                                'sam': segmentation_counts(mask, gt), 'box_fill': segmentation_counts(fill, gt)})
        if (index + 1) % 20 == 0 or index + 1 == len(rows):
            progress('external_pixel_evaluation', arm=arm, images=index + 1, total=len(rows))
    atomic_json(out / 'cases.json', cases)
    result = {'complete': True, 'dataset': 'CSAW-S reserved', 'arm': arm, 'seed': config['seed'],
              'images': len(rows), 'mask_summary': summary,
              'metrics': {mode: {method: summarize_segmentation(values, method) for method in ['sam', 'box_fill']}
                          for mode, values in cases.items()},
              'gate_positive_images_rejected': sum(c['sam']['gt_pixels'] > 0 and not c['gate_pass'] for c in cases['gated']),
              'gate_positive_pixels_rejected': sum(c['sam']['gt_pixels'] for c in cases['gated'] if not c['gate_pass']),
              'scope': config['external_scope'], 'thresholds_reselected_on_csaw': False,
              'official_test_evaluated': False}
    result['overlays'] = overlays(config, 'csaw', arm, rows, gates, gated, out)
    atomic_json(out / 'complete.json', result)
    return result


def padded_roi(row):
    if 'source_sha256' in row:
        assert checksum(row['source_path']) == row['source_sha256']
    image = cv2.imread(row['roi_cache_path'], cv2.IMREAD_UNCHANGED)
    assert image is not None and image.shape == (1520, 912) and image.dtype == np.uint8
    assert pixel_hash(image) == row['roi_sha256']
    padded = np.zeros((1536, 928), dtype=np.uint8)
    padded[:1520, :912] = image
    return padded


def csaw_gate_scores(config, rows, device, digest, progress):
    root = Path(config['root'])
    path = root / 'csaw/gate_scores.npz'
    meta_path = path.with_suffix('.json')
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        assert meta['signature_digest'] == digest and meta['sha256'] == checksum(path)
        with np.load(path, allow_pickle=False) as z:
            assert z['image_ids'].tolist() == [r['image_id'] for r in rows]
            logits = z['logits']
    else:
        encoder, gate, gate_config = load_gate(root / 'frozen/gate/classifier.pt', device)
        values = []
        size = config['gate_batch_size']
        for offset in range(0, len(rows), size):
            batch = np.stack([padded_roi(r) for r in rows[offset:offset + size]])
            values.append(gate_logits(encoder, gate, torch.from_numpy(batch)[:, None], gate_config, device))
            progress('external_gate_scoring', images=min(offset + size, len(rows)), total=len(rows))
        logits = np.concatenate(values)
        atomic_npz(path, image_ids=np.asarray([r['image_id'] for r in rows]), logits=logits)
        atomic_json(meta_path, {'signature_digest': digest, 'sha256': checksum(path),
                               'GT_read': False, 'threshold_selected_here': False})
        del encoder, gate
        gc.collect()
        torch.cuda.empty_cache()
    threshold = json.loads((root / 'frozen/gate/thresholds.json').read_text())['primary']['threshold_logit']
    return {r['image_id']: {'logit': float(score), 'passed': bool(score >= threshold)} for r, score in zip(rows, logits)}


def csaw_detector_predictions(config, arm, rows, gates, device, digest, progress):
    root = Path(config['root'])
    out = root / 'csaw' / arm / 'detector'
    out.mkdir(parents=True, exist_ok=True)
    predictions, model = {}, None
    # Rejected images are processed only as the explicitly separate no-gate baseline.
    ordered = sorted(rows, key=lambda r: (not gates[r['image_id']]['passed'], r['image_id']))
    size = config['detector_batch_size']
    for offset in range(0, len(ordered), size):
        pair = ordered[offset:offset + size]
        pair_ids = [r['image_id'] for r in pair]
        paths = [out / f"{r['image_id']}.npz" for r in pair]
        if not all(path.with_suffix('.json').exists() for path in paths):
            if model is None:
                model = load_detector(arm, root / 'frozen' / arm / 'best.pt', device)
            batch_predictions = detector_predictions(model, [padded_roi(r) for r in pair], [r['roi'] for r in pair], device)
            for row, path, boxes in zip(pair, paths, batch_predictions):
                if path.with_suffix('.json').exists():
                    continue
                atomic_npz(path, boxes=boxes)
                atomic_json(path.with_suffix('.json'), {'signature_digest': digest, 'sha256': checksum(path),
                    'GT_read': False, 'physical_batch_image_ids': pair_ids,
                    'scope': 'gated_pass_inference' if gates[row['image_id']]['passed'] else 'no_gate_baseline_only'})
        for row, path in zip(pair, paths):
            meta = json.loads(path.with_suffix('.json').read_text())
            assert meta['signature_digest'] == digest and meta['sha256'] == checksum(path)
            assert meta['physical_batch_image_ids'] == pair_ids
            with np.load(path, allow_pickle=False) as z:
                predictions[row['image_id']] = z['boxes']
        if (offset + len(pair)) % 10 == 0 or offset + len(pair) == len(rows):
            progress('external_detector', arm=arm, images=offset + len(pair), total=len(rows))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return predictions


def smoke(config, prepared, device, progress, digest):
    root = Path(config['root'])
    out = root / 'smoke'
    out.mkdir(exist_ok=True)
    rows = json.loads((root / 'inputs/vindr_manifest.json').read_text())
    with np.load(root / 'inputs/vindr_gate.npz', allow_pickle=False) as z:
        scores = dict(zip(z['image_ids'].tolist(), z['logits'].tolist()))
    gate_encoder, gate, gate_config = load_gate(root / 'frozen/gate/classifier.pt', device)
    # Reproduce the original validation batch shape/order before judging checkpoint identity.
    probes = rows[:8]
    batch = torch.from_numpy(np.stack([padded_roi(r) for r in probes]))[:, None]
    actual_scores = gate_logits(gate_encoder, gate, batch, gate_config, device)
    reference_scores = np.asarray([scores[r['image_id']] for r in probes])
    max_difference = float(np.max(np.abs(actual_scores - reference_scores)))
    atomic_json(out / 'gate_reproduction.json', {'batch_size': 8,
        'image_ids': [r['image_id'] for r in probes], 'reference_logits': reference_scores.tolist(),
        'actual_logits': actual_scores.tolist(), 'max_abs_difference': max_difference})
    assert max_difference < 1e-4, f'Original batch8 gate checkpoint mismatch: {max_difference}'
    del gate_encoder, gate
    gc.collect()
    torch.cuda.empty_cache()
    checks = {'gate_checkpoint_score_max_abs_difference': max_difference, 'gate_batch_size': 8, 'arms': {}}
    original_val = [r for r in json.loads(Path('runs/retinanet_roi_pair/data/manifest.json').read_text()) if r['split'] == 'val']
    for arm in config['arms']:
        reference = torch.load(root / 'frozen' / arm / 'val_internal.pt', map_location='cpu', weights_only=False)['predictions']
        threshold = prepared['arms'][arm]['threshold']
        row = next(r for r in rows if scores[r['image_id']] >= prepared['gate_threshold']
                   and len(selected_boxes(reference[r['image_id']], threshold, r['height'], r['width'])))
        model = load_detector(arm, root / 'frozen' / arm / 'best.pt', device)
        index = next(i for i, candidate in enumerate(original_val) if candidate['image_id'] == row['image_id'])
        pair = original_val[(index // 2) * 2:(index // 2) * 2 + 2]
        batch = []
        for candidate in pair:
            image = cv2.imread(candidate['roi_cache_path'], cv2.IMREAD_UNCHANGED)
            assert image.dtype == np.uint8 and image.shape == (1520, 912)
            padded = np.zeros((1536, 928), dtype=np.uint8)
            padded[:1520, :912] = image
            batch.append(padded)
        actual = detector_predictions(model, batch, [r['roi'] for r in pair], device)[index % 2]
        expected = reference[row['image_id']]
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual[:, :4], expected[:, :4], atol=.5, rtol=1e-4)
        np.testing.assert_allclose(actual[:, 4], expected[:, 4], atol=1e-4, rtol=1e-3)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        predictor = load_sam(config, device)
        selected = selected_boxes(expected, threshold, row['height'], row['width'])[:1]
        mask, fill, meta = conditional_sam(predictor, lambda: image_loader(row, 'vindr'),
            (row['height'], row['width']), selected, True, config)
        assert meta['sam_calls'] == 1
        mask_path = out / f'{arm}_mask.npz'
        mask_digest = save_masks(mask_path, mask, fill, selected)
        loaded, _, _ = load_masks(mask_path, mask_digest)
        assert np.array_equal(loaded, mask)
        def forbidden_loader():
            raise AssertionError('Rejected gate decoded an image')
        empty, _, skip = conditional_sam(predictor, forbidden_loader, (9, 13), selected, False, config)
        assert skip['sam_calls'] == 0 and not empty.any()
        checks['arms'][arm] = {'image_id': row['image_id'], 'raw_prediction_rows': len(actual),
            'physical_batch_image_ids': [r['image_id'] for r in pair],
            'max_box_abs_difference': float(np.max(np.abs(actual[:, :4] - expected[:, :4]))),
            'max_score_abs_difference': float(np.max(np.abs(actual[:, 4] - expected[:, 4]))),
            'detector_snapshot_reproduces_saved_predictions': True, 'sam_calls': 1,
            'native_output_shape': list(mask.shape), 'mask_reload_exact': True, 'rejected_gate_SAM_calls': 0}
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
    checks.update(complete=True, smoke=True, scope='2 predicted-box SAM calls and model-loading checks only',
                  peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                  signature_digest=digest)
    atomic_json(out / 'complete.json', checks)
    progress('smoke_complete', checks=checks)


def run(args):
    config = json.loads(Path(args.config).read_text())
    root = Path(config['root'])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'failure.json').exists():
        history = root / 'failure_history'
        history.mkdir(exist_ok=True)
        (root / 'failure.json').replace(history / f'failure_{time.time_ns()}.json')
    progress = Progress(root, config['heartbeat_seconds'])
    started = time.monotonic()
    try:
        cv2.setNumThreads(1)
        torch.set_num_threads(config['cpu_threads'])
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        prepared = prepare(config, progress)
        if args.prepare:
            return
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.cuda.set_device(config['gpu'])
        torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'], config['gpu'])
        free, _ = torch.cuda.mem_get_info(config['gpu'])
        if free / 2**20 < config['gpu_min_free_mib']:
            raise RuntimeError('Insufficient free GPU memory; rerun when resources are available')
        device = f"cuda:{config['gpu']}"
        sig = source_signature(args.config, prepared)
        digest = signature_digest(sig)
        if args.smoke:
            smoke(config, prepared, device, progress, digest)
            return
        smoke_done = json.loads((root / 'smoke/complete.json').read_text())
        assert smoke_done['complete'] and smoke_done['signature_digest'] == digest, 'Current-protocol GPU smoke required'
        sig_path = root / 'signature.json'
        if sig_path.exists():
            assert json.loads(sig_path.read_text()) == sig, 'Source/config/input signature changed; use a new version'
        else:
            atomic_json(sig_path, sig)
        if (root / 'complete.json').exists():
            complete = json.loads((root / 'complete.json').read_text())
            assert complete['signature_digest'] == digest
            for path, expected in complete['output_files'].items():
                assert checksum(path) == expected, f'Completed output changed: {path}'
            progress('complete', reused=True)
            return
        rows = json.loads((root / 'inputs/vindr_manifest.json').read_text())
        with np.load(root / 'inputs/vindr_gate.npz', allow_pickle=False) as z:
            gates = {key: {'logit': float(score), 'passed': bool(passed)} for key, score, passed
                     in zip(z['image_ids'].tolist(), z['logits'], z['gate_pass'])}
        predictor = load_sam(config, device)
        results = {'vindr': {}, 'csaw': {}}
        for arm in config['arms']:
            progress('vindr_starting', arm=arm)
            predictions = torch.load(root / 'frozen' / arm / 'val_internal.pt', map_location='cpu', weights_only=False)['predictions']
            threshold = prepared['arms'][arm]['threshold']
            out = root / 'vindr' / arm
            gated, baseline, _, summary = run_masks(config, 'vindr', arm, rows, gates, predictions,
                threshold, predictor, out, digest, progress)
            results['vindr'][arm] = evaluate_vindr(config, arm, rows, gates, predictions, threshold,
                                                  gated, baseline, summary, out)
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
        atomic_json(root / 'vindr_complete.json', {'complete': True, 'results': results['vindr'],
                    'signature_digest': digest, 'scope': config['primary_scope']})
        csaw_rows = json.loads((root / 'inputs/csaw_manifest.json').read_text())
        csaw_gates = csaw_gate_scores(config, csaw_rows, device, digest, progress)
        for arm in config['arms']:
            predictions = csaw_detector_predictions(config, arm, csaw_rows, csaw_gates, device, digest, progress)
            predictor = load_sam(config, device)
            threshold = prepared['arms'][arm]['threshold']
            out = root / 'csaw' / arm
            gated, baseline, _, summary = run_masks(config, 'csaw', arm, csaw_rows, csaw_gates, predictions,
                threshold, predictor, out, digest, progress)
            results['csaw'][arm] = evaluate_csaw(config, arm, csaw_rows, csaw_gates, gated, baseline, summary, out, progress)
            del predictor
            gc.collect()
            torch.cuda.empty_cache()
        done = {'complete': True, 'smoke': False, 'signature_digest': digest,
                'finished_at_unix': time.time(), 'elapsed_seconds': time.monotonic() - started,
                'seed': config['seed'], 'gate_threshold': prepared['gate_threshold'], 'detectors': prepared['arms'],
                'results': results, 'official_test_evaluated': False, 'new_training_performed': False,
                'GT_box_prompts_used': False, 'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20}
        outputs = [p for dataset in ['vindr', 'csaw'] for p in (root / dataset).rglob('*') if p.is_file()]
        done['output_files'] = {str(path): checksum(path) for path in outputs}
        atomic_json(root / 'complete.json', done)
        progress('complete', elapsed_seconds=done['elapsed_seconds'], vindr_images=3172, csaw_images=156)
    except BaseException as exc:
        atomic_json(root / 'failure.json', {'error': repr(exc), 'traceback': traceback.format_exc(),
                                          'updated_at_unix': time.time(), 'pid': os.getpid()})
        progress('failed', error=repr(exc))
        raise
    finally:
        progress.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/calcification_gate_sam.json')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())

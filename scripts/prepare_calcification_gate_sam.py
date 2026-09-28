"""Freeze inputs and prepare image-only CSAW ROIs without reading masks."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from calcification_gate_common import atomic_json, checksum, pixel_hash
from calcification_gate_sam_common import atomic_copy, atomic_npz, disk_guard, freeze_detector_pair, png_roi, raw_png, selected_boxes


def prepare(config, progress):
    root = Path(config['root'])
    inputs = root / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    disk_guard(config, 1024 * 2**20)
    if (root / 'prepared.json').exists():
        prepared = json.loads((root / 'prepared.json').read_text())
        for path, digest in prepared['files'].items():
            assert checksum(path) == digest, f'Prepared input changed: {path}'
        return prepared
    source_gate = Path(config['gate_root'])
    gate_done = json.loads((source_gate / 'complete.json').read_text())
    assert gate_done['complete'] and not gate_done['smoke']
    assert checksum(source_gate / 'final_seed42/classifier.pt') == gate_done['classifier_sha256']
    for path, digest in gate_done['signature']['files'].items():
        assert checksum(path) == digest, f'Gate training provenance changed: {path}'
    frozen_gate = root / 'frozen/gate'
    frozen_gate.mkdir(parents=True, exist_ok=True)
    for name in ['classifier.pt', 'thresholds.json', 'validation_scores.npz']:
        src = source_gate / 'final_seed42' / name
        target = frozen_gate / name
        before = checksum(src)
        if not target.exists():
            atomic_copy(src, target)
        assert checksum(target) == before == checksum(src)
    threshold = json.loads((frozen_gate / 'thresholds.json').read_text())['primary']['threshold_logit']
    assert threshold == -11.328035354614258
    rows = sorted([r for r in json.loads(Path(config['clean_manifest']).read_text()) if r['split'] == 'val'],
                  key=lambda r: r['image_id'])
    assert len(rows) == 3172
    with np.load(frozen_gate / 'validation_scores.npz', allow_pickle=False) as z:
        ids, labels, logits = z['image_ids'], z['labels'], z['logits']
    assert ids.tolist() == [r['image_id'] for r in rows]
    assert labels.tolist() == [r['label'] for r in rows] and np.isfinite(logits).all()
    passed = logits >= threshold
    assert int(passed.sum()) == 2904 and int(labels[passed].sum()) == int(labels.sum()) == 59
    atomic_json(inputs / 'vindr_manifest.json', rows)
    atomic_npz(inputs / 'vindr_gate.npz', image_ids=ids, labels=labels, logits=logits, gate_pass=passed)
    arms, calls = {}, {}
    for arm in config['arms']:
        arms[arm] = freeze_detector_pair(config, arm)
        # Verify the immutable detector source matches the running experiment's recorded source.
        init = json.loads((Path(config['detector_root']) / f'{arm}_seed42/initialization.json').read_text())
        for path, digest in init['source_hashes'].items():
            assert checksum(path) == digest, f'Detector training source changed: {path}'
        pred = torch.load(root / 'frozen' / arm / 'val_internal.pt', map_location='cpu', weights_only=False)['predictions']
        assert {r['image_id'] for r in rows} <= set(pred)
        counts = np.asarray([len(selected_boxes(pred[r['image_id']], arms[arm]['threshold'],
                                                r['height'], r['width'])) for r in rows])
        calls[arm] = {'no_gate_sam_calls': int(counts.sum()), 'gated_sam_calls': int(counts[passed].sum()),
                      'no_gate_additional_sam_calls_on_rejected': int(counts[~passed].sum()),
                      'gated_images_with_boxes': int(np.sum((counts > 0) & passed)),
                      'gated_images_without_boxes': int(np.sum((counts == 0) & passed))}
    split = json.loads(Path(config['csaw_split']).read_text())
    source_rows = sorted([r for r in json.loads(Path(config['csaw_manifest']).read_text()) if r['split'] == 'reserved'],
                         key=lambda r: r['image_id'])
    assert len(source_rows) == 156 and len({r['image_id'] for r in source_rows}) == 156
    assert {r['study_id'] for r in source_rows} <= set(split['reserved_groups'])
    assert not {r['study_id'] for r in source_rows} & set(split['development_groups'])
    cache = inputs / 'csaw_roi'
    cache.mkdir(exist_ok=True)
    csaw_rows = []
    for index, row in enumerate(source_rows):
        image = raw_png(row['source_path'])
        resized, _, roi = png_roi(image)
        assert image.shape == (row['height'], row['width'])
        path = cache / f"{row['image_id']}.png"
        tmp = path.with_name(path.stem + '.tmp.png')
        assert cv2.imwrite(str(tmp), resized, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        tmp.replace(path)
        csaw_rows.append({k: row[k] for k in ['image_id', 'study_id', 'source_path', 'height', 'width']} | {
            'split': 'CSAW-S reserved', 'roi': roi, 'roi_cache_path': str(path.resolve()),
            'roi_sha256': pixel_hash(resized), 'source_sha256': checksum(row['source_path'])})
        if (index + 1) % 10 == 0:
            progress('preparing_csaw_roi', images=index + 1, total=len(source_rows))
    atomic_json(inputs / 'csaw_manifest.json', csaw_rows)
    files = [inputs / 'vindr_manifest.json', inputs / 'vindr_gate.npz', inputs / 'csaw_manifest.json']
    files += [p for p in frozen_gate.iterdir() if p.is_file()]
    files += [p for arm in config['arms'] for p in (root / 'frozen' / arm).iterdir() if p.is_file()]
    files += sorted(cache.glob('*.png'))
    prepared = {'complete': True, 'files': {str(p): checksum(p) for p in files}, 'arms': arms,
                'gate_threshold': threshold, 'vindr_images': len(rows), 'vindr_gate_pass': int(passed.sum()),
                'vindr_gate_rejected': int((~passed).sum()), 'vindr_expected_calls': calls,
                'csaw_images': len(csaw_rows), 'gt_pixel_masks_read_during_preparation': False,
                'official_test_processed': False}
    atomic_json(root / 'prepared.json', prepared)
    progress('prepared', vindr_images=len(rows), csaw_images=len(csaw_rows), expected_calls=calls)
    return prepared

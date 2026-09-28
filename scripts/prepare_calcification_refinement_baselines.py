"""Generate frozen, image-only CSAW development baselines for pixel refinement."""
import fcntl
import gc
import json
import os
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

from calcification_gate_common import atomic_json, checksum, pixel_hash
from calcification_gate_sam_common import disk_guard, load_sam, png_roi, raw_png
from run_calcification_gate_sam import (Progress, csaw_detector_predictions,
    csaw_gate_scores, run_masks, signature_digest, source_signature)

ROOT = Path('runs/calcification_refinement_v1/baseline_development')
SOURCE = Path('runs/calcification_gate_retinanet_sam_clean_v1')


def prepare(config, progress):
    marker = ROOT / 'prepared.json'
    if marker.exists():
        prepared = json.loads(marker.read_text())
        for path, expected in prepared['files'].items():
            assert checksum(path) == expected
        return json.loads((ROOT / 'inputs/development_manifest.json').read_text()), prepared
    completed = json.loads((SOURCE / 'complete.json').read_text())
    assert completed['complete'] and not completed['GT_box_prompts_used']
    frozen = ROOT / 'frozen'
    if not frozen.exists():
        frozen.symlink_to((SOURCE / 'frozen').resolve(), target_is_directory=True)
    assert frozen.resolve() == (SOURCE / 'frozen').resolve()
    for path, expected in json.loads((SOURCE / 'prepared.json').read_text())['files'].items():
        if '/frozen/' in path:
            assert checksum(path) == expected
    source_rows = sorted([r for r in json.loads(Path(config['csaw_manifest']).read_text())
                          if r['split'] == 'development'], key=lambda r: r['image_id'])
    groups = json.loads(Path(config['csaw_split']).read_text())
    assert len(source_rows) == 156 and len({r['image_id'] for r in source_rows}) == 156
    assert {r['study_id'] for r in source_rows} <= set(groups['development_groups'])
    assert not {r['study_id'] for r in source_rows} & set(groups['reserved_groups'])
    cache = ROOT / 'inputs/roi'
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, row in enumerate(source_rows):
        image = raw_png(row['source_path'])
        assert image.shape == (row['height'], row['width'])
        resized, _, roi = png_roi(image)
        path = cache / f"{row['image_id']}.png"
        tmp = path.with_name(path.stem + '.tmp.png')
        assert cv2.imwrite(str(tmp), resized, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        tmp.replace(path)
        rows.append({k: row[k] for k in ['image_id', 'study_id', 'source_path', 'height', 'width']} | {
            'split': 'development', 'roi': roi, 'roi_cache_path': str(path.resolve()),
            'roi_sha256': pixel_hash(resized), 'source_sha256': checksum(row['source_path'])})
        if (index + 1) % 20 == 0:
            progress('development_ROI', images=index + 1, total=len(source_rows))
    manifest = ROOT / 'inputs/development_manifest.json'
    atomic_json(manifest, rows)
    paths = [manifest, SOURCE / 'complete.json'] + list(cache.glob('*.png'))
    paths += [p for p in frozen.rglob('*') if p.is_file()]
    prepared = {'complete': True, 'files': {str(p): checksum(p) for p in paths},
                'GT_masks_read': False, 'images': len(rows)}
    atomic_json(marker, prepared)
    return rows, prepared


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT / 'failure.json').exists():
        history = ROOT / 'failure_history'
        history.mkdir(exist_ok=True)
        (ROOT / 'failure.json').replace(history / f'failure_{time.time_ns()}.json')
    config = json.loads(Path('configs/calcification_gate_sam.json').read_text())
    config.update(root=str(ROOT), version='calcification_refinement_development_baseline_v1',
        primary_scope='CSAW-S development 156 image-only frozen baseline; GT not read',
        external_scope='development baseline, not external evaluation',
        detector_training_exposure='Fixed intermediate epoch1 VinDr detector snapshots, no new training')
    config_path = ROOT / 'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config
    else:
        atomic_json(config_path, config)
    progress = Progress(ROOT, 30)
    def development_progress(state, **data):
        progress(state.replace('external_', 'development_'), cohort='CSAW-S development', **data)
    start = time.monotonic()
    try:
        cv2.setNumThreads(1)
        torch.set_num_threads(config['cpu_threads'])
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        disk_guard(config, 1024 * 2**20)
        rows, prepared = prepare(config, progress)
        signature = source_signature(str(config_path), prepared)
        signature['files'][__file__] = checksum(__file__)
        digest = signature_digest(signature)
        signature_path = ROOT / 'signature.json'
        if signature_path.exists():
            assert json.loads(signature_path.read_text()) == signature
        else:
            atomic_json(signature_path, signature)
        if (ROOT / 'complete.json').exists():
            completed = json.loads((ROOT / 'complete.json').read_text())
            assert completed['signature_digest'] == digest
            for path, expected in completed['output_files'].items():
                assert checksum(path) == expected
            progress('complete', reused=True)
            return
        assert torch.cuda.is_available()
        torch.cuda.set_device(config['gpu'])
        torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'], config['gpu'])
        assert torch.cuda.mem_get_info(config['gpu'])[0] / 2**20 >= config['gpu_min_free_mib']
        device = f"cuda:{config['gpu']}"
        gates = csaw_gate_scores(config, rows, device, digest, development_progress)
        source_prepared = json.loads((SOURCE / 'prepared.json').read_text())
        results = {}
        for arm in config['arms']:
            predictions = csaw_detector_predictions(config, arm, rows, gates, device, digest, development_progress)
            predictor = load_sam(config, device)
            threshold = source_prepared['arms'][arm]['threshold']
            out = ROOT / 'csaw' / arm
            _, _, _, summary = run_masks(config, 'csaw', arm, rows, gates, predictions,
                                           threshold, predictor, out, digest, development_progress)
            results[arm] = {'complete': True, 'dataset': 'CSAW-S development', 'images': len(rows),
                'arm': arm, 'threshold': threshold, 'mask_summary': summary,
                'signature_digest': digest, 'GT_masks_read': False, 'GT_box_prompts_used': False}
            atomic_json(out / 'complete.json', results[arm])
            del predictor
            gc.collect()
            torch.cuda.empty_cache()
        outputs = [p for p in (ROOT / 'csaw').rglob('*') if p.is_file()]
        completed = {'complete': True, 'signature_digest': digest, 'results': results,
            'GT_masks_read': False, 'GT_box_prompts_used': False, 'new_training_performed': False,
            'finished_at_unix': time.time(), 'elapsed_seconds': time.monotonic() - start,
            'output_files': {str(p): checksum(p) for p in outputs}}
        atomic_json(ROOT / 'complete.json', completed)
        progress('complete', images=len(rows), elapsed_seconds=completed['elapsed_seconds'])
    except BaseException as exc:
        atomic_json(ROOT / 'failure.json', {'error': repr(exc), 'traceback': traceback.format_exc(),
            'pid': os.getpid(), 'updated_at_unix': time.time()})
        progress('failed', error=repr(exc))
        raise
    finally:
        progress.close()


if __name__ == '__main__':
    main()

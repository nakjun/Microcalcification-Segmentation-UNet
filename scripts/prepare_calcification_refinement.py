"""Audit exact duplicates and create development-only grouped supervised data."""
import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from calcification_gate_common import atomic_json, checksum, pixel_hash


def atomic_npy(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.npy.tmp')
    with temporary.open('wb') as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def audit_one(task):
    row, root = task
    cv2.setNumThreads(1)
    root = Path(root)
    path = Path(row['source_path'])
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(f'Unexpected native image: {path}')
    assert image.shape == (row['height'], row['width']) and image.max() > image.min()
    result = {k: row[k] for k in ['image_id', 'study_id', 'source_path', 'height', 'width', 'split']}
    result.update(source_sha256=checksum(path), pixel_sha256=pixel_hash(image))
    if row['split'] == 'development':
        mask_path = path.with_name(path.stem + '_calcifications.png')
        gt = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if gt is None or gt.ndim != 2 or gt.shape != image.shape:
            raise ValueError(f'Missing or mismatched development GT: {mask_path}')
        gt = gt > 0
        yy, xx = np.where(gt)
        target = root / 'data/targets' / f"{row['image_id']}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.npz.tmp')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, y=yy.astype(np.int32), x=xx.astype(np.int32), shape=np.asarray(gt.shape))
        temporary.replace(target)
        image_cache = root / 'data/native_images' / f"{row['image_id']}.npy"
        atomic_npy(image_cache, image)
        result.update(gt_source_path=str(mask_path), gt_source_sha256=checksum(mask_path),
                      gt_pixel_sha256=pixel_hash(gt), gt_pixels=int(gt.sum()), positive=bool(gt.any()),
                      native_cache_path=str(image_cache), native_cache_sha256=checksum(image_cache),
                      target_path=str(target), target_sha256=checksum(target))
    else:
        # Deliberately do not open any reserved mask here.
        result.update(gt_read=False)
    return result


def isolate(rows):
    development = [r for r in rows if r['split'] == 'development']
    parent = {r['study_id']: r['study_id'] for r in development}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    byhash = defaultdict(list)
    for row in rows:
        byhash[row['pixel_sha256']].append(row)
    for group in byhash.values():
        studies = sorted({r['study_id'] for r in group if r['split'] == 'development'})
        for study in studies[1:]:
            a, b = find(studies[0]), find(study)
            parent[max(a, b)] = min(a, b)
    external_overlap, conflicting = set(), set()
    duplicate_groups = []
    for digest, group in byhash.items():
        if len(group) <= 1:
            continue
        dev = [r for r in group if r['split'] == 'development']
        reserved = [r for r in group if r['split'] == 'reserved']
        duplicate_groups.append({'pixel_sha256': digest, 'image_ids': [r['image_id'] for r in group],
                                 'splits': [r['split'] for r in group]})
        if dev and reserved:
            external_overlap.update(find(r['study_id']) for r in dev)
        if len({r['gt_pixel_sha256'] for r in dev}) > 1:
            conflicting.update(find(r['study_id']) for r in dev)
    components = defaultdict(list)
    for row in development:
        components[find(row['study_id'])].append(row)
    retained, ledger = [], []
    for component, group in sorted(components.items()):
        representative = min(r['study_id'] for r in group)
        seen = set()
        for row in sorted(group, key=lambda r: (r['study_id'], r['image_id'])):
            if component in external_overlap:
                reason = 'development_component_overlaps_reserved_image'
            elif component in conflicting:
                reason = 'conflicting_masks_for_identical_development_image'
            elif row['study_id'] != representative:
                reason = 'duplicate_study_component'
            elif row['pixel_sha256'] in seen:
                reason = 'duplicate_image_within_retained_study'
            else:
                reason = 'retained'
                seen.add(row['pixel_sha256'])
                retained.append(dict(row, component_id=component))
            ledger.append({'image_id': row['image_id'], 'study_id': row['study_id'],
                           'component_id': component, 'decision': reason})
    return retained, ledger, duplicate_groups


def main(args):
    config = json.loads(Path(args.config).read_text())
    root = Path(config['root'])
    (root / 'data').mkdir(parents=True, exist_ok=True)
    source = json.loads(Path(config['source_manifest']).read_text())
    group_split = json.loads(Path(config['original_group_split']).read_text())
    assert len(source) == 312 and len({r['image_id'] for r in source}) == 312
    for row in source:
        assert row['study_id'] in group_split[row['split'] + '_groups']
    tasks = [(r, str(root)) for r in sorted(source, key=lambda r: r['image_id'])]
    audit_path = root / 'data/audit_rows.json'
    if audit_path.exists():
        rows = json.loads(audit_path.read_text())
        for row in rows:
            assert checksum(row['source_path']) == row['source_sha256']
            if row['split'] == 'development':
                for p, digest in [('gt_source_path', 'gt_source_sha256'), ('native_cache_path', 'native_cache_sha256'), ('target_path', 'target_sha256')]:
                    assert checksum(row[p]) == row[digest]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=config['audit_workers']) as pool:
            for index, row in enumerate(pool.map(audit_one, tasks), 1):
                rows.append(row)
                if index % 20 == 0 or index == len(tasks):
                    print(json.dumps({'state': 'data_audit', 'images': index, 'total': len(tasks)}), flush=True)
        atomic_json(audit_path, rows)
    retained, ledger, duplicates = isolate(rows)
    studies = sorted({r['study_id'] for r in retained})
    labels = [any(r['positive'] for r in retained if r['study_id'] == study) for study in studies]
    fit_groups, held_groups = train_test_split(studies, test_size=config['held_group_fraction'],
                                              random_state=config['seed'], stratify=labels)
    fit_groups, held_groups = set(fit_groups), set(held_groups)
    manifest = [dict(r, role='train' if r['study_id'] in fit_groups else 'validation') for r in retained]
    manifest.sort(key=lambda r: r['image_id'])
    train = [r for r in manifest if r['role'] == 'train']
    held = [r for r in manifest if r['role'] == 'validation']
    for key in ['study_id', 'component_id', 'pixel_sha256']:
        assert not {r[key] for r in train} & {r[key] for r in held}
    reserved_hashes = {r['pixel_sha256'] for r in rows if r['split'] == 'reserved'}
    assert not {r['pixel_sha256'] for r in manifest} & reserved_hashes
    summary = {}
    for name, subset in [('train', train), ('validation', held)]:
        assert {r['positive'] for r in subset} == {False, True}
        summary[name] = {'images': len(subset), 'groups': len({r['study_id'] for r in subset}),
                         'positive_images': sum(r['positive'] for r in subset),
                         'negative_images': sum(not r['positive'] for r in subset),
                         'gt_pixels': sum(r['gt_pixels'] for r in subset)}
    atomic_json(root / 'data/manifest.json', manifest)
    atomic_json(root / 'data/decisions.json', ledger)
    atomic_json(root / 'data/split.json', {'seed': config['seed'], 'train_groups': sorted(fit_groups),
                'validation_groups': sorted(held_groups), 'reserved_groups': group_split['reserved_groups'],
                'grouping': 'distribution folders; actual patient-ID correspondence unverified'})
    complete = {'complete': True, 'summary': summary, 'duplicate_groups': duplicates,
                'excluded_development_images': len([r for r in rows if r['split'] == 'development']) - len(retained),
                'reserved_GT_read': False, 'reserved_used_for_model_or_threshold_selection': False,
                'manifest_sha256': checksum(root / 'data/manifest.json'),
                'audit_sha256': checksum(audit_path), 'config_sha256_at_preparation': checksum(args.config)}
    atomic_json(root / 'data/complete.json', complete)
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/calcification_refinement.json')
    main(parser.parse_args())

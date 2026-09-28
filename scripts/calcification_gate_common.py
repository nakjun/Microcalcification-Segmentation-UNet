"""Shared integrity, feature-cache, and gate metric helpers."""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def pixel_hash(image):
    h = hashlib.sha256(str((image.shape, image.dtype.str)).encode())
    h.update(image.tobytes())
    return h.hexdigest()


def select_records(rows):
    """Quarantine conflict components; keep one priority-ordered study/component."""
    studies, hashes = {}, {}
    for row in rows:
        studies.setdefault(row['study_id'], []).append(row)
        hashes.setdefault(row['roi_sha256'], []).append(row)
    parent = {sid: sid for sid in studies}

    def find(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid

    for group in hashes.values():
        for row in group[1:]:
            a, b = find(group[0]['study_id']), find(row['study_id'])
            parent[max(a, b)] = min(a, b)
    conflicted = {find(group[0]['study_id']) for group in hashes.values()
                  if len({r['label'] for r in group}) > 1}
    components = {}
    for sid in studies:
        assert len({r['split'] for r in studies[sid]}) == 1
        components.setdefault(find(sid), []).append(sid)
    retained, ledger = [], []
    priority = {'test': 0, 'val': 1, 'train': 2}
    for component, sids in sorted(components.items()):
        representative = min(sids, key=lambda s: (priority[studies[s][0]['split']], s))
        seen = set()
        for sid in sorted(sids):
            for row in sorted(studies[sid], key=lambda r: r['image_id']):
                if component in conflicted:
                    reason = 'label_conflict_component'
                elif sid != representative:
                    reason = 'duplicate_study_component'
                elif row['roi_sha256'] in seen:
                    reason = 'duplicate_image_within_representative'
                else:
                    reason = 'retained'
                    retained.append(dict(row, component_id=component))
                    seen.add(row['roi_sha256'])
                ledger.append({k: row[k] for k in ['image_id', 'study_id', 'split', 'label', 'roi_sha256']}
                              | {'component_id': component, 'representative_study_id': representative,
                                 'decision': reason})
    return sorted(retained, key=lambda r: r['image_id']), sorted(ledger, key=lambda r: r['image_id'])


def gate_metrics(labels, logits, target=1.0):
    labels, logits = np.asarray(labels), np.asarray(logits, dtype=np.float64)
    if labels.shape != logits.shape or labels.ndim != 1 or not np.isfinite(logits).all():
        raise ValueError('Invalid, missing, or nonfinite gate scores')
    if set(np.unique(labels)) != {0, 1} or not 0 < target <= 1:
        raise ValueError('Gate calibration requires both classes and 0 < target <= 1')
    positives = np.sort(logits[labels == 1])
    needed = int(np.ceil(target * len(positives) - 1e-12))
    threshold = float(positives[len(positives) - needed])
    passed = logits >= threshold  # Include the entire tied score group.
    tp = int(passed[labels == 1].sum())
    fp = int(passed[labels == 0].sum())
    np_, nn = len(positives), int((labels == 0).sum())
    return {'target_sensitivity': target, 'threshold_logit': threshold,
            'sensitivity': tp / np_, 'negative_rejection': 1 - fp / nn,
            'pass_rate': float(passed.mean()), 'tp': tp, 'fn': np_ - tp,
            'fp': fp, 'tn': nn - fp, 'images': len(labels), 'positive_images': np_}


def classification_metrics(labels, logits):
    if not np.isfinite(logits).all():
        raise ValueError('Nonfinite scores')
    return {'roc_auc': float(roc_auc_score(labels, logits)),
            'average_precision': float(average_precision_score(labels, logits)),
            'operating_points': [gate_metrics(labels, logits, t) for t in [1., .99, .98, .95]]}


def validate_shard(path, rows, signature):
    path = Path(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    if meta['signature'] != signature or meta['sha256'] != checksum(path):
        raise ValueError(f'Feature shard signature/checksum mismatch: {path}')
    with np.load(path, allow_pickle=False) as z:
        ids, labels, features = z['image_ids'], z['labels'], z['features']
    if ids.tolist() != [r['image_id'] for r in rows] or labels.tolist() != [r['label'] for r in rows]:
        raise ValueError(f'Feature shard ID/label ordering mismatch: {path}')
    if features.shape != (len(rows), 2048) or features.dtype != np.float32 or not np.isfinite(features).all():
        raise ValueError(f'Invalid feature shape/dtype/value: {path}')
    return features

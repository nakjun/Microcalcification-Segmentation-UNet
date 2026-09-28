"""Create an immutable, auditable deduplicated classifier data version."""
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

from calcification_gate_common import atomic_json, checksum, select_records

ROOT = Path('runs/calcification_gate_clean_v1')


def main():
    audit = Path('runs/data_revalidation_v2')
    summary = json.loads((audit / 'summary.json').read_text())
    assert json.loads((audit / 'status.json').read_text())['stage'] == 'complete'
    assert not summary['source_annotation_errors'] and not summary['image_errors']
    assert summary['all_duplicate_ROI_reconstructions_match'] and summary['source_files_unchanged']
    for path, sha in summary['source_hashes'].items():
        assert checksum(path) == sha, f'Source changed since audit: {path}'
    rows = json.loads((audit / 'image_audit.json').read_text())
    assert len(rows) == len({r['image_id'] for r in rows}) == 20000
    retained, ledger = select_records(rows)
    cache = {r['image_id']: r for r in json.loads(Path('runs/retinanet_roi_pair/data/manifest.json').read_text())}
    train_studies = sorted({r['study_id'] for r in retained if r['split'] == 'train'})
    positive_studies = {r['study_id'] for r in retained if r['split'] == 'train' and r['label']}
    folds = {}
    for fold, (_, held) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(
            train_studies, [s in positive_studies for s in train_studies])):
        folds.update({train_studies[i]: fold for i in held})
    records = []
    for r in retained:
        record = {k: r[k] for k in ['image_id', 'study_id', 'split', 'official_split', 'component_id',
                                   'label', 'roi_sha256', 'source_path', 'roi', 'height', 'width']}
        record.update(boxes=r['clipped_boxes'], inner_fold=folds.get(r['study_id']))
        if r['split'] != 'test':
            record['roi_cache_path'] = cache[r['image_id']]['roi_cache_path']
            assert Path(record['roi_cache_path']).is_file()
        records.append(record)
    assert len({r['roi_sha256'] for r in records}) == len(records)
    split_sets = {s: {r['component_id'] for r in records if r['split'] == s} for s in ['train', 'val', 'test']}
    assert all(not split_sets[a] & split_sets[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')])
    fold_counts = []
    for fold in range(5):
        held = [r for r in records if r['split'] == 'train' and r['inner_fold'] == fold]
        fit = [r for r in records if r['split'] == 'train' and r['inner_fold'] != fold]
        assert not {r['component_id'] for r in held} & {r['component_id'] for r in fit}
        assert 0 < sum(r['label'] for r in held) < len(held)
        fold_counts.append({'fold': fold, 'held_images': len(held), 'held_positive': sum(r['label'] for r in held)})
    data = ROOT / 'data'
    data.mkdir(parents=True, exist_ok=True)
    if (data / 'READY.json').exists():
        assert json.loads((data / 'manifest.json').read_text()) == records, 'Frozen data version differs'
        print((data / 'summary.json').read_text())
        return
    atomic_json(data / 'manifest.json', records)
    with (data / 'decisions.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)
    counts = {s: {'images': sum(r['split'] == s for r in records),
                  'positive_images': sum(r['split'] == s and r['label'] == 1 for r in records),
                  'studies': len({r['study_id'] for r in records if r['split'] == s})} for s in split_sets}
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'counts': counts,
              'original_images': len(rows), 'retained_images': len(records), 'excluded_images': len(rows) - len(records),
              'decisions': dict(Counter(r['decision'] for r in ledger)), 'folds': fold_counts,
              'duplicate_ROI_inputs': 0, 'cross_split_components': 0, 'cross_fold_components': 0,
              'policy': 'Quarantine all studies in a component with a conflicting binary label; otherwise keep one study by test > val > train priority then lexicographic study ID; deduplicate residual images within representative study.',
              'test_scope': 'Original official-test membership preserved for retained images; source test annotations were used in conflict quarantine, so this is a curated, previously inspected test subset, not a pristine holdout.',
              'labels_modified': False, 'source_files_modified': False,
              'source_hashes': summary['source_hashes'],
              'audit_sha256': checksum(audit / 'image_audit.json'),
              'manifest_sha256': checksum(data / 'manifest.json'), 'decisions_sha256': checksum(data / 'decisions.csv')}
    atomic_json(data / 'summary.json', result)
    atomic_json(data / 'READY.json', {'manifest_sha256': result['manifest_sha256'], 'checks_passed': True})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

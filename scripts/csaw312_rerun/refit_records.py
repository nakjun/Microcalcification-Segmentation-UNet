"""Verify refit stages and immutable evaluation records."""
from pathlib import Path
from common import ROOT, digest, read, sha, write

HERE = Path(__file__).resolve().parent


def stage_record(root, job, expected_signature=None, load=False):
    folder = root / 'jobs' / job['job_id']
    marker = folder / 'complete.json'
    r = read(marker)
    if (r.get('complete') is not True or r.get('smoke') is not False
            or r['job'] != job or Path(r['checkpoint']).name != r['checkpoint']
            or (expected_signature and r['signature'] != expected_signature)
            or sha(folder / r['checkpoint']) != r['checkpoint_sha256']):
        raise ValueError('Invalid stage completion ' + str(marker))
    if job['phase'] == 'refit' and (r['checkpoint'] != 'final.pt'
            or r['convergence_status'] != 'fixed_refit'
            or r['epochs'] != r['selected_epoch']):
        raise ValueError('Refit is not a fixed final checkpoint')
    if load:
        import torch
        ck = torch.load(folder / r['checkpoint'], map_location='cpu', weights_only=False)
        if ck['signature'] != r['signature'] or ck['epoch'] != r['selected_epoch']:
            raise ValueError('Embedded checkpoint binding mismatch')
    return dict(job_id=job['job_id'], signature=r['signature'],
                selected_epoch=r['selected_epoch'], complete_sha256=sha(marker),
                checkpoint_sha256=r['checkpoint_sha256'], checkpoint=r['checkpoint'])


def fixed_file(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('Immutable binding changed: ' + str(path))
    else:
        write(path, value)


def verify_internal(folder, expected_ids, provenance, lookup):
    done = read(folder / 'complete.json')
    binding = read(folder / 'binding.json')
    counts = read(folder / 'counts.json')
    manifest = read(folder / 'predictions.json')
    from metrics import summarize_segmentation
    if (not done.get('complete') or done['binding_sha256'] != sha(folder / 'binding.json')
            or done['scope'] != ('smoke_only' if 'smoke_internal' in folder.parts else folder.name)
            or binding['scope'] != ('smoke' if 'smoke_internal' in folder.parts else folder.name)
            or binding['exporter_sha256'] != sha(HERE / 'export_internal.py')
            or done['counts_sha256'] != sha(folder / 'counts.json')
            or done['predictions_manifest_sha256'] != sha(folder / 'predictions.json')
            or binding['provenance'] != provenance or binding['image_ids'] != expected_ids
            or done['image_count'] != len(expected_ids) or binding['threshold'] != .5
            or [r['image_id'] for r in counts['records']] != expected_ids
            or [r['image_id'] for r in done['predictions']] != expected_ids
            or manifest['predictions'] != done['predictions']
            or manifest['binding_sha256'] != done['binding_sha256']
            or counts['binding_sha256'] != done['binding_sha256']
            or counts['predictions_sha256'] != done['predictions_manifest_sha256']
            or counts['summary'] != summarize_segmentation(counts['records'])):
        raise ValueError('Internal export binding mismatch')
    for item, record in zip(done['predictions'], counts['records']):
        row = lookup[item['image_id']]
        if (Path(item['file']).name != item['file'] or sha(folder / item['file']) != item['sha256']
                or item['shape'] != [row['height'], row['width']]
                or item['image_sha256'] != row['image_sha256']
                or record['mask_sha256'] != row['mask_sha256']
                or record['group_id'] != row['split_group_id']
                or record['prediction_sha256'] != item['sha256']):
            raise ValueError('Internal prediction byte mismatch')
    return dict(complete_sha256=sha(folder / 'complete.json'), images=len(expected_ids))


def verify_external(folder, provenance, expected_count):
    done = read(folder / 'complete.json')
    prediction = read(folder / 'predictions_complete.json')
    protocol = read(folder / 'protocol.json')
    dataset = folder.parents[2].name
    inventory_path = ROOT / f'runs/external_v3/inputs/{dataset}.json'
    inventory = read(inventory_path)
    annotation = ROOT / f'runs/external_v3/annotations/{dataset}.json'
    evaluator_sources = {x: sha(ROOT / x) for x in [
        'scripts/external_v3/evaluate.py', 'scripts/external_current_models/evaluate.py',
        'scripts/csaw312_rerun/export_external.py', 'scripts/csaw312_rerun/export_internal.py',
        'scripts/calcification_gate_sam_common.py']}
    if (not done.get('complete') or done['pipeline'] != provenance
            or protocol['pipeline'] != provenance or protocol['smoke']
            or done['signature'] != digest(protocol)
            or prediction['signature'] != done['signature'] or prediction['smoke']
            or not prediction['complete'] or prediction['GT_read'] is not False
            or done['images'] != expected_count or len(prediction['image_ids']) != expected_count
            or prediction['image_ids'] != [row['image_id'] for row in inventory]
            or protocol['inventory_sha256'] != sha(inventory_path)
            or protocol['evaluator_source'] != evaluator_sources
            or done['annotation_sha256'] != sha(annotation)
            or done['dataset'] != dataset or done['selection_performed'] is not False
            or len(set(prediction['image_ids'])) != expected_count
            or len(prediction['files']) != 2 * expected_count
            or len(done['case_sha256']) != expected_count
            or done['prediction_complete_sha256'] != sha(folder / 'predictions_complete.json')):
        raise ValueError('External export binding mismatch')
    for name, value in {**prediction['files'], **done['case_sha256']}.items():
        path = Path(name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path) != value:
            raise ValueError('External export byte/path mismatch')
    for row in inventory:
        marker = read(folder / (row['image_id'] + '.json'))
        if (marker['image_id'] != row['image_id'] or marker['signature'] != done['signature']
                or marker['GT_read'] is not False or marker['source_dicom_sha256'] != row['dicom_sha256']
                or marker['prediction_sha256'] != sha(folder / (row['image_id'] + '.npz'))):
            raise ValueError('External per-image provenance mismatch')
    return dict(complete_sha256=sha(folder / 'complete.json'), images=expected_count)


def readonly_caches(rows, canonical_root):
    """Use exactly the frozen worker's audited cache inputs without writing them."""
    result=[]
    for row in rows:
        r=dict(row);path=canonical_root/'cache/images'/f'{r["image_id"]}.npy';marker=path.with_suffix('.json')
        m=read(marker)
        if m['source_sha256']!=r['image_sha256'] or sha(path)!=m['sha256'] or sha(r['source_mask'])!=r['mask_sha256']:
            raise ValueError('Training cache or mask hash mismatch: '+r['image_id'])
        r['native_cache_path']=str(path);r['role']='train';result.append(r)
    return result

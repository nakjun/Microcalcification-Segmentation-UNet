"""Immutable inputs and checkpoint lineage for non-ensemble INbreast evaluation."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch

SCRIPTS = Path(__file__).resolve().parents[1]
ROOT = Path('runs/external_comparative_inbreast')
PLAN = Path('configs/external_comparative_inbreast/protocol.json')
INBREAST = Path('/data/nas/Breast/Public_Mammo/InBreast/INbreast')
VARIANTS = ('C', 'L', 'H', 'LH')
FOLDS = (0, 1, 2)
GATE_ROOT = Path('runs/calcification_gate_retinanet_sam_clean_v1/frozen/gate')


def read(path):
    return json.loads(Path(path).read_text())


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    # Match the retained training runner's JSON digest for job source_signature validation.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def immutable(path, value):
    path = Path(path)
    if path.exists():
        assert read(path) == value, path
    else:
        atomic_json(path, value)


def verify(path, expected_signature=None):
    result = read(path)
    assert result['complete'], path
    if expected_signature is not None:
        assert result['signature'] == expected_signature, path
    for output, sha in result.get('output_files', {}).items():
        assert checksum(output) == sha, output
    return result


def atomic_npz(path, **payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as handle:
        np.savez_compressed(handle, **payload)
    temp.replace(path)


def model_specs():
    return (
        dict(key='proposed', label='PatchUNet', kind='study', root='runs/comparative_study/proposed_seed42'),
        dict(key='unext', label='UNeXt', kind='unext', root='runs/improvement_unext_deterministic/unext_v5_losses_deterministic_seed42'),
        dict(key='unet2d', label='U-Net 2D', kind='study', root='runs/comparative_study/unet2d_seed42'),
        dict(key='utnet', label='UTNet', kind='study', root='runs/comparative_study/utnet_seed42'),
    )


def _checkpoint_record(spec, variant, fold):
    root = Path(spec['root'])
    job = root / 'jobs' / variant / f'fold_{fold}'
    root_complete = read(root / 'complete.json')
    assert root_complete['complete']
    protocol_path = root / 'protocol.json'
    assert root_complete['output_files'][str(protocol_path)] == checksum(protocol_path)
    done = read(job / 'complete.json')
    assert done['complete'] and done['variant'] == variant and done['fold'] == fold
    assert done['epochs'] == 16 and not done['smoke']
    assert done['source_signature'] == digest(read(protocol_path))
    checkpoint = job / 'checkpoint.pt'
    assert done['output_files'][str(checkpoint)] == checksum(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['epoch'] == 16 and state['job'] == done['job']
    return dict(model=spec['key'], label=spec['label'], kind=spec['kind'], root=str(root), variant=variant,
                fold=fold, checkpoint=str(checkpoint), checkpoint_sha256=checksum(checkpoint),
                complete_path=str(job / 'complete.json'), complete_sha256=checksum(job / 'complete.json'),
                job_signature=done['signature'], model_config=state['config'])


def checkpoints():
    rows = []
    for spec in model_specs():
        for variant in VARIANTS:
            for fold in FOLDS:
                rows.append(_checkpoint_record(spec, variant, fold))
    assert len(rows) == 48
    return rows


def input_rows():
    rows = []
    for path in sorted((INBREAST / 'AllDICOMs').glob('*.dcm')):
        data = pydicom.dcmread(path, stop_before_pixels=True)
        spacing = data.get('PixelSpacing', data.get('ImagerPixelSpacing'))
        rows.append(dict(image_id=path.stem.split('_')[0], dataset='inbreast', dicom_path=str(path),
                         dicom_sha256=checksum(path), height=int(data.Rows), width=int(data.Columns),
                         spacing=None if spacing is None else list(map(float, spacing)), study_id=path.stem.split('_')[1]))
    assert len(rows) == 410 and len({r['image_id'] for r in rows}) == 410
    return rows


def prepare():
    """Write only image inventory and protocol; no XML is opened here."""
    rows = input_rows()
    immutable(ROOT / 'inputs/inbreast.json', rows)
    cps = checkpoints()
    sources = {str(PLAN): checksum(PLAN), str(ROOT / 'inputs/inbreast.json'): checksum(ROOT / 'inputs/inbreast.json')}
    for name in ('common.py', 'prepare.py', 'infer.py', 'verify_inference.py', 'evaluate.py', 'annotations.py', 'test_evaluation.py', 'test_inference.py'):
        path = Path(__file__).with_name(name)
        sources[str(path)] = checksum(path)
    for cp in cps:
        sources[cp['checkpoint']] = cp['checkpoint_sha256']
        sources[cp['complete_path']] = cp['complete_sha256']
    for path in (GATE_ROOT / 'classifier.pt', GATE_ROOT / 'thresholds.json', Path('scripts/calcification_gate_sam_common.py'),
                 Path('scripts/calcification_refinement_common.py'), Path('scripts/comparative_study/model.py'),
                 Path('scripts/improvement_unext_deterministic/model.py')):
        sources[str(path)] = checksum(path)
    protocol = dict(plan=read(PLAN), input_manifest=str(ROOT / 'inputs/inbreast.json'), checkpoints=cps,
                    gate=dict(checkpoint=str(GATE_ROOT / 'classifier.pt'), checkpoint_sha256=checksum(GATE_ROOT / 'classifier.pt'),
                              thresholds=str(GATE_ROOT / 'thresholds.json'), thresholds_sha256=checksum(GATE_ROOT / 'thresholds.json'),
                              threshold_logit=read(GATE_ROOT / 'thresholds.json')['primary']['threshold_logit']),
                    sources=sources, runtime=dict(torch=str(torch.__version__), numpy=np.__version__, opencv=cv2.__version__),
                    GT_read=False)
    immutable(ROOT / 'protocol.json', protocol)
    return dict(rows=rows, checkpoints=cps, protocol=protocol, signature=digest(protocol))


def context():
    protocol = read(ROOT / 'protocol.json')
    assert protocol['GT_read'] is False
    for path, sha in protocol['sources'].items():
        assert checksum(path) == sha, path
    rows = read(protocol['input_manifest'])
    assert len(rows) == 410
    return dict(rows=rows, checkpoints=protocol['checkpoints'], protocol=protocol, signature=digest(protocol))


def prediction_dir(checkpoint):
    return ROOT / 'predictions' / checkpoint['model'] / checkpoint['variant'] / f"fold_{checkpoint['fold']}"


def import_models():
    """Load the two ambiguous ``model.py`` files by absolute path and unique names."""
    comparative = str(SCRIPTS / 'comparative_study')
    if comparative in sys.path:
        sys.path.remove(comparative)
    sys.path.insert(0, comparative)
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    comparative_module = load('external_comparative_study_model', SCRIPTS / 'comparative_study' / 'model.py')
    unext_module = load('external_comparative_unext_model', SCRIPTS / 'improvement_unext_deterministic' / 'model.py')
    return comparative_module.StudyModel, unext_module.UNeXtAuthor2ch

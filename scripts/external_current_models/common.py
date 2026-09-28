"""Frozen current-model external protocol; image-only inputs, no annotation imports."""
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / 'scripts'
RUN = REPO / 'runs/external_current_models_20260927'
CONDITIONS = ('proposed_LH', 'unext_LH', 'unet2d_LH', 'utnet_LH', 'unet2d_H', 'deepmica', 'yolo')
DATASETS = {'inbreast': 410, 'cbis_test': 284}
sys.path.insert(0, str(SCRIPTS))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert read(path) == value, f'Preserve immutable artifact: {path}'
        return
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def selected(condition=None, fold=None):
    return [(c, f) for c in CONDITIONS for f in range(3)
            if (condition is None or c == condition) and (fold is None or f == fold)]


def checkpoint_record(condition, fold):
    original = load_module('current_original_inventory', SCRIPTS/'external_comparative_inbreast/common.py')
    if condition not in ('deepmica', 'yolo'):
        name, variant = condition.rsplit('_', 1)
        spec = next(s for s in original.model_specs() if s['key'] == name)
        record = original._checkpoint_record(spec, variant, fold)
        assert record['model_config']['patch_size'] == 256 and record['model_config']['tile_stride'] == 192
        files = {str(Path(record['checkpoint']).resolve()): record['checkpoint_sha256']}
        evidence = {str(Path(record['complete_path']).resolve()): record['complete_sha256']}
        return dict(condition=condition, fold=fold, epoch=16, seed=42, kind=record['kind'],
                    config=record['model_config'], checkpoints=files, evidence=evidence,
                    precision='BF16', inference_batch_size=record['model_config']['inference_batch_size'])
    base = REPO/'runs/additional_baselines_20260926'/condition
    done = read(base/'training_complete.json')
    assert done['folds'] == 3 and done['epochs'] == 16 and done['seed'] == 42 and done['selected_epoch'] == 16
    internal = read(base/'internal_complete.json')
    if condition == 'deepmica':
        paths = [base/f'fold_{fold}/last.pt']
    else:
        paths = [base/f'fold_{fold}/detector/epoch_016.pt', base/f'fold_{fold}/segmentor/epoch_016.pt', REPO/'runs/mammo_hypothesis/weights/sam_vit_b.pth']
    for path in paths:
        assert internal['checkpoint_sha256'][str(path)] == sha(path), path
    for path in paths[:1 if condition == 'deepmica' else 2]:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        assert checkpoint['epoch'] == 16
        if condition == 'deepmica':
            assert checkpoint['metadata']['channels'] == 1 and checkpoint['metadata']['scale'] == 255.
        else:
            assert checkpoint['metadata']['settings']['seed'] == 42
    return dict(condition=condition, fold=fold, epoch=16, seed=42, kind=condition,
                checkpoints={str(p):sha(p) for p in paths},
                evidence={str(p):sha(p) for p in (base/'training_complete.json', base/'internal_complete.json')},
                precision='FP32', inference_batch_size=1)


def require_disk(needed_bytes=0):
    free=shutil.disk_usage(REPO).free
    assert free > needed_bytes + 8*2**30, f'Insufficient disk: free={free/2**30:.1f}GiB; needed={needed_bytes/2**30:.1f}GiB plus 8GiB reserve'


def prepare():
    inventories = {}
    sources = {}
    for dataset, size in DATASETS.items():
        path = REPO/f'runs/external_v3/inputs/{dataset}.json'
        rows = read(path)
        assert len(rows) == size and len({r['image_id'] for r in rows}) == size
        assert all(r['dataset'] == dataset for r in rows)
        allowed = set('image_id dataset dicom_path dicom_sha256 height width spacing study_id'.split())
        assert all(set(row) <= allowed for row in rows), 'Inventory must contain image-only metadata'
        destination = RUN/f'inputs/{dataset}.json'
        write(destination, rows)
        inventories[dataset] = str(destination)
        sources[str(path)] = sha(path)
        sources[str(destination)] = sha(destination)
    native_bytes=sum(r['height']*r['width'] for path in inventories.values() for r in read(path))
    # Conservative uncompressed bound: uint8 native cache +21 packed masks.
    require_disk(int(native_bytes*(1+21/8)))
    checkpoints = [checkpoint_record(c, f) for c, f in selected()]
    for folder in ('scripts/external_current_models', 'scripts/comparative_study', 'scripts/improvement_unext_deterministic', 'DeepMiCa', 'YOLO-HLSAM', 'external/DeepMiCa/Step2_Segmentation/_03_Train', 'external/yolo-hlsam'):
        for path in (REPO/folder).rglob('*.py'):
            sources[str(path)] = sha(path)
    for name in ('calcification_refinement_common.py', 'calcification_gate_sam_common.py', 'external_comparative_inbreast/common.py', 'external_v3/evaluate.py', 'external_v3/common.py', 'external_v3/annotations.py'):
        path = SCRIPTS/name
        sources[str(path)] = sha(path)
    protocol = dict(version='current-model-ungated-external-v1', conditions=list(CONDITIONS), folds=[0,1,2],
                    seed=42, epoch=16, threshold=.5, patch_size=256, stride=192, gate=None,
                    inventory=inventories, checkpoints=checkpoints, sources=sources, GT_read=False,
                    preprocessing='unchanged native_dicom: slope/intercept; valid-pixel p0.5/p99.5 normalization; MONOCHROME1 inversion; native uint8',
                    policy='all images, ungated p>=0.5, no tuning, no ensemble, per-fold independent predictions',
                    yolo=dict(proposals='predicted boxes only per native256 tile', detector_size=640, confidence=.25, nms_iou=.45, max_det=100),
                    annotation_policy='retained external_v3 point/partial-contour and ROI-localization endpoints, opened only after selected full-cohort prediction verification',
                    storage='packed binary p>=0.5; no threshold sweep', estimated_cache_gib=native_bytes/2**30, uncompressed_cache_and_masks_gib=native_bytes*(1+21/8)/2**30, disk_reserve_gib=8, torch=str(torch.__version__))
    write(RUN/'protocol.json', protocol)
    return protocol


def context(condition=None, fold=None):
    protocol = read(RUN/'protocol.json')
    for path, expected in protocol['sources'].items():
        assert sha(path) == expected, f'Frozen source changed: {path}'
    checkpoints = [r for r in protocol['checkpoints'] if (condition is None or r['condition'] == condition) and (fold is None or r['fold'] == fold)]
    for row in checkpoints:
        for path, expected in {**row['checkpoints'], **row['evidence']}.items():
            assert sha(path) == expected, path
    return protocol, checkpoints, digest(protocol)


def prediction_dir(dataset, condition, fold, smoke=False):
    return RUN/dataset/'predictions'/condition/f'fold_{fold}'/('smoke' if smoke else 'full')


def verify(path, signature):
    value = read(path)
    assert value['complete'] and value['signature'] == signature, path
    for output, expected in value['output_files'].items():
        assert sha(output) == expected, output
    return value


def save_mask(path, mask):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, shape=np.asarray(mask.shape, np.int64), packed=np.packbits(mask.ravel()))
    temp.replace(path)


def read_mask(path, shape):
    with np.load(path, allow_pickle=False) as z:
        assert tuple(z['shape']) == tuple(shape)
        packed = z['packed']
        assert packed.dtype == np.uint8 and packed.ndim == 1 and len(packed) == (int(np.prod(shape))+7)//8
        return np.unpackbits(packed, count=int(np.prod(shape))).reshape(shape).astype(bool)

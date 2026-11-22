"""Export verified refit predictions and then score CSAW312 OOF/internal test.

Inference is image-only. The entire requested split's predictions are saved and
hashed before any GT mask is opened. --limit writes only to a smoke namespace.
"""
import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from common import ROOT, PLAN, context, read, sha, source_signature, write, disk_guard
import backbone
import specialized
from metrics import segmentation_counts, summarize_segmentation


def _job(plan, model, variant, seed, fold, stage):
    matches = [j for j in plan['jobs'] if j['model'] == model and j['variant'] == variant
               and j['seed'] == seed and j['fold'] == fold and j['phase'] == 'refit'
               and j['stage'] == stage]
    if len(matches) != 1:
        raise ValueError(f'Expected one refit job, found {len(matches)} for {model}/{variant}/{seed}/{fold}/{stage}')
    return matches[0]


def _checkpoint(job, out):
    directory = out / 'jobs' / job['job_id']
    marker = directory / 'complete.json'
    rec = read(marker)
    signature, _ = source_signature(job)
    if not rec.get('complete') or rec['job'] != job or rec['signature'] != signature:
        raise ValueError('Invalid or incompatible refit completion')
    checkpoint = rec['checkpoint']
    if Path(checkpoint).name != checkpoint:
        raise ValueError('Unsafe checkpoint filename')
    path = directory / checkpoint
    if sha(path) != rec['checkpoint_sha256']:
        raise ValueError('Refit checkpoint hash mismatch')
    data = torch.load(path, map_location='cpu', weights_only=False)
    if data['signature'] != signature or data['epoch'] != rec['selected_epoch']:
        raise ValueError('Checkpoint signature/epoch mismatch')
    provenance = dict(job_id=job['job_id'], signature=signature, selected_epoch=rec['selected_epoch'],
                      checkpoint=str(path), checkpoint_sha256=rec['checkpoint_sha256'],
                      complete_sha256=sha(marker))
    return data['model'], provenance


def _load_pipeline(model_name, variant, seed, fold, device='cuda:0'):
    """Load verified final refit, including YOLO's verified refit detector.

    Returns an opaque dictionary accepted by predict(). No GT inputs are read.
    """
    p, _, _, out = context()
    device = torch.device(device)
    stage = 'segmentor' if model_name == 'yolo' else 'segmentation'
    job = _job(p, model_name, variant, seed, fold, stage)
    weights, main_prov = _checkpoint(job, out)
    detector = None
    detector_prov = None
    if model_name == 'yolo':
        model, _, _ = specialized.build('segmentor', device, p['sam_checkpoint'])
        model.decoder.load_state_dict(weights, strict=True)
        det_job = _job(p, model_name, variant, seed, fold, 'detector')
        det_weights, detector_prov = _checkpoint(det_job, out)
        detector, _, _ = specialized.build('detector', device)
        detector.load_state_dict(det_weights, strict=True)
        detector.eval()
    elif model_name == 'deepmica':
        model, _, _ = specialized.build('deepmica', device)
        model.load_state_dict(weights, strict=True)
    else:
        model, _ = backbone.build(model_name, device)
        model.load_state_dict(weights, strict=True)
    model.eval()
    return dict(model_name=model_name, variant=variant, seed=seed, fold=fold,
                model=model, detector=detector, device=device,
                provenance=dict(refit=main_prov, detector_refit=detector_prov,
                                data_signature=p['data_signature'], protocol_sha256=sha(PLAN),
                                sam_sha256=sha(p['sam_checkpoint']) if model_name == 'yolo' else None))


def predict(pipeline, image):
    """Predict a native uint8 grayscale image, without access to annotations."""
    name = pipeline['model_name']
    if name == 'yolo':
        return specialized.predict('segmentor', pipeline['model'], image, pipeline['device'],
                                   detector=pipeline['detector'], confidence=.25)
    if name == 'deepmica':
        return specialized.predict('deepmica', pipeline['model'], image, pipeline['device'])
    return backbone.predict(pipeline['model'], image, pipeline['device'])


def load_pipeline(model_name, variant, seed, fold, device='cuda:0'):
    """Return image-only prediction callable and immutable refit provenance."""
    pipeline = _load_pipeline(model_name, variant, seed, fold, device)
    return lambda image: predict(pipeline, image), pipeline['provenance']


def _read_gray(path, shape):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or image.shape != tuple(shape):
        raise ValueError(f'Invalid uint8 native image: {path}')
    return image


def _save_prediction(path, prob, sparse):
    prob = np.asarray(prob)
    if prob.dtype != np.float32 or prob.ndim != 2 or not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)):
        raise ValueError('Invalid float32 native probability')
    payload = dict(shape=np.asarray(prob.shape, dtype=np.int64),
                   mask_p50=np.packbits((prob >= .5).reshape(-1)))
    if sparse:
        # Flat C-order index and exact FP32 values reconstruct every threshold >= .01.
        flat = prob.reshape(-1)
        idx = np.flatnonzero(flat >= np.float32(.01))
        payload['flat_indices_p01'] = idx.astype(np.uint64 if flat.size > 2**32 - 1 else np.uint32)
        payload['probabilities_p01'] = flat[idx].astype(np.float32, copy=False)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **payload)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def export_split(pipeline, split, limit=None):
    p, lookup, splits, out = context()
    fold = next(f for f in splits['folds'] if f['fold'] == pipeline['fold'])
    ids = fold['outer_val_ids'] if split == 'oof' else splits['test_ids'] if split == 'test' else None
    if ids is None:
        raise ValueError('Split must be oof or test')
    ids = sorted(ids)
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError('Empty or duplicated evaluation IDs')
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive')
        ids = ids[:limit]
    rows = [lookup[i] for i in ids]
    is_sparse = pipeline['model_name'] == 'unet2d' and pipeline['seed'] == 42 and pipeline['variant'] in ('C', 'L', 'H', 'LH')
    namespace = Path('evaluation/smoke_internal') / f'limit_{limit}' if limit is not None else Path('evaluation/internal')
    folder = out / namespace / f"{pipeline['model_name']}_{pipeline['variant']}" / f"seed_{pipeline['seed']}" / f"fold_{pipeline['fold']}" / split
    folder.mkdir(parents=True, exist_ok=True)
    binding = dict(schema_version=1, scope='smoke' if limit is not None else split, image_ids=ids,
                   provenance=pipeline['provenance'], threshold=.5, sparse_min_probability=.01 if is_sparse else None,
                   inference='native 256/192 overlap mean, image-only, ungated',
                   exporter_sha256=sha(Path(__file__).resolve()))
    binding_file = folder / 'binding.json'
    if binding_file.exists():
        if read(binding_file) != binding:
            raise ValueError('Existing export has different source or ID binding')
    else:
        write(binding_file, binding)
    marker = folder / 'complete.json'
    if marker.exists():
        rec = read(marker)
        if (not rec.get('complete') or rec['binding_sha256'] != sha(binding_file)
                or sha(folder/'counts.json') != rec['counts_sha256']
                or sha(folder/'predictions.json') != rec['predictions_manifest_sha256']
                or [item['image_id'] for item in rec['predictions']] != ids):
            raise ValueError('Completed export binding, manifest or counts mismatch')
        for item in rec['predictions']:
            if sha(folder/item['file']) != item['sha256']:
                raise ValueError('Completed export prediction mismatch')
        return rec
    predictions = []
    for index, row in enumerate(rows):
        disk_guard(p, out)
        ident = row['image_id']
        if Path(ident).name != ident:
            raise ValueError('Unsafe image ID')
        shape = (row['height'], row['width'])
        if sha(row['source_image']) != row['image_sha256']:
            raise ValueError(f'Image hash mismatch: {ident}')
        # A partial artifact has no trusted journal; recompute instead of reuse.
        image = _read_gray(row['source_image'], shape)
        prob = predict(pipeline, image)
        if prob.shape != shape:
            raise ValueError(f'Prediction shape mismatch: {ident}')
        filename = f'{ident}.npz'
        _save_prediction(folder / filename, prob, is_sparse)
        predictions.append(dict(image_id=ident, file=filename, sha256=sha(folder/filename),
                                image_sha256=row['image_sha256'], shape=list(shape)))
        print(f'{split} image-only prediction {index+1}/{len(rows)} {ident}', flush=True)
    # Persist and verify every prediction before consulting any GT.
    write(folder / 'predictions.json', dict(binding_sha256=sha(binding_file), predictions=predictions))
    for pred in predictions:
        if sha(folder/pred['file']) != pred['sha256']:
            raise ValueError('Prediction hash changed before scoring')
    counts = []
    for row, pred in zip(rows, predictions):
        if sha(row['source_mask']) != row['mask_sha256']:
            raise ValueError(f'Mask hash mismatch: {row["image_id"]}')
        truth = _read_gray(row['source_mask'], pred['shape']) > 0
        with np.load(folder/pred['file'], allow_pickle=False) as z:
            shape = tuple(z['shape'].tolist())
            if shape != tuple(pred['shape']):
                raise ValueError('Prediction shape metadata mismatch')
            predicted = np.unpackbits(z['mask_p50'], count=truth.size).reshape(shape).astype(np.float32)
        record = segmentation_counts(predicted, truth, .5)
        record.update(image_id=row['image_id'], group_id=row['split_group_id'],
                      mask_sha256=row['mask_sha256'], prediction_sha256=pred['sha256'])
        counts.append(record)
    result = dict(binding_sha256=sha(binding_file), predictions_sha256=sha(folder/'predictions.json'),
                  summary=summarize_segmentation(counts), records=counts)
    write(folder/'counts.json', result)
    complete = dict(complete=True, scope='smoke_only' if limit is not None else split,
                    binding_sha256=sha(binding_file), counts_sha256=sha(folder/'counts.json'),
                    predictions_manifest_sha256=sha(folder/'predictions.json'), predictions=predictions,
                    image_count=len(ids), time=time.time())
    write(marker, complete)
    return complete


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, choices=['proposed', 'unext', 'unet2d', 'utnet', 'deepmica', 'yolo'])
    ap.add_argument('--variant', required=True, choices=['C', 'L', 'H', 'LH', 'native'])
    ap.add_argument('--seed', required=True, type=int, choices=[42, 43, 44])
    ap.add_argument('--fold', required=True, type=int, choices=[0, 1, 2])
    ap.add_argument('--split', required=True, choices=['oof', 'test', 'all'])
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    torch.set_num_threads(4); cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    pipe = _load_pipeline(args.model, args.variant, args.seed, args.fold, args.device)
    for split in ('oof', 'test') if args.split == 'all' else (args.split,):
        export_split(pipe, split, args.limit)


if __name__ == '__main__':
    main()

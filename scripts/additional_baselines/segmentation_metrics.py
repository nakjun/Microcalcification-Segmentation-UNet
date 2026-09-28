"""Fixed-threshold, native-resolution segmentation evaluation (no tuning)."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def require_device(device):
    dev=torch.device(device)
    if dev.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
        index=dev.index if dev.index is not None else torch.cuda.current_device()
        if index>=torch.cuda.device_count():raise ValueError('CUDA device index out of range')
    return dev


def counts(probability,mask,threshold):
    probability=np.asarray(probability);mask=np.asarray(mask)
    if probability.ndim!=2 or probability.shape!=mask.shape:
        raise ValueError('Prediction and GT must have identical native 2D shapes')
    if not np.isfinite(probability).all() or probability.min()<0 or probability.max()>1:
        raise ValueError('Invalid probabilities')
    if not np.isfinite(mask).all() or not set(np.unique(mask)).issubset({0,1,255}):
        raise ValueError('Expected binary GT mask (0/1 or 0/255)')
    pred=probability>=threshold;gt=mask>0
    tp=int((pred & gt).sum());fp=int((pred & ~gt).sum());fn=int((~pred & gt).sum());tn=int((~pred & ~gt).sum())
    positive=bool(gt.any())
    return dict(TP=tp,FP=fp,FN=fn,TN=tn,positive=positive,
                dice=2*tp/(2*tp+fp+fn) if positive else None,
                iou=tp/(tp+fp+fn) if positive else None)


def summarize(records):
    pos=[r for r in records if r['positive']];neg=[r for r in records if not r['positive']]
    tp=sum(r['TP'] for r in pos);fn=sum(r['FN'] for r in pos)
    return dict(images=len(records),positive_images=len(pos),negative_images=len(neg),
                positive_macro_dice=float(np.mean([r['dice'] for r in pos])) if pos else None,
                positive_macro_iou=float(np.mean([r['iou'] for r in pos])) if pos else None,
                positive_pixel_recall=tp/(tp+fn) if tp+fn else None,
                negative_fp_pixels=sum(r['FP'] for r in neg),
                negative_fp_images=sum(r['FP']>0 for r in neg),
                pooled_counts={k:sum(r[k] for r in records) for k in ('TP','FP','FN','TN')})


def evaluate(rows,predict,load_mask,output,threshold,settings):
    if not np.isfinite(threshold) or not 0<threshold<1:raise ValueError('Threshold must be in (0,1)')
    if not rows:raise ValueError('Requested evaluation split is empty')
    out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    records=[]
    for row in rows:
        # The inference callback receives only an image path, never a mask/box.
        probability=predict(row['image'])
        record=counts(probability,load_mask(row['mask']),threshold)
        record.update(image=str(row['image']),patient_id=row['patient_id'],split=row['split'])
        records.append(record)
        with (out/'per_image.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
    result={'settings':settings,'threshold':threshold,'evaluation_scope':'native image; GT-free inference; positive-only macro Dice/IoU',
            'summary':summarize(records)}
    (out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (out/'evaluation_complete.json').write_text(json.dumps({'images':len(records),'metrics_sha256':hashlib.sha256((out/'metrics.json').read_bytes()).hexdigest()})+'\n')
    return result

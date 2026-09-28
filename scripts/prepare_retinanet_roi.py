"""Image-only high-resolution ROI cache; preserves original study split and GT."""
import argparse
import fcntl
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pydicom

ROOT = Path('runs/retinanet_roi_pair').resolve()
SOURCE = Path('runs/vindr_baselines/data')


def longest_run(nonconstant):
    edges = np.diff(np.r_[False, nonconstant, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if not len(starts):
        return 0, len(nonconstant)
    i = np.argmax(ends - starts)
    return int(starts[i]), int(ends[i])


def breast_roi(image):
    probe = np.where(image <= 40, 0, image)
    h, w = probe.shape
    x0, x1 = longest_run(probe[h//2-int(h*.4):h//2+int(h*.4)].std(axis=0) != 0)
    narrowed = probe[:, x0:x1]
    width = narrowed.shape[1]
    a, b = width//2-int(width*.4), width//2+int(width*.4)
    y0, y1 = longest_run(narrowed[:, a:max(a+1, b)].std(axis=1) != 0)
    return x0, y0, x1, y1


def transform_boxes(boxes, roi, inverse=False):
    b = np.asarray(boxes, dtype=float).reshape(-1, 4).copy()
    x0, y0, x1, y1 = roi
    scale = np.array([912/(x1-x0), 1520/(y1-y0)] * 2)
    offset = np.array([x0, y0] * 2)
    return b/scale + offset if inverse else (b-offset)*scale


def convert(row):
    cv2.setNumThreads(1)
    dest = ROOT/'data/images'/row['split']/(row['image_id']+'.png')
    side = dest.with_suffix('.json')
    if dest.exists() and side.exists():
        return json.loads(side.read_text())
    ds = pydicom.dcmread(row['source_path'])
    raw = ds.pixel_array
    assert raw.shape == (row['height'], row['width'])
    a = raw[5:-5, 5:-5].astype(np.float32)
    if ds.PhotometricInterpretation == 'MONOCHROME1':
        a = a.max()-a
    a -= a.min()
    if not a.max() > 0:
        raise ValueError(f"Empty image: {row['image_id']}")
    a = (a/a.max()*255).astype(np.uint8)
    x0, y0, x1, y1 = breast_roi(a)
    roi = [x0+5, y0+5, x1+5, y1+5]
    cropped = cv2.resize(a[y0:y1, x0:x1], (912, 1520), interpolation=cv2.INTER_AREA)
    b = transform_boxes(row['boxes'], roi)
    clipped = b.clip([0, 0, 0, 0], [912, 1520, 912, 1520])
    keep = (clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])
    out = dict(row, roi=roi, roi_cache_path=str(dest), roi_boxes=clipped[keep].tolist(),
               cropped_boxes=int(np.any(b != clipped, axis=1).sum()),
               removed_boxes=int((~keep).sum()))
    assert cv2.imwrite(str(dest), cropped, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    side.write_text(json.dumps(out))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--sample', type=int, default=0)
    args = p.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT/'prepare.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads(Path('configs/retinanet_roi_pair.json').read_text())
    assert hashlib.sha256((SOURCE/'split_manifest.csv').read_bytes()).hexdigest() == protocol['split_sha256']
    rows = [r for r in json.loads((SOURCE/'manifest.json').read_text()) if r['split'] in ('train', 'val')]
    assert len(rows) == 16000
    assert not ({r['study_id'] for r in rows if r['split']=='train'} & {r['study_id'] for r in rows if r['split']=='val'})
    if args.sample:
        rows = ([r for r in rows if r['boxes']][:args.sample] + [r for r in rows if not r['boxes']][:args.sample])
    for split in ['train', 'val']:
        (ROOT/'data/images'/split).mkdir(parents=True, exist_ok=True)
    result = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(convert, rows, chunksize=4), 1):
            result.append(row)
            if i % 100 == 0:
                print(f'ROI cache {i}/{len(rows)}', flush=True)
                (ROOT/'prepare_status.json').write_text(json.dumps({'done': i, 'total':len(rows)}))
    name = 'sample_manifest.json' if args.sample else 'manifest.json'
    (ROOT/'data'/name).write_text(json.dumps(result))
    summary = {s: {'images':sum(r['split']==s for r in result),
                   'cropped_boxes':sum(r['cropped_boxes'] for r in result if r['split']==s),
                   'removed_boxes':sum(r['removed_boxes'] for r in result if r['split']==s)} for s in ['train','val']}
    print(json.dumps(summary), flush=True)
    if not args.sample:
        (ROOT/'data/summary.json').write_text(json.dumps(summary, indent=2))
        (ROOT/'data/READY').write_text('complete\n')


if __name__ == '__main__':
    main()

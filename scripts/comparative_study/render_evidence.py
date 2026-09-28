"""Parent-created fixed full-image examples, not a model selection tool."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import context, ROOT, OUT, read, checksum
from calcification_gate_common import atomic_json
from pathlib import Path

c = context()
out = OUT
ids = ['105_0', '130_1', '025_1']
lookup = {r['image_id']: r for r in c['rows']}
fig, axes = plt.subplots(3, 4, figsize=(17, 15))
evidence = []
for row_index, ident in enumerate(ids):
    row = lookup[ident]
    fi = next(f['fold'] for f in c['folds'] if ident in f['validation_ids'])
    raw = np.load(row['native_cache_path'], mmap_mode='r')
    h, w = raw.shape
    step = max(1, max(h, w) // 1200)
    with np.load(row['target_path']) as z:
        gy, gx = z['y'], z['x']
    gt = np.zeros(raw.shape, bool)
    gt[gy, gx] = True
    case = read(ROOT / 'oof/cases' / f'{ident}.json')
    for col in range(4):
        ax = axes[row_index, col]
        ax.imshow(raw[::step, ::step], cmap='gray', vmin=0, vmax=255,
                  extent=(-.5, w-.5, h-.5, -.5))
        if col == 0:
            ax.scatter(gx, gy, s=.6, c='#00d76f', linewidths=0, rasterized=True)
            title = f'{ident}: image + GT ({len(gy)} pixels)'
        else:
            model = ['C','H','LH'][col-1]
            path = ROOT / 'jobs' / model / f'fold_{fi}' / 'predictions' / f'{ident}.npz'
            with np.load(path) as z:
                p = z['probability']
                flat = z['flat']
                fill = np.unpackbits(z['fill_imagenet'], count=raw.size).astype(bool)
                gate = bool(z['gate_imagenet'])
            pred = np.zeros(raw.size, bool)
            if gate:
                pred[flat[p >= np.float32(.9)]] = True
            pred = pred.reshape(raw.shape)
            for mask, color in [(pred & ~gt, '#ff3b30'), (gt & ~pred, '#ffb000'), (gt & pred, '#00d76f')]:
                y, x = np.nonzero(mask)
                ax.scatter(x, y, s=.6, c=color, linewidths=0, rasterized=True)
            metric = case['primary']['imagenet'][model]
            score = 'N/A' if not len(gy) else f"{metric['Dice']:.3f}"
            title = f"{model}: Dice {score}\nTP {metric['TP']} / FP {metric['FP']} / FN {metric['FN']}"
            evidence.append(dict(image_id=ident, fold=fi, model=model, prediction_sha256=checksum(path), metrics=metric))
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-.5, w-.5); ax.set_ylim(h-.5, -.5); ax.set_axis_off()
fig.suptitle('Study C/H/LH, full native field, gate and p>=0.9\nGreen: TP/GT | Red: FP | Orange: FN | Markers enlarged; not pixel-area display', fontsize=13)
fig.tight_layout(rect=(0, 0, 1, .955))
png = out / 'fixed_full_view.png'
fig.savefig(png, dpi=140)
plt.close(fig)
atomic_json(out / 'qualitative_evidence.json', dict(complete=True, signature=c['signature'],
            ids=ids, arm='imagenet', full_image=True, GT_crop=False,
            note='Fixed historical examples, not representative sampling; all mask coordinates shown as enlarged marks; grayscale downsampled for display only.',
            examples=evidence, png=str(png), png_sha256=checksum(png), source_sha256=checksum(__file__)))
print(png)

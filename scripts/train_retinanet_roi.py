"""Paired official RetinaNet encoders under one high-resolution validation protocol."""
import argparse
import fcntl
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import nms

sys.path.insert(0, str(Path('external/Mammo-CLIP/src/codebase').resolve()))
from Detectors.retinanet.detector_model import RetinaNet
from Detectors.retinanet.efficient_net import EfficientNet
from mammo_baselines.metrics import evaluate, save_json
from mammo_baselines.runtime import seed_all, checksum
from prepare_retinanet_roi import transform_boxes

ROOT = Path('runs/retinanet_roi_pair').resolve()
CONFIG = Path('configs/retinanet_roi_pair.json')


def signature():
    paths = [CONFIG, Path(__file__), Path('scripts/prepare_retinanet_roi.py'),
             Path('src/mammo_baselines/metrics.py'), Path('src/mammo_baselines/runtime.py')]
    paths += sorted(Path('external/Mammo-CLIP/src/codebase/Detectors').rglob('*.py'))
    return {str(p): checksum(p) for p in paths}


class FloatSigmoid(nn.Module):
    def forward(self, x):
        return x.float().sigmoid()


class Detector(RetinaNet):
    def __init__(self, init, seed=42):
        encoder = EfficientNet.from_name('efficientnet-b5', override_params={'image_size': None})
        state = torch.load(Path('runs/mammo_hypothesis/weights')/f'{init}_encoder.pt', map_location='cpu', weights_only=True)
        result = encoder.load_state_dict(state, strict=False)
        assert set(result.missing_keys) <= {'_fc.weight', '_fc.bias'} and not result.unexpected_keys
        encoder._fc = nn.Identity()
        encoder.source_layer_indexes = [26, 37]
        torch.manual_seed(seed)
        super().__init__(1, encoder, [176, 512, 2048], focal_alpha=.6, focal_gamma=2.)
        self.classificationModel.output_act = FloatSigmoid()
        self.anchor_cache = None

    def raw(self, images):
        _, c3, c4, c5 = self.efficientnet(images)
        features = self.fpn([c3, c4, c5])
        regression = torch.cat([self.regressionModel(f) for f in features], dim=1).float()
        classification = torch.cat([self.classificationModel(f) for f in features], dim=1).float()
        if self.anchor_cache is None:
            self.anchor_cache = self.anchors(images).to(images.device)
        anchors = self.anchor_cache
        assert anchors.shape[1] == regression.shape[1] == classification.shape[1]
        center = (anchors[0, :, :2]+anchors[0, :, 2:])/2
        keep = (center[:, 0] < 912) & (center[:, 1] < 1520)
        return classification[:, keep], regression[:, keep], anchors[:, keep]

    def training_loss(self, images, annotations):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=images.is_cuda):
            classification, regression, anchors = self.raw(images)
        positive = (annotations[:, :, 4] >= 0).any(dim=1)
        # Keep upstream positive-image loss unchanged; train negative images too.
        cls = classification.sum()*0
        reg = regression.sum()*0
        if positive.any():
            pc, pr = self.focalLoss(classification[positive], regression[positive], anchors, annotations[positive])
            cls = cls + pc.mean()*positive.sum()
            reg = reg + pr.mean()*positive.sum()
        if (~positive).any():
            q = classification[~positive].clamp(1e-4, 1-1e-4)
            cls = cls + (-(1-self.focalLoss.alpha)*q.pow(self.focalLoss.gamma)*torch.log1p(-q)).sum()
        return cls/len(images), reg/len(images)

    @torch.inference_mode()
    def predict(self, images):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            cls, reg, anchors = self.raw(images)
        boxes = self.regressBoxes(anchors, reg)
        boxes[:, :, [0, 2]] = boxes[:, :, [0, 2]].clamp(0, 912)
        boxes[:, :, [1, 3]] = boxes[:, :, [1, 3]].clamp(0, 1520)
        result = []
        for b, c in zip(boxes, cls[:, :, 0]):
            keep = (c >= .001) & torch.isfinite(b).all(1) & (b[:, 2]>b[:, 0]) & (b[:, 3]>b[:, 1])
            b, c = b[keep], c[keep]
            ids = nms(b, c, .5)[:100]
            result.append(torch.cat([b[ids], c[ids, None]], dim=1).cpu().numpy())
        return result


class Images(Dataset):
    def __init__(self, rows, augment=False):
        self.rows, self.augment = rows, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        im = cv2.imread(row['roi_cache_path'], 0)
        assert im is not None and im.shape == (1520, 912)
        b = np.asarray(row['roi_boxes'], np.float32).reshape(-1, 4).copy()
        if self.augment:
            scale = random.uniform(.9, 1.1)
            tx = (1-scale)*456+random.uniform(-.03, .03)*912
            ty = (1-scale)*760+random.uniform(-.03, .03)*1520
            im = cv2.warpAffine(im, np.array([[scale,0,tx],[0,scale,ty]], np.float32), (912,1520))
            b[:, [0,2]] = b[:, [0,2]]*scale+tx
            b[:, [1,3]] = b[:, [1,3]]*scale+ty
            b = b.clip([0,0,0,0], [912,1520,912,1520])
            b = b[(b[:,2]-b[:,0]>=2) & (b[:,3]-b[:,1]>=2)]
            if random.random() < .5:
                im = im[:, ::-1]
                b[:, [0,2]] = 912-b[:, [2,0]]
        padded = np.zeros((1536, 928), np.uint8)
        padded[:1520, :912] = im
        target = np.column_stack([b, np.zeros(len(b))]).astype(np.float32)
        return torch.from_numpy(padded)[None], torch.from_numpy(target), index


def collate(samples):
    images, boxes, indices = zip(*samples)
    annotations = torch.full((len(samples), max(1, max(len(b) for b in boxes)), 5), -1.)
    for i, b in enumerate(boxes):
        annotations[i, :len(b)] = b
    return torch.stack(images), annotations, indices


def normalize(images, device):
    return (images.to(device).float().div(255).expand(-1,3,-1,-1)-.3089279)/.25053555408335154


@torch.inference_mode()
def validate(model, rows, device, batch, out=None):
    model.eval()
    predictions = {}
    loader = DataLoader(Images(rows), batch_size=batch, num_workers=4, collate_fn=collate)
    for step, (images, _, indices) in enumerate(loader):
        for p, index in zip(model.predict(normalize(images, device)), indices):
            row = rows[index]
            p[:, :4] = transform_boxes(p[:, :4], row['roi'], inverse=True)
            predictions[row['image_id']] = p
        if out and step % 100 == 0:
            save_json(out/'status.json', {'state':'validation', 'images':len(predictions), 'total':len(rows)})
    metrics = evaluate(rows, predictions)
    return metrics, predictions


def atomic_save(value, path):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def train(args):
    protocol = json.loads(CONFIG.read_text())
    seed_all(args.seed)
    torch.cuda.set_device(args.device)
    device = f'cuda:{args.device}'
    out = ROOT/('smoke' if args.smoke else 'jobs')/f'{args.init}_seed{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out/'complete.json').exists() and not args.smoke:
        return
    if not args.smoke:
        assert (ROOT/'data/READY').exists()
    rows = json.loads((ROOT/'data'/('sample_manifest.json' if args.smoke else 'manifest.json')).read_text())
    tr = [r for r in rows if r['split']=='train']
    va = [r for r in rows if r['split']=='val']
    if args.smoke:
        tr = [r for r in rows if r['roi_boxes']][:2] + [r for r in rows if not r['roi_boxes']][:2]
        va = tr
    else:
        assert len(tr)==12800 and len(va)==3200
        tr = [r for r in tr for _ in range(protocol['positive_repeat'] if r['boxes'] else 1)]
    model = Detector(args.init, args.seed).to(device)
    head = {k:v for k,v in model.state_dict().items() if not k.startswith('efficientnet.')}
    head_hash = hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in head.values())).hexdigest()
    before_update = model.classificationModel.output.bias.detach().clone()
    sig = signature()
    save_json(out/'initialization.json', {'head_sha256':head_hash, 'encoder_sha256':checksum(Path('runs/mammo_hypothesis/weights')/f'{args.init}_encoder.pt'), 'source_hashes':sig})
    save_json(out/'protocol.json', protocol)
    opt = torch.optim.Adam(model.parameters(), lr=protocol['learning_rate'])
    batch = protocol['physical_batch']
    accum = protocol['effective_batch']//batch
    epochs = 1 if args.smoke else protocol['epochs']
    history, start_epoch, best = [], 0, -1.
    if (out/'last.pt').exists() and not args.smoke:
        ckpt = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
        assert ckpt['source_hashes'] == sig, 'Source changed; refusing mixed-protocol resume'
        model.load_state_dict(ckpt['model']); opt.load_state_dict(ckpt['optimizer'])
        history, start_epoch, best = ckpt['history'], ckpt['epoch'], ckpt['best']
        random.setstate(ckpt['random']); np.random.set_state(ckpt['numpy'])
        torch.set_rng_state(ckpt['torch']); torch.cuda.set_rng_state(ckpt['cuda'], device)
        del ckpt
    started = time.monotonic()
    prior_seconds = history[-1]['elapsed_seconds'] if history else 0.
    for epoch in range(start_epoch, epochs):
        # Epoch-specific worker/order seeds make epoch-boundary resume reproducible.
        loader = DataLoader(Images(tr, True), batch_size=batch, shuffle=True, num_workers=4,
                            collate_fn=collate, generator=torch.Generator().manual_seed(args.seed+epoch))
        model.train(); model.freeze_bn()
        opt.zero_grad(set_to_none=True)
        total, steps = 0., 0
        for step, (images, annotations, _) in enumerate(loader):
            cls, reg = model.training_loss(normalize(images, device), annotations.to(device))
            loss = cls+reg
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            window = min(accum, len(loader)-(step//accum)*accum)
            (loss/window).backward()
            if (step+1)%accum==0 or step+1==len(loader):
                norm = nn.utils.clip_grad_norm_(model.parameters(), .1)
                if not torch.isfinite(norm):
                    raise RuntimeError('Nonfinite gradient')
                opt.step(); opt.zero_grad(set_to_none=True)
            total += loss.item(); steps += 1
            if step%50==0 or step+1==len(loader):
                state = {'state':'training', 'epoch':epoch+1, 'epochs':epochs, 'step':step+1,
                         'steps':len(loader), 'loss':loss.item(), 'mean_loss':total/steps,
                         'elapsed_seconds':prior_seconds+time.monotonic()-started,
                         'gpu_peak_GiB':torch.cuda.max_memory_allocated()/2**30}
                save_json(out/'status.json', state)
                print(json.dumps(state), flush=True)
        metrics, predictions = validate(model, va, device, batch, out)
        history.append({'epoch':epoch+1, 'loss':total/steps, 'metrics':metrics,
                        'elapsed_seconds':prior_seconds+time.monotonic()-started})
        if metrics['AP50'] > best:
            best = metrics['AP50']
            atomic_save({'model':model.state_dict(), 'epoch':epoch+1, 'metrics':metrics}, out/'best.pt')
            atomic_save({'metrics':metrics,'predictions':predictions}, out/'val_internal.pt')
            save_json(out/'best.json', {'epoch':epoch+1, 'metrics':metrics})
        atomic_save({'model':model.state_dict(), 'optimizer':opt.state_dict(), 'epoch':epoch+1,
                     'best':best, 'history':history, 'source_hashes':sig,
                     'random':random.getstate(), 'numpy':np.random.get_state(), 'torch':torch.get_rng_state(),
                     'cuda':torch.cuda.get_rng_state(device)}, out/'last.pt')
        save_json(out/'history.json', history)
        print(json.dumps({'epoch':epoch+1,'metrics':metrics}), flush=True)
    selected = torch.load(out/'val_internal.pt', map_location='cpu', weights_only=False)
    if args.smoke:
        assert not torch.equal(before_update, model.classificationModel.output.bias), 'Optimizer did not update weights'
    filtered = {k:v[v[:,4]>=.05] for k,v in selected['predictions'].items()}
    native_floor = evaluate(va, filtered)
    summary = {'state':'complete', 'seed':args.seed, 'epochs':epochs, 'split':'val', 'images':len(va),
               'best':json.loads((out/'best.json').read_text()), 'metrics_score005':native_floor,
               'checkpoint_sha256':checksum(out/'best.pt')}
    save_json(out/'complete.json', summary)
    save_json(out/'status.json', {'state':'complete','epochs':epochs})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--init', choices=['imagenet','mammo'], required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    try:
        train(args)
    except Exception as exc:
        save_json(ROOT/('smoke' if args.smoke else 'jobs')/f'{args.init}_seed{args.seed}'/'failure.json', {'error':repr(exc)})
        raise

"""Paired, three-fold CSAW312 YOLO-HLSAM retraining with fixed detector prompts."""
import argparse
import fcntl
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import export_internal
import specialized
import worker
from common import PLAN, ROOT, context, digest, lr_for_epoch, read, sha, source_signature, write
from refit_records import fixed_file, stage_record, verify_internal
from refit_records import readonly_caches
from sampling import PatchSampler

OUTPUT = ROOT / 'runs/CSAW312_FP_Rerun_20261119/yolo'
ARMS = ('control', 'candidate')
FRACTION = .01
NATIVE_PREDICT = export_internal.predict
NATIVE_DETECT = specialized.detect


def hard_background_loss(logits, target):
    background = logits[target == 0]
    if background.numel() == 0:
        return logits.sum() * 0
    count = math.ceil(FRACTION * background.numel())
    return F.softplus(background.topk(count, sorted=False).values).mean()


class CandidateFunctional:
    """Apply the extra term only at the original segmentor BCE call."""
    @staticmethod
    def binary_cross_entropy_with_logits(logits, target):
        return F.binary_cross_entropy_with_logits(logits, target) + hard_background_loss(logits, target)

    def __getattr__(self, name):
        return getattr(F, name)


class RedrawSampler(PatchSampler):
    def epoch(self, rng, specialist=False, draws=4096):
        if not specialist:
            raise ValueError('Redraw is restricted to YOLO detector stages')
        self.reset_exposure()
        self.accepted = self.rejected = self.accepted_positive = 0
        for _ in range(draws * 10):
            image, mask = self.sample(rng, specialist=True)
            tensor, _ = specialized._detector_sample(image, specialized.boxes_from_mask(mask), 'cpu')
            if not torch.isfinite(tensor).all():
                raise FloatingPointError('Nonfinite post-letterbox detector input')
            keep = bool(torch.count_nonzero(tensor))
            if keep != bool(np.any(image)):
                raise ValueError('Raw/post-letterbox nonzero mismatch')
            if not keep:
                self.rejected += 1
                continue
            self.accepted += 1
            self.accepted_positive += bool(mask.any())
            yield image, mask
            if self.accepted == draws:
                return
        raise RuntimeError('Detector redraw budget exhausted')

    def exposure(self):
        return dict(patches=self.accepted, accepted_positive=self.accepted_positive,
                    accepted_negative=self.accepted-self.accepted_positive, rejected=self.rejected,
                    proposals=self.accepted+self.rejected, proposal_exposure=super().exposure())


def train_smoke(plan, lookup, splits, canonical, device):
    """Real CUDA updates using only inner-training images, before any full stage."""
    # Check the top-k pixel selection and gradient against explicit scalar softplus.
    test = torch.tensor([-4., 0., 2., 10.], requires_grad=True)
    loss = hard_background_loss(test, torch.tensor([0, 0, 0, 1]))
    loss.backward()
    assert torch.allclose(loss, F.softplus(test.detach()[2]))
    assert test.grad.nonzero().flatten().tolist() == [2]
    reports = []
    for fold in splits['folds']:
        ids = fold['inner_train_ids']
        selected = [next(i for i in ids if lookup[i]['positive'] == positive) for positive in [True, False]]
        rows = readonly_caches([lookup[i] for i in selected], canonical)
        centers = worker.initial_centers(rows, plan)
        for stage, arm in [('detector','control'), ('segmentor','control'), ('segmentor','candidate')]:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            sampler = (RedrawSampler if stage == 'detector' else PatchSampler)(rows, centers)
            model, optimizer, criterion = specialized.build(stage, device, plan['sam_checkpoint'])
            for group in optimizer.param_groups:
                group['lr'] = lr_for_epoch(1e-4, 1, plan)
            patches = sampler.epoch(np.random.default_rng(1051), specialist=True,
                                    draws=3 if stage == 'detector' else 6)
            func = CandidateFunctional() if arm == 'candidate' else F
            with patch.object(specialized, 'F', func):
                stats = specialized.train_one_epoch(stage, model, optimizer, criterion, patches, device)
            if stats['updates'] < 1 or not math.isfinite(stats['training_loss']):
                raise ValueError('Invalid smoke loss/update count')
            if not all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):
                raise FloatingPointError('Nonfinite smoke weights')
            reports.append(dict(fold=fold['fold'], arm=arm, stage=stage, stats=stats,
                                exposure=sampler.exposure(), training_image_ids=selected))
            del model, optimizer, criterion
            torch.cuda.empty_cache()
    return reports


def stage_link(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        if dest.resolve(strict=True) != src.resolve(strict=True):
            raise ValueError(f'Changed checkpoint source {dest}')
    else:
        dest.symlink_to(src.resolve(strict=True), target_is_directory=True)


def stages(plan, lookup, splits, canonical, sources, signature, fold, device):
    jobs = [j for j in plan['jobs'] if j['model']=='yolo' and j['fold']==fold and j['seed']==42]
    if len(jobs)!=4:
        raise ValueError('Incomplete YOLO stage plan')
    control = OUTPUT / 'control'
    for arm in ARMS:
        out = OUTPUT / arm
        fixed_file(out/'preflight.json', dict(ready=True, protocol_sha256=sha(PLAN), source_sha256=sources))
        for job in jobs:
            job_id = job['job_id']
            stage = job['stage']
            if arm=='candidate' and stage=='detector':
                stage_link(control/'jobs'/job_id, out/'jobs'/job_id)
                continue
            deps = {i: sha(out/'jobs'/i/'complete.json') for i in job['depends_on']}
            job_signature = digest(dict(experiment=signature, job=job, arm=arm, dependencies=deps))
            sampler = RedrawSampler if stage=='detector' else PatchSampler
            functional = CandidateFunctional() if arm=='candidate' else F
            write(OUTPUT/'status.json', dict(phase='training', arm=arm, job=job_id, time=time.time()))
            with patch.object(worker,'context',lambda:(plan, lookup, splits, out)), \
                    patch.object(worker,'cache_rows',lambda rows, _:readonly_caches(rows, canonical)), \
                    patch.object(worker,'source_signature',lambda job,smoke=False:(job_signature,sources)), \
                    patch.object(worker,'PatchSampler',sampler), patch.object(specialized,'F',functional):
                worker.run(SimpleNamespace(job_index=plan['jobs'].index(job), smoke=False, device=device))
            stage_record(out, job, job_signature, load=True)


def instrumented_predict(pipeline, image, image_id, directory):
    """Observe the actual native tiled prompts and count their union in image space."""
    h, w = image.shape
    coords = iter((y,x) for y in specialized._deep().positions(max(256,h),256,192)
                  for x in specialized._deep().positions(max(256,w),256,192))
    union = np.zeros((h,w), dtype=bool)
    count = 0
    def observe(*args, **kwargs):
        nonlocal count
        y,x = next(coords)
        boxes,scores = NATIVE_DETECT(*args, **kwargs)
        for x1,y1,x2,y2 in boxes:
            union[y+math.floor(y1):y+math.ceil(y2), x+math.floor(x1):x+math.ceil(x2)] = True
        count += len(boxes)
        return boxes,scores
    with patch.object(specialized,'detect',observe):
        prob = NATIVE_PREDICT(pipeline,image)
    assert next(coords,None) is None
    mask = prob >= .5
    write(directory/(image_id+'.json'), dict(image_id=image_id, tile_prompts=count,
          box_union_pixels=int(union.sum()), mask_pixels=int(mask.sum()),
          mask_inside_union=int(np.count_nonzero(mask & union)), image_pixels=int(mask.size)))
    return prob


def evaluate(plan, lookup, splits, fold, arm, device):
    out = OUTPUT / arm
    jobs = [j for j in plan['jobs'] if j['model']=='yolo' and j['fold']==fold
            and j['seed']==42 and j['phase']=='refit']
    records = {j['job_id']:stage_record(out,j,load=True) for j in jobs}
    def signature(job,smoke=False):
        return records[job['job_id']]['signature'], {}
    entries = []
    with patch.object(export_internal,'context',lambda:(plan,lookup,splits,out)), \
            patch.object(export_internal,'source_signature',signature):
        pipeline = export_internal._load_pipeline('yolo','native',42,fold,device)
        for split in ('oof','test'):
            ids = sorted(next(f['outer_val_ids'] for f in splits['folds'] if f['fold']==fold)
                         if split=='oof' else splits['test_ids'])
            folder = out/'evaluation/internal/yolo_native/seed_42'/f'fold_{fold}'/split
            diagnostic = folder/'box_diagnostics'
            current = iter(ids)
            def predict(pipe,image):
                return instrumented_predict(pipe,image,next(current),diagnostic)
            write(OUTPUT/'status.json', dict(phase='evaluation', arm=arm, fold=fold, split=split, time=time.time()))
            with patch.object(export_internal, 'predict', predict if split=='oof' else NATIVE_PREDICT):
                export_internal.export_split(pipeline,split)
            verify_internal(folder,ids,pipeline['provenance'],lookup)
            counts = read(folder/'counts.json')
            if split=='oof':
                for row in counts['records']:
                    obs = read(diagnostic/(row['image_id']+'.json'))
                    assert row['predicted_pixels']==obs['mask_pixels']
            entries.append(dict(arm=arm, fold=fold, split=split, counts=str(folder/'counts.json'),
                                counts_sha256=sha(folder/'counts.json'), complete_sha256=sha(folder/'complete.json'),
                                detector_sha256=pipeline['provenance']['detector_refit']['checkpoint_sha256']))
    return entries


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--smoke-only',action='store_true')
    args=ap.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run(args)
        except Exception as exc:
            prior = read(OUTPUT/'status.json') if (OUTPUT/'status.json').exists() else {}
            write(OUTPUT/'failure.json',dict(error=repr(exc), phase=prior, time=time.time()))
            raise


def run(args):
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.cuda.set_device(args.device)
    plan,lookup,splits,canonical=context()
    _,sources=source_signature(next(j for j in plan['jobs'] if j['model']=='yolo' and not j['depends_on']))
    if sources != read(canonical/'preflight.json')['source_sha256']:
        raise ValueError('Source manifest differs from the original paper protocol')
    sources=dict(sources)
    for name in ['fp_yolo.py','refit_records.py','export_internal.py']:
        p=Path(__file__).parent/name
        sources[str(p.relative_to(ROOT))]=sha(p)
    spec=dict(sources=sources, seed=42, folds=[0,1,2], detector='post-letterbox zero redraw',
              decoder_candidate='BCE+Dice+mean softplus(top 1% target-negative logits)',
              decoder_control='BCE+Dice', penalty_weight=1, background_tail_fraction=FRACTION,
              sam=plan['sam_checkpoint'], training=plan['training'], threshold=.5, detector_confidence=.25,
              data_signature=plan['data_signature'])
    fixed_file(OUTPUT/'spec.json',spec)
    signature=digest(spec)
    smoke=OUTPUT/'smoke.json'
    if not smoke.exists():
        write(OUTPUT/'status.json',dict(phase='smoke',time=time.time()))
        write(smoke,dict(signature=signature, results=train_smoke(plan,lookup,splits,canonical,args.device)))
    if read(smoke)['signature']!=signature:
        raise ValueError('Changed experiment after smoke')
    if args.smoke_only:
        return
    entries=[]
    for fold in range(3):
        stages(plan,lookup,splits,canonical,sources,signature,fold,args.device)
        for arm in ARMS:
            entries.extend(evaluate(plan,lookup,splits,fold,arm,args.device))
            paths = {name:{} for name in ARMS}
            for item in entries:
                paths[item['arm']].setdefault(str(item['fold']), {})[item['split']] = item['counts']
            write(OUTPUT/'evaluation_bindings.json',entries)
            write(OUTPUT/'evaluations.json',paths)
    write(OUTPUT/'complete.json',dict(complete=True,signature=signature,
                                      evaluations_sha256=sha(OUTPUT/'evaluations.json')))
    write(OUTPUT/'status.json',dict(phase='complete',time=time.time()))


if __name__=='__main__':
    main()

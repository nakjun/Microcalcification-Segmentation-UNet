"""Training-only current-model mining with frozen stage checkpoints and verified pools."""
import time
from pathlib import Path
import numpy as np
import torch
from common import spec, job_signature, verify, read, checksum, immutable
from calcification_refinement_common import tiled_probability, target_patch
from calcification_refinement_v2_common import mine_negative_centers
from calcification_gate_sam_common import atomic_npz, disk_guard
import train_calcification_refinement_v2 as v2


def stage_path(job, epoch):
    return job / 'mining' / f'after_{epoch:02d}'


def binding(path, marker):
    return dict(epoch=marker['epoch'], complete_path=str(path/'complete.json'),
                complete_sha256=checksum(path/'complete.json'),
                checkpoint_sha256=marker['checkpoint_sha256'],
                pool_path=marker['pool_path'], pool_sha256=checksum(marker['pool_path']))


def load_current(c, job, fold, epoch):
    stages=[e for e in (3,7,11) if e<epoch]
    if not stages:return None,None
    stage=max(stages);path=stage_path(job,stage);m=verify(path/'complete.json',job_signature(c,fold))
    assert m['epoch']==stage and m['fold']==fold and m['smoke'] is False
    assert m['image_ids']==spec(c,fold)['train_ids']
    with np.load(m['pool_path']) as z:
        centers={k:z[k] for k in z.files if k.endswith('_hard')}
    assert set(centers)=={ident+'_hard' for ident in m['image_ids']}
    return centers,binding(path,m)


def refresh(c, job, fold, epoch, model, device, smoke=False):
    sig=job_signature(c,fold,smoke);path=stage_path(job,epoch)
    ids=spec(c,fold)['train_ids'];ids=ids[:1] if smoke else ids
    assert smoke or epoch in (3,7,11)
    if (path/'complete.json').exists():
        m=verify(path/'complete.json',sig)
        assert m['epoch']==epoch and m['image_ids']==ids and m['smoke']==smoke
        assert m['model_state_sha256']==v2.state_digest(model)
        return binding(path,m)
    cfg=c['learning_config'];lookup={r['image_id']:r for r in c['rows']}
    assert set(ids)<=set(spec(c,fold)['train_ids']) and not set(ids)&set(spec(c,fold)['validation_ids'])
    before=v2.state_digest(model);training_mode=model.training
    cpu_rng=torch.get_rng_state().clone();gpu_rng=[s.clone() for s in torch.cuda.get_rng_state_all()]
    checkpoint=path/'source.pt';source_meta=path/'source.json'
    if source_meta.exists():
        src=verify(source_meta,sig)
        assert src['epoch']==epoch and src['fold']==fold and src['model_state_sha256']==before
    else:
        v2.atomic_torch(checkpoint,dict(signature=sig,fold=fold,epoch=epoch,config=cfg,model=model.state_dict(),model_state_sha256=before))
        immutable(source_meta,dict(complete=True,signature=sig,epoch=epoch,fold=fold,model_state_sha256=before,output_files={str(checkpoint):checksum(checkpoint)}))
    cp_sha=checksum(checkpoint);outputs={str(checkpoint):cp_sha,str(source_meta):checksum(source_meta)}
    pools={};started=time.monotonic();model.eval()
    for ident in ids:
        row=lookup[ident];assert row['role']=='train'
        case=path/'cases'/f'{ident}.json';arrays=case.with_suffix('.npz')
        if case.exists():
            meta=verify(case,sig)
            assert meta['image_id']==ident and meta['checkpoint_sha256']==cp_sha
            assert meta['native_sha256']==row['native_cache_sha256'] and meta['target_sha256']==row['target_sha256']
            with np.load(arrays) as z:points,scores=z['centers'],z['scores']
        else:
            disk_guard(dict(cfg,root=str(job)),extra=row['height']*row['width']*16)
            assert checksum(row['native_cache_path'])==row['native_cache_sha256'] and checksum(row['target_path'])==row['target_sha256']
            raw=np.load(row['native_cache_path'],mmap_mode='r')
            with np.load(row['target_path']) as z:gy,gx=z['y'],z['x']
            tick=time.monotonic()
            with torch.no_grad():prob,tiles=tiled_probability(model,raw,cfg,device)
            assert prob.shape==raw.shape
            points,scores=mine_negative_centers(row,prob,gy,gx,256,.3,128,4096)
            for cy,cx in points:
                y=int(np.clip(cy-128,0,max(0,raw.shape[0]-256)));x=int(np.clip(cx-128,0,max(0,raw.shape[1]-256)))
                assert not target_patch(gy,gx,y,x,256).any()
            atomic_npz(arrays,centers=points,scores=scores)
            import hashlib
            immutable(case,dict(complete=True,signature=sig,image_id=ident,fold=fold,epoch=epoch,
                      training_only=True,checkpoint_sha256=cp_sha,native_sha256=row['native_cache_sha256'],
                      target_sha256=row['target_sha256'],probability_sha256=hashlib.sha256(prob.tobytes()).hexdigest(),
                      centers=len(points),tiles=tiles,elapsed_seconds=time.monotonic()-tick,
                      output_files={str(arrays):checksum(arrays)}))
        pools[ident+'_hard']=points;pools[ident+'_scores']=scores
        outputs[str(case)]=checksum(case);outputs[str(arrays)]=checksum(arrays)
        print(f'current mining fold{fold} after{epoch} {ident} centers={len(points)}',flush=True)
    model.train(training_mode)
    assert v2.state_digest(model)==before and torch.equal(cpu_rng,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(gpu_rng,torch.cuda.get_rng_state_all()))
    pool_path=path/'centers.npz';atomic_npz(pool_path,**pools);outputs[str(pool_path)]=checksum(pool_path)
    immutable(path/'complete.json',dict(complete=True,signature=sig,fold=fold,epoch=epoch,smoke=smoke,
              image_ids=ids,training_only=True,model_state_sha256=before,checkpoint_sha256=cp_sha,
              pool_path=str(pool_path),elapsed_seconds=time.monotonic()-started,
              model_and_rng_unchanged=True,output_files=outputs))
    return binding(path,read(path/'complete.json'))

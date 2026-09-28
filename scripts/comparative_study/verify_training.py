"""Parent-owned sampler, checkpoint, warmup and training-only mining verification."""
import argparse,hashlib
from pathlib import Path
import numpy as np
import torch
from common import ROOT,OUT,VARIANTS,context,spec,initial_pool,read,verify,checksum,job_signature
from model import StudyModel
from miner import load_current
from calcification_refinement_v5_common import V5Sampler
from calcification_refinement_v2_common import RecallAwareSampler
from calcification_gate_common import atomic_json
import train_calcification_refinement_v2 as v2

def cpu(c):
    f=spec(c,0);lookup={r['image_id']:r for r in c['rows']};rows=[lookup[i] for i in f['train_ids']];cfg=c['learning_config']
    reference=RecallAwareSampler(rows,cfg,initial_pool(c,0));sam=V5Sampler(rows,cfg,initial_pool(c,0));a=np.random.default_rng(1051);b=np.random.default_rng(1051);n=0
    for _ in range(32):
        x,y,mode=reference.sample(a);xx,yy,mm,w,v=sam.sample(b)
        assert np.array_equal(x,xx) and np.array_equal(y,yy) and mode==mm and a.bit_generator.state==b.bit_generator.state
        assert w.dtype==v.dtype==np.float32 and (w>=0).all() and ((v==0)|(v==1)).all() and not np.any(w[y==0]);n+=1
    return dict(actual_sampler_examples=n,original_input_target_mode_rng_exact=True,performance_evaluation=False)

def check_mining(c,job,fi):
    f=spec(c,fi);lookup={r['image_id']:r for r in c['rows']};stages=[]
    for epoch in (3,7,11):
        path=job/'mining'/f'after_{epoch:02d}';done=verify(path/'complete.json',job_signature(c,fi))
        assert done['image_ids']==f['train_ids'] and not set(done['image_ids'])&set(f['validation_ids']) and done['training_only'] and not done['smoke'] and done['model_and_rng_unchanged']
        state=torch.load(path/'source.pt',map_location='cpu',weights_only=False);model=StudyModel(c['learning_config']);model.load_state_dict(state['model'],strict=True)
        assert v2.state_digest(model)==done['model_state_sha256'] and checksum(path/'source.pt')==done['checkpoint_sha256']
        with np.load(done['pool_path']) as z:pool={k:z[k] for k in z.files}
        assert set(pool)=={i+s for i in f['train_ids'] for s in ('_hard','_scores')}
        number=0
        for ident in f['train_ids']:
            row=lookup[ident];m=verify(path/'cases'/f'{ident}.json',job_signature(c,fi))
            assert m['training_only'] and m['checkpoint_sha256']==done['checkpoint_sha256'] and m['target_sha256']==row['target_sha256'] and m['native_sha256']==row['native_cache_sha256']
            with np.load(path/'cases'/f'{ident}.npz') as z:
                points,scores=z['centers'],z['scores'];assert np.array_equal(points,pool[ident+'_hard']) and np.array_equal(scores,pool[ident+'_scores'])
            assert points.shape==(len(scores),2) and len(points)<=4096 and m['centers']==len(points)
            ranking=[(-float(s),int(y),int(x)) for (y,x),s in zip(points,scores)];assert ranking==sorted(ranking)
            with np.load(row['target_path']) as z:ys,xs=z['y'],z['x']
            for (cy,cx),score in zip(points,scores):
                assert np.float32(.3)<=score<=1 and 0<=cy<row['height'] and 0<=cx<row['width']
                y=max(0,min(int(cy)-128,max(0,row['height']-256)));x=max(0,min(int(cx)-128,max(0,row['width']-256)))
                assert not np.any((ys>=y)&(ys<y+256)&(xs>=x)&(xs<x+256))
            number+=1
        stages.append(dict(epoch=epoch,training_images=number,source_sha256=done['checkpoint_sha256'],complete_sha256=checksum(path/'complete.json')))
    return stages

def lineage(smoke=False):
    jobs=[];initial=None;warmup={};warmup_models={};params=None
    for variant in VARIANTS:
        c=context(variant);cfg=c['learning_config'];torch.manual_seed(42);model=StudyModel(cfg);newinit=v2.state_digest(model)
        if initial is None:initial=newinit;params=sum(p.numel() for p in model.parameters())
        assert initial==newinit
        for f in ([spec(c,0)] if smoke else c['folds']):
            fi=f['fold'];job=ROOT/('smoke' if smoke else 'jobs')/variant/f'fold_{fi}';sig=job_signature(c,fi,smoke);done=verify(job/'complete.json',sig);state=torch.load(job/'checkpoint.pt',map_location='cpu',weights_only=False)
            epochs,steps=(1,2) if smoke else (16,2048)
            assert done['epochs']==state['epoch']==epochs and state['signature']==sig and state['job']==done['job'] and state['job']['initial_model_sha256']==initial
            assert state['job']['train_ids']==f['train_ids'] and state['job']['validation_ids']==f['validation_ids']
            restored=StudyModel(cfg);restored.load_state_dict(state['model'],strict=True);assert all(torch.isfinite(v).all() for v in state['model'].values())
            assert {int(s['step']) for s in state['optimizer']['state'].values()}=={steps}
            assert v2.state_digest(restored)!=initial
            for epoch in range(1,epochs+1):
                r=read(job/'epochs'/f'{epoch:02d}.json');assert r['signature']==sig and r['optimizer_steps']==(2 if smoke else 128) and r['patches']==(32 if smoke else 2048)
                assert all(np.isfinite(v) for v in r['mean_stats'].values())
                assert sum(r['sampling_modes'].values())==r['patches']
                if not smoke:assert load_current(c,job,fi,epoch)[1]==r['active_mining'] and load_current(c,job,fi,epoch+1)[1]==r['available_mining']
                if epoch<=3:
                    key=(fi,epoch);pair=(r['input_target_sha256'],r['geometry_sha256'])
                    if key in warmup:assert warmup[key]==pair
                    else:warmup[key]=pair
                if smoke:
                    rows=[next(x for x in c['rows'] if x['image_id']==i) for i in f['train_ids']];sam=V5Sampler(rows,cfg,initial_pool(c,fi));rng=np.random.default_rng(42+epoch*1009);h=hashlib.sha256();g=hashlib.sha256()
                    for _ in range(2):
                        ex=[sam.sample(rng) for _ in range(16)]
                        for k in (0,1):h.update(np.stack([v[k] for v in ex]).tobytes())
                        for k in (3,4):g.update(np.stack([v[k] for v in ex]).tobytes())
                    assert h.hexdigest()==r['input_target_sha256'] and g.hexdigest()==r['geometry_sha256'] and r['smoke_loss_epoch']==6
            stages=[] if smoke else check_mining(c,job,fi)
            if not smoke:
                stage=read(job/'mining/after_03/complete.json');value=stage['model_state_sha256']
                if fi in warmup_models:assert warmup_models[fi]==value
                else:warmup_models[fi]=value
            jobs.append(dict(variant=variant,fold=fi,epochs=epochs,optimizer_steps=steps,checkpoint_sha256=checksum(job/'checkpoint.pt'),mining_stages=stages))
    return c,dict(jobs=jobs,initial_model_sha256=initial,parameters=params,warmup_stream_geometry_exact=True,warmup_model_state_exact=not smoke,strict_reload_finite=True,performance_evaluation=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['cpu','smoke','lineage'],required=True);a=p.parse_args();torch.set_num_threads(2)
    if a.mode=='cpu':c=context();result=cpu(c)
    else:c,result=lineage(a.mode=='smoke')
    name={'cpu':'sampler_verification.json','smoke':'smoke_verification.json','lineage':'training_lineage.json'}[a.mode]
    atomic_json(OUT/name,dict(complete=True,signature=c['signature'],verifier_sha256=checksum(__file__),**result));print('INDEPENDENT',a.mode,'PASS',flush=True)

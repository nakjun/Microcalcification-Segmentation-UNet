"""Fixed C/L/H/LH UNeXt training, own-model mining and GT-free OOF export."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse, fcntl, hashlib, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from common import ROOT, ARMS, VARIANTS, checksum, context, immutable, initial_pool, job_signature, read, spec, verify
from model import UNeXtAuthor2ch
from miner import load_current, refresh
from calcification_refinement_v5_common import V5Sampler, training_loss
from calcification_refinement_common import tiled_probability, baseline_masks
from calcification_gate_common import atomic_json
from calcification_gate_sam_common import atomic_npz, disk_guard
import train_calcification_refinement_v2 as v2

def fresh(cfg,device):
    torch.manual_seed(cfg['seed']);model=UNeXtAuthor2ch(cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    return model,optimizer

def job_path(c,fold,smoke=False):return ROOT/('smoke' if smoke else 'jobs')/c['variant']/f'fold_{fold}'

def train(c,fold,device,smoke=False,until_epoch=None):
    f=spec(c,fold);lookup={r['image_id']:r for r in c['rows']};rows=[lookup[i] for i in f['train_ids']]
    job=job_path(c,fold,smoke);sig=job_signature(c,fold,smoke);cfg=dict(c['learning_config'],root=str(job))
    if (job/'complete.json').exists():verify(job/'complete.json',sig);return
    model,opt=fresh(cfg,device);epochs,total=(1,32) if smoke else (16,2048);batch=16
    meta=dict(signature=sig,model='UNeXt_author_2ch',variant=c['variant'],fold=fold,train_ids=f['train_ids'],validation_ids=f['validation_ids'],initial_model_sha256=v2.state_digest(model),parameters=sum(p.numel() for p in model.parameters()),epochs=epochs,patches_per_epoch=total,optimizer=dict(name='AdamW',lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']))
    immutable(job/'job.json',meta);start=1;available=None
    if (job/'resume.json').exists():
        m=verify(job/'resume.json',sig);st=torch.load(job/'resume.pt',map_location=device,weights_only=False)
        assert st['signature']==sig and st['job']==meta and st['epoch']==m['epoch']
        assert read(job/'epochs'/f'{st["epoch"]:02d}.json')==st['epoch_report']
        model.load_state_dict(st['model'],strict=True);opt.load_state_dict(st['optimizer']);start=st['epoch']+1
        torch.set_rng_state(st['rng_cpu'].cpu());torch.cuda.set_rng_state(st['rng_cuda'].cpu(),device)
        available=st['available_mining']
        if not smoke:assert load_current(c,job,fold,start)[1]==available
    for epoch in range(start,(until_epoch or epochs)+1):
        disk_guard(cfg);torch.cuda.reset_peak_memory_stats(device)
        centers=initial_pool(c,fold);current,active=(None,None) if smoke else load_current(c,job,fold,epoch)
        assert active==available
        if current is not None:
            assert set(current)=={ident+'_hard' for ident in f['train_ids']}
            for k,value in current.items():centers[k]=value
        sampler=V5Sampler(rows,cfg,centers);rng=np.random.default_rng(42+epoch*1009)
        stream=hashlib.sha256();geometry=hashlib.sha256();sums=Counter();modes=Counter();started=time.monotonic();model.train()
        for _ in range(total//batch):
            examples=[sampler.sample(rng) for _ in range(batch)]
            xa,ya,wa,va=[np.stack([x[k] for x in examples]) for k in (0,1,3,4)]
            stream.update(xa.tobytes());stream.update(ya.tobytes());geometry.update(wa.tobytes());geometry.update(va.tobytes());modes.update(x[2] for x in examples)
            x,y,w,v=[torch.from_numpy(a).to(device) for a in (xa,ya,wa,va)];opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):z=model(x)
            # Smoke tests full-strength losses separately from the fixed epoch1-3 warmup.
            loss,stats=training_loss(z,y,w,v,cfg,6 if smoke else epoch)
            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(norm);opt.step()
            sums.update(stats);sums['loss']+=float(loss.detach())
        torch.cuda.synchronize(device)
        report=dict(complete=True,signature=sig,variant=c['variant'],epoch=epoch,patches=total,optimizer_steps=total//batch,input_target_sha256=stream.hexdigest(),geometry_sha256=geometry.hexdigest(),sampling_modes=dict(modes),active_mining=active,elapsed_seconds=time.monotonic()-started,peak_memory_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),peak_memory_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),mean_stats={k:v/(total//batch) for k,v in sums.items()},smoke_loss_epoch=6 if smoke else None)
        if not smoke and epoch in (3,7,11):available=refresh(c,job,fold,epoch,model,device)
        report['available_mining']=available
        immutable(job/'epochs'/f'{epoch:02d}.json',report)
        state=dict(signature=sig,job=meta,config=cfg,epoch=epoch,epoch_report=report,model=model.state_dict(),optimizer=opt.state_dict(),available_mining=available,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(device))
        v2.atomic_torch(job/'resume.pt',state)
        atomic_json(job/'resume.json',dict(complete=True,signature=sig,epoch=epoch,output_files={str(job/'resume.pt'):checksum(job/'resume.pt')}))
        print(f'UNeXt {c["variant"]} fold{fold} epoch{epoch}/{epochs}',flush=True)
    if until_epoch is not None:return
    st=torch.load(job/'resume.pt',map_location='cpu',weights_only=False);assert st['epoch']==epochs
    v2.atomic_torch(job/'checkpoint.pt',st)
    outputs={str(p):checksum(p) for p in job.rglob('*') if p.is_file() and p.name!='complete.json'}
    for p in job.glob('mining/*/complete.json'):outputs[str(p)]=checksum(p)
    immutable(job/'complete.json',dict(complete=True,signature=sig,source_signature=c['signature'],job=meta,variant=c['variant'],fold=fold,smoke=smoke,epochs=epochs,output_files=outputs))

def infer(c,fold,device):
    job=job_path(c,fold);sig=job_signature(c,fold);done=verify(job/'complete.json',sig);f=spec(c,fold)
    assert done['epochs']==16 and not done['smoke']
    st=torch.load(job/'checkpoint.pt',map_location=device,weights_only=False);assert st['signature']==sig and st['epoch']==16 and st['job']==done['job']
    model=UNeXtAuthor2ch(c['learning_config']).to(device);model.load_state_dict(st['model'],strict=True);model.eval()
    destination=job/'predictions';outputs={};lookup={r['image_id']:r for r in c['rows']}
    for ident in f['validation_ids']:
        row=lookup[ident];output=destination/f'{ident}.npz';metadata=output.with_suffix('.json')
        if metadata.exists():
            m=verify(metadata,c['signature']);assert m['image_id']==ident and m['variant']==c['variant'] and m['fold']==fold and not m['GT_read']
            assert m['job_signature']==sig and m['checkpoint_sha256']==checksum(job/'checkpoint.pt') and m['native_sha256']==row['native_cache_sha256']
        else:
            disk_guard(dict(c['learning_config'],root=str(ROOT)),extra=row['height']*row['width']*16)
            assert checksum(row['native_cache_path'])==row['native_cache_sha256'];raw=np.load(row['native_cache_path'],mmap_mode='r')
            prob,tiles=tiled_probability(model,raw,c['learning_config'],device)
            assert prob.dtype==np.float32 and prob.shape==raw.shape and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
            flat=np.flatnonzero(prob.ravel()>=np.float32(.3));payload=dict(flat=flat,probability=prob.ravel()[flat],shape=np.asarray(raw.shape))
            for arm in ARMS:
                _,fill,base=baseline_masks(c['learning_config']['development_baseline_root'],arm,ident)
                payload['fill_'+arm]=np.packbits(fill);payload['gate_'+arm]=np.asarray(bool(base['gate_pass']))
            atomic_npz(output,**payload)
            immutable(metadata,dict(complete=True,signature=c['signature'],variant=c['variant'],fold=fold,image_id=ident,GT_read=False,job_signature=sig,checkpoint_sha256=checksum(job/'checkpoint.pt'),native_sha256=row['native_cache_sha256'],tiles=tiles,output_files={str(output):checksum(output)}))
        outputs[str(output)]=checksum(output);outputs[str(metadata)]=checksum(metadata)
        print(f'UNeXt {c["variant"]} fold{fold} score {ident}',flush=True)
    outputs[str(job/'complete.json')]=checksum(job/'complete.json')
    immutable(destination/'complete.json',dict(complete=True,signature=c['signature'],variant=c['variant'],fold=fold,image_ids=f['validation_ids'],GT_read=False,job_signature=sig,checkpoint_sha256=checksum(job/'checkpoint.pt'),output_files=outputs))

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','smoke','warmup','train','infer'],required=True);p.add_argument('--variant',choices=VARIANTS,default='C');p.add_argument('--fold',type=int,choices=[0,1,2],default=0);p.add_argument('--gpu',type=int,default=0);a=p.parse_args();c=context(a.variant);runtime=ROOT/'runtime';runtime.mkdir(parents=True,exist_ok=True)
    with (runtime/f'{a.variant}_{a.fold}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.mode=='prepare':immutable(ROOT/'protocol.json',c['protocol']);return
        assert (ROOT/'protocol.json').exists();torch.use_deterministic_algorithms(True,warn_only=False);torch.set_num_threads(4);torch.cuda.set_device(a.gpu);torch.cuda.set_per_process_memory_fraction(.4,a.gpu);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;device=torch.device(f'cuda:{a.gpu}')
        if a.mode=='train':
            verified=verify(ROOT/'warmup_verified.json',c['signature']);assert verified['jobs']==12 and verified['all_model_optimizer_pool_equal']
        if a.mode=='infer':infer(c,a.fold,device)
        else:train(c,a.fold,device,a.mode=='smoke',3 if a.mode=='warmup' else None)
if __name__=='__main__':main()

"""GT-free held32 native inference for each retained epoch16 fold checkpoint."""
import argparse, fcntl, gc, time, sys, shutil
from pathlib import Path
import numpy as np, torch
from common import ARMS, ROOT, atomic_npz, checksum, context, import_models, immutable, prediction_dir, read, verify
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from calcification_refinement_common import baseline_masks, tiled_probability

def load_model(cp,device):
 StudyModel,UNeXtAuthor2ch=import_models();state=torch.load(cp['checkpoint'],map_location='cpu',weights_only=False)
 assert checksum(cp['checkpoint'])==cp['checkpoint_sha256'] and state['epoch']==16 and state['job']['train_ids']
 model=UNeXtAuthor2ch(cp['model_config']) if cp['kind']=='unext' else StudyModel(cp['model_config'])
 model.load_state_dict(state['model'],strict=True);return model.requires_grad_(False).eval().to(device)

def run(cp,rows,sig,device,smoke):
 out=prediction_dir(cp)/('smoke' if smoke else 'predictions');out.mkdir(parents=True,exist_ok=True)
 with (out/'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);model=load_model(cp,device);outputs={};start=time.time();active=rows[:1] if smoke else rows
  for i,row in enumerate(active):
   assert shutil.disk_usage(ROOT).free > 2*2**30,'held32 inference stops below 2 GiB free disk reserve'
   assert checksum(row['native_cache_path'])==row['native_cache_sha256'];path=out/f"{row['image_id']}.npz";meta=path.with_suffix('.json')
   if meta.exists():
    m=verify(meta,sig);assert m['checkpoint_sha256']==cp['checkpoint_sha256'] and m['native_sha256']==row['native_cache_sha256'] and m['smoke']==smoke
   else:
    raw=np.load(row['native_cache_path'],mmap_mode='r');prob,tiles=tiled_probability(model,raw,cp['model_config'],device);assert prob.dtype==np.float32 and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
    payload=dict(shape=np.asarray(raw.shape),flat=np.flatnonzero(prob.ravel()>=np.float32(.3)).astype(np.int64));payload['probability']=prob.ravel()[payload['flat']]
    for arm in ARMS:
     _,fill,gate=baseline_masks('runs/calcification_refinement_v1/baseline_development',arm,row['image_id']);payload['fill_'+arm]=np.packbits(fill);payload['gate_'+arm]=np.asarray(bool(gate['gate_pass']))
    atomic_npz(path,**payload);immutable(meta,dict(complete=True,signature=sig,model=cp['model'],variant=cp['variant'],fold=cp['fold'],checkpoint_sha256=cp['checkpoint_sha256'],image_id=row['image_id'],native_sha256=row['native_cache_sha256'],GT_read=False,smoke=smoke,tiles=tiles,output_files={str(path):checksum(path)}))
   outputs[str(path)]=checksum(path);outputs[str(meta)]=checksum(meta);print(f"held32 {cp['model']}/{cp['variant']}/f{cp['fold']} {i+1}/{len(active)} {time.time()-start:.1f}s",flush=True)
  immutable(out/'complete.json',dict(complete=True,signature=sig,model=cp['model'],variant=cp['variant'],fold=cp['fold'],checkpoint_sha256=cp['checkpoint_sha256'],image_ids=[r['image_id'] for r in active],GT_read=False,smoke=smoke,output_files=outputs))
 del model;gc.collect();torch.cuda.empty_cache()

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--model',choices=('proposed','unext','unet2d','utnet'));p.add_argument('--variant',choices=('C','L','H','LH'));p.add_argument('--fold',type=int,choices=(0,1,2));p.add_argument('--smoke',action='store_true');a=p.parse_args();c=context()
 chosen=[x for x in c['checkpoints'] if (a.model is None or x['model']==a.model) and (a.variant is None or x['variant']==a.variant) and (a.fold is None or x['fold']==a.fold)]
 assert chosen and (not a.smoke or len(chosen)==1),'smoke requires one explicit checkpoint';assert torch.cuda.is_available();torch.use_deterministic_algorithms(True,warn_only=False);torch.set_num_threads(4);torch.cuda.set_device(a.gpu);torch.cuda.set_per_process_memory_fraction(.45,a.gpu);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;device=torch.device(f'cuda:{a.gpu}')
 for cp in chosen:run(cp,c['smoke_rows'] if a.smoke else c['rows'],c['signature'],device,a.smoke)
if __name__=='__main__':main()

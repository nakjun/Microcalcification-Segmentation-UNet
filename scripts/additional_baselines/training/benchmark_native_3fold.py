"""Six-method native inference benchmark, preserving evaluated inference paths."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys,json,time,hashlib,argparse,subprocess
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/additional_baselines/efficiency_native_3fold_20260927'
NAMES=['proposed','unext','unet2d','utnet','deepmica','yolo']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=NAMES);ap.add_argument('--fold',type=int,choices=[0,1,2],default=0);a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 if not a.model:
  assert not (OUT/'complete.json').exists(), 'Preserve completed benchmark'
  original=ROOT/'reports/additional_baselines/efficiency_native_20260927/complete.json'
  existing=json.loads(original.read_text());assert existing['complete']
  protocol=dict(existing['protocol'],folds=[0,1,2],fold0_source=str(original.relative_to(ROOT)),fold0_source_sha256=sha(original),aggregation='mean and sample SD (ddof=1) across three fold checkpoints; per-fold latency is median of 3 passes, memory is maximum peak over 3 passes; throughput computed per fold before aggregation')
  write(OUT/'protocol.json',protocol)
  for name in NAMES:
   record=dict(existing['results'][name],fold=0,reused=True)
   write(OUT/(name+'_fold0.json'),record)
  for name in NAMES:
   for fold in [1,2]:
    target=OUT/(name+f'_fold{fold}.json')
    if not target.exists():subprocess.run([sys.executable,__file__,'--model',name,'--fold',str(fold)],cwd=ROOT,check=True)
  results={}
  for name in NAMES:
   records=[json.loads((OUT/(name+f'_fold{fold}.json')).read_text()) for fold in range(3)]
   assert all(x['complete'] and x['fold']==i for i,x in enumerate(records))
   fields={'latency_s':[x['native_seconds'] for x in records],'throughput_tiles_s':[700/x['native_seconds'] for x in records],'memory_mib':[x['native_peak_allocated_bytes']/1024**2 for x in records],'parameters_m':[x['parameters']/1e6 for x in records]}
   results[name]=dict(folds=records,summary={k:dict(mean=float(np.mean(v)),sample_sd=float(np.std(v,ddof=1)),values=v) for k,v in fields.items()})
  write(OUT/'complete.json',dict(complete=True,protocol=protocol,results=results));return
 torch.set_num_threads(4);torch.cuda.set_device(0);torch.manual_seed(42);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.use_deterministic_algorithms(True)
 old=json.loads((ROOT/'reports/comparative_study/efficiency.json').read_text());ref=old['results']['unet2d']
 rows=json.loads((ROOT/'runs/calcification_refinement_v1/data/manifest.json').read_text());row=next(r for r in rows if r['image_id']==ref['native_image_id']);assert row['role']=='train'
 path=ROOT/row['native_cache_path'];assert sha(path)==ref['native_sha256'];raw=np.array(np.load(path,mmap_mode='r'),copy=True)
 h,w=raw.shape
 def starts(n):return sorted(set(range(0,max(n-256,0)+1,192))|{max(n-256,0)})
 ys,xs=starts(h),starts(w);assert len(ys)*len(xs)==700
 cps={};precision='FP32';batch=1
 if a.model in NAMES[:4]:
  sys.path.insert(0,str(ROOT/'scripts/comparative_study'));import verify_efficiency as v
  import importlib.util
  root=ROOT/'runs/comparative_study'/(a.model+'_seed42') if a.model!='unext' else ROOT/'runs/improvement_unext_deterministic/unext_v5_losses_deterministic_seed42'
  cp=root/f'jobs/LH/fold_{a.fold}/checkpoint.pt';state=torch.load(cp,map_location='cpu',weights_only=False);assert state['epoch']==16;cfg=state['config']
  if a.model=='unext':
   spec=importlib.util.spec_from_file_location('benchmark_unext_author',ROOT/'scripts/improvement_unext_deterministic/model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);model=module.UNeXtAuthor2ch(cfg)
  else:model=v.StudyModel(cfg)
  model.load_state_dict(state['model'],strict=True);model=model.to('cuda:0').eval();modules=[model];cps[str(cp)]=sha(cp)
  evaluation=json.loads((ROOT/'runs/comparative_held32_seed42/evaluation_counts/complete.json').read_text())
  assert evaluation['conditions'][a.model]['LH']['folds'][a.fold]['checkpoint_sha256']==sha(cp)
  def predict():return v.tiled_probability(model,raw,cfg,torch.device('cuda:0'))[0]
  precision='BF16';batch=16
 elif a.model=='deepmica':
  sys.path.insert(0,str(ROOT/'DeepMiCa'));from pipeline import load_checkpoint,tiled_predict
  cp=ROOT/f'runs/additional_baselines_20260926/deepmica/fold_{a.fold}/last.pt';model,meta=load_checkpoint(cp,'cuda:0');modules=[model];cps[str(cp)]=sha(cp)
  assert meta['channels']==1 and meta['scale']==255
  def predict():return tiled_predict(model,torch.from_numpy(raw.astype(np.float32)/255)[None],256,192,'cuda:0')
 else:
  sys.path.insert(0,str(ROOT/'YOLO-HLSAM'));from model import Detector,Segmentor,predict as patch
  base=ROOT/f'runs/additional_baselines_20260926/yolo/fold_{a.fold}';dc=base/'detector/epoch_016.pt';sc=base/'segmentor/epoch_016.pt';sam=ROOT/'runs/mammo_hypothesis/weights/sam_vit_b.pth'
  detector=Detector(dc,'cuda:0',640,.25,.45,100);segmentor=Segmentor('vit_b',sam,sc).to('cuda:0').eval();modules=[detector.model,segmentor]
  cps={str(p):sha(p) for p in [dc,sc,sam]}
  def predict():
   result=np.zeros((h,w),np.float32);num=np.zeros((h,w),np.uint16)
   for y in ys:
    for x in xs:
     tile=np.repeat(raw[y:y+256,x:x+256,None],3,axis=2);q,_,_=patch(tile,detector,segmentor);result[y:y+q.shape[0],x:x+q.shape[1]]+=q;num[y:y+q.shape[0],x:x+q.shape[1]]+=1
   assert (num>0).all();return result/num
 if a.model in ['deepmica','yolo']:
  evaluation=json.loads((ROOT/f'runs/additional_baselines_20260926/{a.model}/internal_complete.json').read_text())
  for p,digest in cps.items():assert evaluation['checkpoint_sha256'][p]==digest
 print(a.model,a.fold,'warmup',flush=True)
 with torch.inference_mode():
  q=predict();assert q.shape==raw.shape and np.isfinite(q).all();del q;torch.cuda.synchronize()
  times=[];peaks=[];hashes=[]
  for i in range(3):
   torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();q=predict();torch.cuda.synchronize();times.append(time.perf_counter()-start);peaks.append(torch.cuda.max_memory_allocated());assert q.shape==raw.shape and np.isfinite(q).all();hashes.append(hashlib.sha256(q.tobytes()).hexdigest());del q;print(a.model,a.fold,i,times[-1],peaks[-1],flush=True)
 assert len(set(hashes))==1,'Repeated native probabilities changed'
 params={id(p):p for m in modules for p in m.parameters()}
 write(OUT/(a.model+f'_fold{a.fold}.json'),dict(model=a.model,fold=a.fold,complete=True,native_image_id=row['image_id'],native_shape=list(raw.shape),native_sha256=sha(path),native_tiles=700,native_seconds=float(np.median(times)),native_seconds_runs=times,native_peak_allocated_bytes=max(peaks),native_peak_runs=peaks,parameters=sum(p.numel() for p in params.values()),trainable_parameters=sum(p.numel() for p in params.values() if p.requires_grad),precision=precision,batch=batch,checkpoint_sha256=cps,probability_sha256=hashes[0],GT_read=False,gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__),cuda=torch.version.cuda,script_sha256=sha(__file__)))
if __name__=='__main__':main()

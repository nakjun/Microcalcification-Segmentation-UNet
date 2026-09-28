"""Six-method native inference benchmark, preserving evaluated inference paths."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys,json,time,hashlib,argparse,subprocess
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/additional_baselines/efficiency_native_20260927'
NAMES=['proposed','unext','unet2d','utnet','deepmica','yolo']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=NAMES);a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 if not a.model:
  assert not (OUT/'complete.json').exists(), 'Preserve completed benchmark'
  protocol=dict(models=NAMES,device='cuda:0',native_image='276_71',tiles=700,patch=256,stride=192,repetitions=3,warmup='one complete native pass',scope='evaluated inference pipelines; four backbones BF16 batch16, DeepMiCa FP32 batch1, YOLO-HLSAM FP32 sequential detector+SAM/HQ predicted boxes',timing='resident uint8 image through probability map; checkpoint/file loading excluded; model-specific preprocessing, transfer, inference and merging included',memory='absolute peak allocated CUDA memory during native inference, including resident models; not patch-forward increment',GT_read=False)
  write(OUT/'protocol.json',protocol)
  for name in NAMES:subprocess.run([sys.executable,__file__,'--model',name],cwd=ROOT,check=True)
  write(OUT/'complete.json',dict(complete=True,protocol=protocol,results={n:json.loads((OUT/(n+'.json')).read_text()) for n in NAMES}));return
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
  model,cfg,cp=v.load_model(a.model);model=model.to('cuda:0').eval();modules=[model];cps[str(cp)]=sha(cp);assert sha(cp)==old['results'][a.model]['checkpoint_sha256']
  def predict():return v.tiled_probability(model,raw,cfg,torch.device('cuda:0'))[0]
  precision='BF16';batch=16
 elif a.model=='deepmica':
  sys.path.insert(0,str(ROOT/'DeepMiCa'));from pipeline import load_checkpoint,tiled_predict
  cp=ROOT/'runs/additional_baselines_20260926/deepmica/fold_0/last.pt';model,meta=load_checkpoint(cp,'cuda:0');modules=[model];cps[str(cp)]=sha(cp)
  assert meta['channels']==1 and meta['scale']==255
  def predict():return tiled_predict(model,torch.from_numpy(raw.astype(np.float32)/255)[None],256,192,'cuda:0')
 else:
  sys.path.insert(0,str(ROOT/'YOLO-HLSAM'));from model import Detector,Segmentor,predict as patch
  base=ROOT/'runs/additional_baselines_20260926/yolo/fold_0';dc=base/'detector/epoch_016.pt';sc=base/'segmentor/epoch_016.pt';sam=ROOT/'runs/mammo_hypothesis/weights/sam_vit_b.pth'
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
 print(a.model,'warmup',flush=True)
 with torch.inference_mode():
  q=predict();assert q.shape==raw.shape and np.isfinite(q).all();del q;torch.cuda.synchronize()
  times=[];peaks=[];hashes=[]
  for i in range(3):
   torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();q=predict();torch.cuda.synchronize();times.append(time.perf_counter()-start);peaks.append(torch.cuda.max_memory_allocated());assert q.shape==raw.shape and np.isfinite(q).all();hashes.append(hashlib.sha256(q.tobytes()).hexdigest());del q;print(a.model,i,times[-1],peaks[-1],flush=True)
 assert len(set(hashes))==1,'Repeated native probabilities changed'
 params={id(p):p for m in modules for p in m.parameters()}
 write(OUT/(a.model+'.json'),dict(model=a.model,complete=True,native_image_id=row['image_id'],native_shape=list(raw.shape),native_sha256=sha(path),native_tiles=700,native_seconds=float(np.median(times)),native_seconds_runs=times,native_peak_allocated_bytes=max(peaks),native_peak_runs=peaks,parameters=sum(p.numel() for p in params.values()),trainable_parameters=sum(p.numel() for p in params.values() if p.requires_grad),precision=precision,batch=batch,checkpoint_sha256=cps,probability_sha256=hashes[0],GT_read=False,gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__),cuda=torch.version.cuda,script_sha256=sha(__file__)))
if __name__=='__main__':main()

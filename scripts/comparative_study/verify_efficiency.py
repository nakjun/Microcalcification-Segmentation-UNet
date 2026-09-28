"""Isolated same-GPU forward and whole-native efficiency; no GT reads."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys,json,time,hashlib,importlib.util,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from calcification_gate_common import atomic_json,checksum
from calcification_refinement_common import tiled_probability
from model import StudyModel

def load_model(name):
 root=Path('runs/comparative_study')/(name+'_seed42') if name!='unext' else Path('runs/improvement_unext_deterministic/unext_v5_losses_deterministic_seed42')
 p=root/'jobs/LH/fold_0/checkpoint.pt';state=torch.load(p,map_location='cpu',weights_only=False);assert state['epoch']==16
 if name=='unext':
  spec=importlib.util.spec_from_file_location('benchmark_unext_author','scripts/improvement_unext_deterministic/model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);m=module.UNeXtAuthor2ch(state['config'])
 else:m=StudyModel(state['config'])
 m.load_state_dict(state['model'],strict=True)
 return m,state['config'],p

def main():
 torch.set_num_threads(4);torch.cuda.set_device(0);torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 device=torch.device('cuda:0');torch.manual_seed(42);x=torch.randn(16,2,256,256,device='cpu').to(device)
 fold=json.loads(Path('runs/calcification_refinement_v5_seed42/folds.json').read_text());rows=json.loads(Path(fold['source_manifest']).read_text());row=sorted([r for r in rows if r['role']=='train'],key=lambda r:(-r['height']*r['width'],r['image_id']))[0]
 assert checksum(row['native_cache_path'])==row['native_cache_sha256'];raw=np.load(row['native_cache_path'],mmap_mode='r');results={}
 for name in ('proposed','unext','unet2d','utnet'):
  m,cfg,path=load_model(name)
  assert int(cfg['input_channels']) == 2
  assert int(cfg['patch_size']) == 256
  assert int(cfg['tile_stride']) == 192
  assert int(cfg['inference_batch_size']) == 16
  params=sum(p.numel() for p in m.parameters());m=m.to(device).eval()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   for _ in range(10):z=m(x)
   torch.cuda.synchronize();del z;torch.cuda.empty_cache();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();values=[]
   for _ in range(50):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();z=m(x);b.record();b.synchronize();values.append(a.elapsed_time(b));del z
   increment=torch.cuda.max_memory_allocated()-base
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();native_start=time.perf_counter();prob,tiles=tiled_probability(m,raw,cfg,device);torch.cuda.synchronize();native_seconds=time.perf_counter()-native_start
  assert prob.shape==raw.shape and np.isfinite(prob).all()
  results[name]=dict(parameters=params,batch=16,input_shape=[16,2,256,256],warmup_iterations=10,timed_iterations=50,forward_ms_mean=float(np.mean(values)),forward_ms_median=float(np.median(values)),images_per_second=16000/float(np.mean(values)),incremental_peak_allocated_bytes=int(increment),native_image_id=row['image_id'],native_shape=list(raw.shape),native_sha256=row['native_cache_sha256'],native_tiles=tiles,native_seconds=native_seconds,native_peak_allocated_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=checksum(path),checkpoint=str(path),GT_read=False,raw_forward_ms=[float(v) for v in values])
  print(name,'native_seconds',native_seconds,flush=True);del m,prob;gc.collect();torch.cuda.empty_cache()
 atomic_json('reports/comparative_study/efficiency.json',dict(complete=True,gpu=0,gpu_name=torch.cuda.get_device_name(0),torch=str(torch.__version__),cuda=torch.version.cuda,precision='BF16 autocast forward',throughput_unit='256x256 input patches/second; not whole-native mammograms/second',strict_deterministic=True,scope='isolated refinement-network forward only; no optimizer/backward, no training throughput claim',native_timing='refinement network whole-native tiled inference only; model/checkpoint load, common detector/SAM/gate generation, and their timing are excluded; mmap opened before timing; CPU tile preprocessing/transfer/overlap merge included; OS cache state uncontrolled',verifier_sha256=checksum(__file__),results=results))
if __name__=='__main__':main()

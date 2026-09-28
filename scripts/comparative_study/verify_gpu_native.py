"""Parent GPU whole-native source replay and independent mining oracle."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse,hashlib,time
import numpy as np
import torch
from common import ROOT,OUT,context,spec,read,checksum
from model import StudyModel
from miner import refresh
from calcification_refinement_common import tiled_probability,tile_starts
from calcification_gate_common import atomic_json

p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);a=p.parse_args();torch.use_deterministic_algorithms(True,warn_only=False);torch.set_num_threads(4);torch.cuda.set_device(a.gpu);torch.cuda.set_per_process_memory_fraction(.4,a.gpu);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;device=torch.device(f'cuda:{a.gpu}')
c=context('LH');cfg=c['learning_config'];job=ROOT/'smoke/LH/fold_0';state=torch.load(job/'checkpoint.pt',map_location=device,weights_only=False);model=StudyModel(cfg).to(device);model.load_state_dict(state['model'],strict=True)
native=ROOT/'native_smoke/LH/fold_0';torch.cuda.reset_peak_memory_stats(device);started=time.monotonic();bound=refresh(c,native,0,1,model,device,smoke=True)
lookup={r['image_id']:r for r in c['rows']};first=lookup[spec(c,0)['train_ids'][0]];stage=native/'mining/after_01';meta=read(stage/'cases'/f'{first["image_id"]}.json');src=torch.load(stage/'source.pt',map_location=device,weights_only=False);restored=StudyModel(cfg).to(device);restored.load_state_dict(src['model'],strict=True)
raw=np.load(first['native_cache_path'],mmap_mode='r');prob,tiles=tiled_probability(restored,raw,cfg,device)
assert hashlib.sha256(prob.tobytes()).hexdigest()==meta['probability_sha256']
with np.load(first['target_path']) as z:ys,xs=z['y'],z['x']
oracle=[];h,w=raw.shape
for y0 in range(0,h,128):
    for x0 in range(0,w,128):
        cell=prob[y0:y0+128,x0:x0+128];i=int(cell.argmax());dy,dx=divmod(i,cell.shape[1]);cy,cx=y0+dy,x0+dx;s=float(cell[dy,dx])
        if s<float(np.float32(.3)):continue
        y=max(0,min(cy-128,max(0,h-256)));x=max(0,min(cx-128,max(0,w-256)))
        if not np.any((ys>=y)&(ys<y+256)&(xs>=x)&(xs<x+256)):oracle.append((s,cy,cx))
oracle=sorted(oracle,key=lambda t:(-t[0],t[1],t[2]))[:4096]
with np.load(stage/'cases'/f'{first["image_id"]}.npz') as z:
    expected=np.asarray([(y,x) for s,y,x in oracle],np.int32).reshape(-1,2);scores=np.asarray([s for s,y,x in oracle],np.float32)
    assert np.array_equal(z['centers'],expected) and np.array_equal(z['scores'],scores)
checks=[]
maximum=max([lookup[i] for i in spec(c,0)['train_ids']],key=lambda r:r['height']*r['width'])
for row in [first]+([maximum] if maximum['image_id']!=first['image_id'] else []):
    image=np.load(row['native_cache_path'],mmap_mode='r')
    if row is first:result,num=prob,tiles
    else:result,num=tiled_probability(restored,image,cfg,device)
    assert result.dtype==np.float32 and result.shape==image.shape and np.isfinite(result).all() and ((result>=0)&(result<=1)).all()
    cover=np.zeros(image.shape,np.uint8)
    for yy in tile_starts(image.shape[0],256,192):
        for xx in tile_starts(image.shape[1],256,192):cover[yy:yy+256,xx:xx+256]+=1
    assert cover.min()>0 and num==len(tile_starts(image.shape[0],256,192))*len(tile_starts(image.shape[1],256,192))
    checks.append(dict(image_id=row['image_id'],shape=list(image.shape),tiles=num,full_native_coverage=True,GT_used_for_inference=False))
assert refresh(c,native,0,1,model,device,smoke=True)==bound
atomic_json(OUT/'gpu_native_verification.json',dict(complete=True,signature=c['signature'],verifier_sha256=checksum(__file__),checkpoint_sha256=checksum(job/'checkpoint.pt'),source_probability_reload_exact=True,independent_mining_centers_scores_exact=True,mining_centers=len(oracle),checks=checks,source_binding=bound,peak_memory_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),elapsed_seconds=time.monotonic()-started,performance_evaluation=False))
print('INDEPENDENT GPU native PASS',flush=True)

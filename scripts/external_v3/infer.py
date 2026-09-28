"""Whole-image inference only. No labels, GT ROIs, fitting, or threshold selection."""
from common import *
import argparse,fcntl,gc,time
from calcification_gate_sam_common import (native_dicom,load_gate,gate_logits,load_detector,detector_prediction,selected_boxes,box_fill_mask,disk_guard)
from calcification_refinement_common import PatchUNet,tiled_probability
from prepare_retinanet_roi import breast_roi
def dicom_roi(path,shape):
 d=pydicom.dcmread(path);raw=d.pixel_array;assert raw.shape==tuple(shape)
 a=raw[5:-5,5:-5].astype(np.float32)
 assert d.PhotometricInterpretation in ['MONOCHROME1','MONOCHROME2']
 if d.PhotometricInterpretation=='MONOCHROME1':a=a.max()-a
 a-=a.min();assert a.max()>0
 a=(a/a.max()*255).astype(np.uint8);x,y,u,v=breast_roi(a);assert u>x and v>y
 padded=np.zeros((1536,928),np.uint8);padded[:1520,:912]=cv2.resize(a[y:v,x:u],(912,1520),interpolation=cv2.INTER_AREA)
 return padded,[x+5,y+5,u+5,v+5]
def policy(p,fill,gate):return bool(gate)&((fill&(p>=np.float32(.3)))|(~fill&(p>=np.float32(.9))))
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['inbreast','cbis_test'],required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 c=context(a.dataset);assert (ROOT/a.dataset/'protocol.json').exists()
 out=ROOT/a.dataset/('smoke' if a.smoke else 'predictions');out.mkdir(parents=True,exist_ok=True)
 runtime=ROOT/'runtime';runtime.mkdir(exist_ok=True)
 with (runtime/f'{a.dataset}_{a.smoke}.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  torch.set_num_threads(4);cv2.setNumThreads(1);torch.cuda.set_device(a.gpu);torch.cuda.set_per_process_memory_fraction(.45,a.gpu)
  torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;device=torch.device(f'cuda:{a.gpu}')
  state=torch.load(c['checkpoint'],map_location='cpu',weights_only=False)
  assert state['epoch']==15 and state['job']==c['protocol']['job']
  model=PatchUNet(c['config']).to(device);model.load_state_dict(state['model'],strict=True);model.eval();del state
  encoder,gate,gconfig=load_gate(PIPE/'frozen/gate/classifier.pt',device)
  detectors={arm:load_detector(arm,PIPE/f'frozen/{arm}/best.pt',device) for arm in ARMS}
  thresholds={arm:read(PIPE/f'frozen/{arm}/snapshot.json')['threshold'] for arm in ARMS}
  gthreshold=read(PIPE/'frozen/gate/thresholds.json')['primary']['threshold_logit']
  rows=c['rows'][:1] if a.smoke else c['rows'];outputs={};start=time.time()
  for i,row in enumerate(rows):
   assert checksum(row['dicom_path'])==row['dicom_sha256']
   path=out/f'{row["image_id"]}.npz';mp=path.with_suffix('.json')
   if mp.exists():
    m=verify(mp,c['signature']);assert m['image_id']==row['image_id'] and m['dataset']==a.dataset and m['dicom_sha256']==row['dicom_sha256'] and m['GT_read'] is False and m['smoke']==a.smoke
    assert set(m['output_files'])=={str(path)}
   else:
    disk_guard(dict(root=str(ROOT),disk_reserve_gib=8),extra=row['height']*row['width']*16)
    shape=(row['height'],row['width'])
    padded,roi=dicom_roi(row['dicom_path'],shape)
    logit=float(gate_logits(encoder,gate,torch.from_numpy(padded[None,None]),gconfig,device)[0]);passed=bool(logit>=gthreshold)
    # Rejected images are exact empty under the frozen gate; raw model probabilities are not measured.
    if passed:
     native=native_dicom(row['dicom_path'],shape);prob=tiled_probability(model,native,c['config'],device)[0]
    else:prob=np.zeros(shape,np.float32)
    assert prob.dtype==np.float32 and prob.shape==shape and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
    flat=np.flatnonzero(prob.ravel()>=np.float32(.3));payload=dict(shape=np.array(shape),flat=flat,probability=prob.ravel()[flat],gate_logit=np.array(logit),gate_pass=np.array(passed))
    for arm in ARMS:
     boxes=selected_boxes(detector_prediction(detectors[arm],padded,roi,device),thresholds[arm],*shape) if passed else np.empty((0,5),np.float32)
     fill=box_fill_mask(shape,boxes);mask=policy(prob,fill,passed)
     payload['boxes_'+arm]=boxes;payload['fill_'+arm]=np.packbits(fill);payload['mask_'+arm]=np.packbits(mask)
    atomic_npz(path,**payload)
    immutable(mp,dict(complete=True,signature=c['signature'],image_id=row['image_id'],dataset=a.dataset,dicom_sha256=row['dicom_sha256'],GT_read=False,smoke=a.smoke,refinement_executed=passed,roi=roi,output_files={str(path):checksum(path)}))
   outputs[str(path)]=checksum(path);outputs[str(mp)]=checksum(mp)
   print(f'{a.dataset} {i+1}/{len(rows)} elapsed={time.time()-start:.1f}s',flush=True)
  immutable(out/'complete.json',dict(complete=True,signature=c['signature'],image_ids=[r['image_id'] for r in rows],dataset=a.dataset,GT_read=False,smoke=a.smoke,output_files=outputs))
  print(json.dumps(dict(peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,seconds=time.time()-start)))
if __name__=='__main__':main()

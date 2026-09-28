"""Image-only independent artifact checks; never reads annotation files."""
from common import ROOT,ARMS,read,checksum,verify,immutable,atomic_json,digest
from pathlib import Path
import numpy as np
import argparse,fcntl

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['inbreast','cbis_test']);a=p.parse_args();ds=a.dataset
 rows=read(ROOT/'inputs'/f'{ds}.json');sig=digest(read(ROOT/ds/'protocol.json'));src=ROOT/ds/'predictions';out=Path('reports/external_v3/partial')/ds;out.mkdir(parents=True,exist_ok=True)
 gt=read('runs/calcification_gate_retinanet_sam_clean_v1/frozen/gate/thresholds.json')['primary']['threshold_logit'];ts={k:read(f'runs/calcification_gate_retinanet_sam_clean_v1/frozen/{k}/snapshot.json')['threshold'] for k in ARMS};checked=[]
 with (out/'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for row in rows:
   ident=row['image_id'];mp=src/f'{ident}.json'
   if not mp.exists():continue
   cp=out/f'{ident}.json';npz=mp.with_suffix('.npz');m=verify(mp,sig);ph=checksum(npz);mh=checksum(mp)
   assert m['image_id']==ident and m['dataset']==ds and m['dicom_sha256']==row['dicom_sha256'] and m['GT_read'] is False and m['smoke'] is False and m['output_files']=={str(npz):ph}
   if cp.exists():
    r=read(cp);assert r['signature']==sig and r['prediction_sha256']==ph and r['metadata_sha256']==mh;checked.append(ident);continue
   with np.load(npz) as z:g={k:z[k] for k in z.files}
   shape=(row['height'],row['width']);n=int(np.prod(shape));assert tuple(g['shape'])==shape
   flat,q=g['flat'],g['probability'];assert flat.dtype==np.int64 and q.dtype==np.float32 and flat.shape==q.shape and np.all(flat[1:]>flat[:-1]) and np.all((flat>=0)&(flat<n)) and np.isfinite(q).all() and np.all((q>=np.float32(.3))&(q<=1))
   gate=bool(g['gate_pass']);assert np.isfinite(g['gate_logit']) and gate==(float(g['gate_logit'])>=gt) and m['refinement_executed']==gate
   if not gate:assert not len(flat)
   for arm in ARMS:
    fill=np.zeros(shape,bool);boxes=g['boxes_'+arm];assert boxes.ndim==2 and boxes.shape[1]==5 and np.isfinite(boxes).all() and np.all(boxes[:,4]>=ts[arm])
    if not gate:assert len(boxes)==0
    for x,y,u,v,_ in boxes:
     assert 0<=x<u<=shape[1] and 0<=y<v<=shape[0];fill[int(np.floor(y)):int(np.ceil(v)),int(np.floor(x)):int(np.ceil(u))]=True
    np.testing.assert_array_equal(fill.ravel(),np.unpackbits(g['fill_'+arm],count=n).astype(bool))
    pred=np.zeros(n,bool)
    if gate:pred[flat[fill.ravel()[flat]|(q>=np.float32(.9))]]=True
    np.testing.assert_array_equal(pred,np.unpackbits(g['mask_'+arm],count=n).astype(bool))
   immutable(cp,dict(complete=True,signature=sig,image_id=ident,prediction_sha256=ph,metadata_sha256=mh,sparse_mask_exact=True,GT_read=False));checked.append(ident)
  atomic_json(out/'progress.json',dict(checked_images=len(checked),expected_images=len(rows),image_ids=checked,GT_read=False,dataset_inference_complete=(src/'complete.json').exists(),scope='artifact reconstruction only;not performance evaluation'))
  print(ds,len(checked),len(rows),flush=True)
if __name__=='__main__':main()

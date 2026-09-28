"""Held32 metrics only after the 48 GT-free prediction cohorts are complete."""
import fcntl,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ARMS,ROOT,atomic_json,atomic_npz,checksum,context,digest,immutable,prediction_dir,read,verify
from calcification_refinement_common import baseline_masks
from calcification_refinement_v2_common import component_pixel_groups
from evaluate_calcification_refinement_v3_union import counts
from train_calcification_refinement_v2 import aggregate

POLICIES=('primary','secondary_v3','ungated_p5')
def masks(prob,fill,gate):
 return dict(primary=bool(gate)&(prob>=np.float32(.9)),secondary_v3=bool(gate)&((fill&(prob>=np.float32(.3)))|(~fill&(prob>=np.float32(.9)))),ungated_p5=prob>=np.float32(.5))
def stats(values):return dict(mean=float(np.mean(values)) if values else None,sample_sd=float(np.std(values,ddof=1)) if len(values)>1 else None,n_folds=len(values))
def flatten(value,prefix=''):
 if isinstance(value,dict):
  out={}
  for k,v in value.items():out.update(flatten(v,prefix+('.' if prefix else '')+k))
  return out
 return {prefix:value} if isinstance(value,(int,float)) and not isinstance(value,bool) else {}
def fold_stats(folds,policy):
 flat=[flatten(x['metrics'][policy]) for x in folds];keys=set().union(*[set(x) for x in flat])
 return {k:stats([x[k] for x in flat if k in x]) | {'n_valid':sum(k in x for x in flat)} for k in keys}
def dense(path,shape):
 with np.load(path) as z:
  assert tuple(z['shape'])==shape;flat,p=z['flat'],z['probability'];assert flat.dtype==np.int64 and p.dtype==np.float32 and flat.shape==p.shape and np.all(flat[1:]>flat[:-1]) and np.all((flat>=0)&(flat<np.prod(shape))) and np.all(np.isfinite(p)) and np.all((p>=.3)&(p<=1))
  q=np.zeros(np.prod(shape),np.float32);q[flat]=p
  return q.reshape(shape),{k:np.array(z[k],copy=True) for k in z.files}
def verify_predictions(c):
 ids=[r['image_id'] for r in c['rows']]
 for cp in c['checkpoints']:
  out=prediction_dir(cp)/'predictions';done=verify(out/'complete.json',c['signature']);assert done['image_ids']==ids and done['GT_read'] is False and not done['smoke'] and done['checkpoint_sha256']==cp['checkpoint_sha256']
  expected={}
  for row in c['rows']:
   p=out/f"{row['image_id']}.npz";m=verify(p.with_suffix('.json'),c['signature']);assert m['native_sha256']==row['native_cache_sha256'] and m['checkpoint_sha256']==cp['checkpoint_sha256'] and m['GT_read'] is False;expected[str(p)]=checksum(p);expected[str(p.with_suffix('.json'))]=checksum(p.with_suffix('.json'))
  assert done['output_files']==expected
def main():
 c=context();out=ROOT/'evaluation_counts';out.mkdir(parents=True,exist_ok=True)
 with (out/'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);aggregate_done=verify(ROOT/'predictions/complete.json',c['signature']);assert aggregate_done['checkpoints']==48 and aggregate_done['images_per_checkpoint']==32
  verify_predictions(c) # No target manifest is opened before this line.
  original=read(c['protocol']['manifest']);held={r['image_id']:r for r in original if r['role']!='train'};assert set(held)=={r['image_id'] for r in c['rows']}
  source=dict(inference_protocol=checksum(ROOT/'protocol.json'),inference_complete=checksum(ROOT/'predictions/complete.json'),manifest=checksum(c['protocol']['manifest']),evaluator=checksum(__file__))
  esig=digest(dict(inference_signature=c['signature'],source=source,policies=POLICIES));immutable(out/'protocol.json',dict(inference_signature=c['signature'],source=source,policies=POLICIES));outputs={str(out/'protocol.json'):checksum(out/'protocol.json')};groups={}
  for cp in c['checkpoints']:
   cases=[];pred=prediction_dir(cp)/'predictions'
   for input_row in c['rows']:
    row=held[input_row['image_id']];shape=(row['height'],row['width']);p=pred/f"{row['image_id']}.npz";case_path=out/'cases'/cp['model']/cp['variant']/f"fold_{cp['fold']}"/f"{row['image_id']}.json"
    if case_path.exists():case=read(case_path);assert case['signature']==esig and case['probability_sha256']==checksum(p)
    else:
     probability,z=dense(p,shape)
     assert checksum(row['target_path'])==row['target_sha256']
     with np.load(row['target_path']) as target:ys,xs=target['y'],target['x']
     components=component_pixel_groups(ys,xs,shape) if len(ys) else [];case=dict(signature=esig,image_id=row['image_id'],study_id=row['study_id'],model=cp['model'],variant=cp['variant'],fold=cp['fold'],checkpoint_sha256=cp['checkpoint_sha256'],probability_sha256=checksum(p),GT_sha256=row['target_sha256'],arms={},output_files={})
     for arm in ARMS:
      _,fill,gate=baseline_masks('runs/calcification_refinement_v1/baseline_development',arm,row['image_id']);stored_fill=np.unpackbits(z['fill_'+arm],count=np.prod(shape)).reshape(shape).astype(bool);stored_gate=bool(z['gate_'+arm]);assert np.array_equal(fill,stored_fill) and stored_gate==bool(gate['gate_pass'])
      arm_masks=masks(probability,fill,stored_gate);case['arms'][arm]={}
      for policy,mask in arm_masks.items():
       # Persist additive counts and provenance only. Packed masks for 48×32×2×3
       # policies exceed the available workspace; the immutable sparse probabilities
       # and stored fill/gate values reconstruct every mask exactly.
       case['arms'][arm][policy]=counts(mask,ys,xs,fill,components)
     atomic_json(case_path,case)
    for q,h in case.get('output_files',{}).items():assert checksum(q)==h;outputs[q]=h
    cases.append(case);outputs[str(case_path)]=checksum(case_path)
   result=dict(fold=cp['fold'],checkpoint=cp['checkpoint'],checkpoint_sha256=cp['checkpoint_sha256'],images=32,metrics={policy:{arm:aggregate([{policy:x['arms'][arm][policy]} for x in cases],policy) for arm in ARMS} for policy in POLICIES})
   groups.setdefault((cp['model'],cp['variant']),[]).append(result)
  conditions={}
  for (model,variant),folds in groups.items():
   assert [x['fold'] for x in folds]==[0,1,2];conditions.setdefault(model,{})[variant]=dict(folds=folds,fold_mean_sample_sd={policy:{arm:fold_stats([dict(metrics={policy:x['metrics'][policy][arm]}) for x in folds],policy) for arm in ARMS} for policy in POLICIES})
  immutable(out/'complete.json',dict(complete=True,signature=esig,inference_signature=c['signature'],scope='previously exposed held32 development confirmation; not OOF or independent test',images_per_fold=32,checkpoints=48,ensemble=False,policies=POLICIES,conditions=conditions,output_files=outputs))
if __name__=='__main__':main()

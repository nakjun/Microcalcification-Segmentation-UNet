"""Independent from image-only cache generation; fixed-policy OOF evaluation only."""
from baseline import *
from policies import POLICIES,apply
import fcntl
import cv2,numpy as np
from calcification_refinement_v2_common import component_pixel_groups
from calcification_refinement_common import baseline_masks
from evaluate_calcification_refinement_v3_union import counts
from calcification_refinement_v6_common import failures
from train_calcification_refinement_v2 import aggregate

def local_fail(metrics,baseline,fpbudget):
 result={}
 for arm,m in metrics.items():
  b=baseline[arm];f=fpbudget[arm]
  checks={'Dice_not_above_v3':m['positive_macro_Dice']>b['positive_macro_Dice'],'TP_below_v3':m['TP']>=b['TP'],'positive_overlap_below_v3':m['positive_images_with_TP']>=b['positive_images_with_TP'],'negative_FP_area_above_v4b':m['negative_FP_pixels']<=f['negative_FP_pixels'],'negative_FP_images_above_v4b':m['negative_images_with_FP']<=f['negative_images_with_FP']}
  checks.update({f'component_{k}_below_v3':m['components'][k]['any_overlap']>=b['components'][k]['any_overlap'] for k in b['components']})
  result[arm]=[k for k,v in checks.items() if not v]
 return result

def main():
 c=evaluation_context();esig=c['evaluation_signature'];assert (ROOT/'protocol.json').exists();dest=ROOT/'oof';dest.mkdir(parents=True,exist_ok=True)
 immutable(dest/'protocol.json',c['evaluation_protocol'])
 with (dest/'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lookup={r['image_id']:r for r in c['rows']};markers={}
  # All predictions must be committed and verified before loading any target.
  for fold in c['folds']:
   markerpath=ROOT/f'probabilities/fold_{fold["fold"]}/complete.json';m=verify(markerpath,c['signature']);markers[fold['fold']]=m
   assert m['image_ids']==fold['validation_ids'] and m['GT_read'] is False
   assert m['checkpoint_sha256']==c['jobs'][str(fold['fold'])]['checkpoint_sha256']
   expected={}
   for ident in fold['validation_ids']:
    pp=markerpath.parent/f'{ident}.npz';mp=pp.with_suffix('.json');meta=verify(mp,c['signature']);row=lookup[ident]
    assert meta['image_id']==ident and meta['fold']==fold['fold'] and meta['GT_read'] is False
    assert meta['native_sha256']==row['native_cache_sha256'] and meta['checkpoint_sha256']==m['checkpoint_sha256']
    assert meta['output_files']=={str(pp):checksum(pp)}
    expected[str(pp)]=checksum(pp);expected[str(mp)]=checksum(mp)
   assert m['output_files']==expected
  records=[];outputs={str(ROOT/f'probabilities/fold_{f["fold"]}/complete.json'):checksum(ROOT/f'probabilities/fold_{f["fold"]}/complete.json') for f in c['folds']}
  outputs[str(dest/'protocol.json')]=checksum(dest/'protocol.json')
  for fold in c['folds']:
   for ident in fold['validation_ids']:
    p=ROOT/f'probabilities/fold_{fold["fold"]}/{ident}.npz';cp=dest/'cases'/f'{ident}.json';row=lookup[ident]
    if cp.exists():
     case=read(cp);assert case['signature']==esig and case['probability_sha256']==checksum(p) and case['image_id']==ident and case['fold']==fold['fold']
    else:
     with np.load(p) as z:g={k:z[k] for k in z.files}
     shape=tuple(g['shape']);raw=np.load(row['native_cache_path'],mmap_mode='r')
     assert shape==raw.shape and g['probability'].shape==shape and g['probability'].dtype==np.float32
     assert np.isfinite(g['probability']).all() and ((g['probability']>=0)&(g['probability']<=1)).all()
     for arm in ARMS:
      _,original_fill,original_gate=baseline_masks(c['config']['development_baseline_root'],arm,ident)
      np.testing.assert_array_equal(np.unpackbits(g['fill_'+arm],count=int(np.prod(shape))).reshape(shape).astype(bool),original_fill)
      assert bool(g['gate_'+arm])==bool(original_gate['gate_pass'])
     assert checksum(row['target_path'])==row['target_sha256']
     with np.load(row['target_path']) as gt:gy,gx=gt['y'],gt['x']
     shape=tuple(g['shape']);groups=component_pixel_groups(gy,gx,shape) if len(gy) else []
     old=read(Path(c['jobs'][str(fold['fold'])]['path'])/'epochs/15/cases'/f'{ident}.json')
     case=dict(signature=esig,image_id=ident,fold=fold['fold'],probability_sha256=checksum(p),arms={})
     for arm in ARMS:
      fill=np.unpackbits(g['fill_'+arm],count=int(np.prod(shape))).reshape(shape).astype(bool);gate=bool(g['gate_'+arm]);values={}
      for policy in POLICIES:
       mask=apply(g['probability'],fill,gate,policy);m=counts(mask,gy,gx,fill,groups)
       if policy['id']=='v3':
        original=old['arms'][arm][c['protocol']['selected']['policy']]
        for key in m:assert m[key]==original[key],(ident,arm,key,m[key],original[key])
       n,labels=cv2.connectedComponents(mask.astype(np.uint8),connectivity=8)
       hit=set(labels[gy,gx].tolist())-{0};m['FP_components_without_GT']=int(n-1-len(hit));m['gate_rejected_GT_pixels']=0 if gate else len(gy)
       values[policy['id']]=m
      case['arms'][arm]=values
     immutable(cp,case)
    records.append(case);outputs[str(cp)]=checksum(cp);print(f'evaluated {len(records)}/124',flush=True)
  candidates=[]
  for policy in POLICIES:
   metrics={arm:aggregate([r['arms'][arm] for r in records],policy['id']) for arm in ARMS}
   for arm in ARMS:
    metrics[arm]['negative_FP_components']=sum(r['arms'][arm][policy['id']]['FP_components_without_GT'] for r in records if r['arms'][arm][policy['id']]['gt_pixels']==0)
    for key in ['FP_inside_boxes','FP_outside_boxes','gate_rejected_GT_pixels']:metrics[arm][key]=sum(r['arms'][arm][policy['id']][key] for r in records)
   candidates.append(dict(policy=policy,metrics=metrics))
  base=candidates[0]['metrics'];refs=c['protocol']['reference']
  for candidate in candidates:
   candidate['local_failures']=local_fail(candidate['metrics'],base,refs['v4b']);candidate['global_failures']=failures(candidate['metrics'],refs['v2'],refs['v4b'])
   candidate['delta_vs_v3']={a:{k:candidate['metrics'][a][k]-base[a][k] for k in ['positive_macro_Dice','TP','FP','negative_FP_pixels','negative_images_with_FP','TP_outside_boxes','FP_outside_boxes','negative_FP_components']} for a in ARMS}
  eligible=[v for v in candidates if v['policy']['kind']=='bounded' and not any(v['global_failures'].values())]
  eligible.sort(key=lambda v:(-np.mean([m['positive_macro_Dice'] for m in v['metrics'].values()]),np.mean([m['negative_FP_pixels'] for m in v['metrics'].values()]),v['policy']['radius']))
  immutable(dest/'complete.json',dict(complete=True,signature=esig,images=len(records),candidates=candidates,global_goal_success=bool(eligible),local_effect_success=any(v['policy']['kind']=='bounded' and not any(v['local_failures'].values()) for v in candidates),output_files=outputs))
  immutable(ROOT/'selection.json',dict(complete=True,signature=esig,selected=eligible[0] if eligible else None,oof_sha256=checksum(dest/'complete.json'),held_allowed=bool(eligible),continue_ideas2and3=True,output_files={str(dest/'complete.json'):checksum(dest/'complete.json')}))
if __name__=='__main__':main()

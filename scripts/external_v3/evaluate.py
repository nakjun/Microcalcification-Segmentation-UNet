"""Annotation-limited external endpoints after committed whole-image predictions."""
from common import ROOT,ARMS,read,checksum,verify,immutable,atomic_json,digest
import argparse,fcntl
from pathlib import Path
import cv2,numpy as np
ENDPOINTS=dict(point='np.rint ties-to-even;exact mask hit and Chebyshev radius1 native pixel;not clinical sensitivity',contour='union of valid Calcification contours;full-image Dice/IoU against incomplete contour annotations,not exhaustive pixel accuracy;point-only lesions not boundary GT',cbis='per-ROI any-overlap and area coverage;union-ROI coverage;outside-ROI predictions unannotated,not established false positives',burden='predicted pixels/image,fraction,image positivity,8-connected predicted components;all inputs including unknown labels',selection='none',negative_images='not established;missing XML and absent ROI are unknown')
def average(x):return float(np.mean(x)) if x else None
def metrics(mask,annotation,dataset):
 h,w=mask.shape;n,_=cv2.connectedComponents(mask.astype(np.uint8),connectivity=8);m=dict(predicted_pixels=int(mask.sum()),predicted_fraction=float(mask.mean()),predicted_components=int(n-1),image_predicted_positive=bool(mask.any()))
 if dataset=='inbreast':
  points=[]
  for p in annotation['points']:
   x,y=p['x'],p['y'];points.append(dict(roi_id=p['roi_id'],exact=bool(mask[y,x]),within1=bool(mask[max(0,y-1):min(h,y+2),max(0,x-1):min(w,x+2)].any())))
  m.update(points_total=len(points),points_exact=sum(p['exact'] for p in points),points_within1=sum(p['within1'] for p in points),point_hits=points)
  gt=np.zeros(mask.shape,np.uint8);contours=[]
  for r in annotation['contours']:
   xy=np.asarray(r['raster_xy'],np.int32);cv2.fillPoly(gt,[xy],1)
   # Per-contour retrieval is separate from dense union agreement.
   x0,y0=xy.min(0);x1,y1=xy.max(0);local=np.zeros((y1-y0+1,x1-x0+1),np.uint8);cv2.fillPoly(local,[xy-[x0,y0]],1)
   contours.append(dict(roi_id=r['roi_id'],any_overlap=bool(np.any(mask[y0:y1+1,x0:x1+1]&local.astype(bool)))))
  pixels=int(gt.sum());overlap=int(np.count_nonzero(mask&gt.astype(bool)));pred=m['predicted_pixels'];m.update(contours_total=len(contours),contours_any_overlap=sum(p['any_overlap'] for p in contours),contour_hits=contours,contour_union_pixels=pixels,contour_overlap_pixels=overlap,partial_contour_Dice=(2*overlap/(pixels+pred)) if pixels else None,partial_contour_IoU=overlap/(pixels+pred-overlap) if pixels else None,predicted_pixels_outside_contour_union_not_confirmed_FP=pred-overlap if pixels else None)
 else:
  rois=[];union=np.zeros(h*w,bool)
  for r in annotation['rois']:
   assert checksum(r['mask_path'])==r['mask_sha256']
   with np.load(r['mask_path']) as z:flat=z['flat'];assert tuple(z['shape'])==mask.shape
   union[flat]=True;hit=int(mask.ravel()[flat].sum());rois.append(dict(roi_id=r['roi_id'],roi_pixels=len(flat),overlap_pixels=hit,any_overlap=hit>0,coverage=hit/len(flat)))
  area=int(union.sum());hit=int(mask.ravel()[union].sum());m.update(ROIs=rois,ROIs_total=len(rois),ROIs_any_overlap=sum(r['any_overlap'] for r in rois),ROI_union_pixels=area,ROI_union_overlap_pixels=hit,ROI_union_coverage=hit/area if area else None,predicted_pixels_outside_ROI_union_unannotated=m['predicted_pixels']-hit if area else None)
 return m

def main():
 a=argparse.ArgumentParser();a.add_argument('--dataset',choices=['inbreast','cbis_test'],required=True);args=a.parse_args();ds=args.dataset;ip=ROOT/'inputs'/f'{ds}.json';pp=ROOT/ds/'protocol.json';rows=read(ip);inference_protocol=read(pp);sig=digest(inference_protocol);prediction_dir=ROOT/ds/'predictions';dest=ROOT/ds/'evaluation';dest.mkdir(parents=True,exist_ok=True)
 with (dest/'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);done=verify(prediction_dir/'complete.json',sig);assert done['image_ids']==[r['image_id'] for r in rows] and done['dataset']==ds and done['GT_read'] is False and done['smoke'] is False
  expected={}
  for r in rows:
   p=prediction_dir/f'{r["image_id"]}.npz';mp=p.with_suffix('.json');m=verify(mp,sig);assert m['image_id']==r['image_id'] and m['dataset']==ds and m['dicom_sha256']==r['dicom_sha256'] and m['GT_read'] is False and m['smoke'] is False and m['output_files']=={str(p):checksum(p)};expected[str(p)]=checksum(p);expected[str(mp)]=checksum(mp)
  assert done['output_files']==expected
  # Labels are only opened after the complete image-only prediction cohort is verified.
  ap=ROOT/'annotations'/f'{ds}.json';ann=read(ap);assert ann['complete'] and ann['dataset']==ds
  for p,h in ann['source_files'].items():assert checksum(p)==h
  alook={r['image_id']:r for r in ann['records']};assert set(alook)=={r['image_id'] for r in rows}
  inputs={str(p):checksum(p) for p in [ip,pp,ap,prediction_dir/'complete.json',Path(__file__),Path(__file__).with_name('annotations.py'),Path(__file__).with_name('test_evaluation.py')]};protocol=dict(inference_signature=sig,endpoints=ENDPOINTS,inputs=inputs);esig=digest(protocol);immutable(dest/'protocol.json',protocol);records=[];outputs=dict(inputs);outputs[str(dest/'protocol.json')]=checksum(dest/'protocol.json')
  gate_threshold=read('runs/calcification_gate_retinanet_sam_clean_v1/frozen/gate/thresholds.json')['primary']['threshold_logit']
  for row in rows:
   ident=row['image_id'];p=prediction_dir/f'{ident}.npz';cp=dest/'cases'/f'{ident}.json';meta=read(p.with_suffix('.json'));annotation=alook[ident]
   if cp.exists():case=read(cp);assert case['signature']==esig and case['probability_sha256']==checksum(p)
   else:
    with np.load(p) as z:g={k:z[k] for k in z.files}
    shape=(row['height'],row['width']);assert tuple(g['shape'])==shape;flat,prob=g['flat'],g['probability'];assert flat.dtype==np.int64 and prob.dtype==np.float32 and flat.shape==prob.shape and np.all(flat[1:]>flat[:-1]) and np.all((flat>=0)&(flat<np.prod(shape))) and np.isfinite(prob).all() and np.all((prob>=np.float32(.3))&(prob<=1))
    gate=bool(g['gate_pass']);assert np.isfinite(g['gate_logit']) and gate==bool(float(g['gate_logit'])>=gate_threshold) and meta['refinement_executed']==gate
    if not gate:assert len(flat)==0
    case=dict(signature=esig,image_id=ident,dataset=ds,annotation_status=annotation['status'],annotation_issues=annotation['issues'],probability_sha256=checksum(p),gate_pass=gate,refinement_executed=meta['refinement_executed'],arms={})
    for arm in ARMS:
     fill=np.zeros(shape,bool);boxes=g['boxes_'+arm];threshold=read(f'runs/calcification_gate_retinanet_sam_clean_v1/frozen/{arm}/snapshot.json')['threshold'];assert boxes.ndim==2 and boxes.shape[1]==5 and np.isfinite(boxes).all() and np.all(boxes[:,4]>=threshold)
     for x,y,u,v,_ in boxes:
      assert 0<=x<u<=shape[1] and 0<=y<v<=shape[0];fill[int(np.floor(y)):int(np.ceil(v)),int(np.floor(x)):int(np.ceil(u))]=True
     np.testing.assert_array_equal(fill,np.unpackbits(g['fill_'+arm],count=int(np.prod(shape))).reshape(shape).astype(bool))
     rebuilt=np.zeros(np.prod(shape),bool)
     if gate:rebuilt[flat[(fill.ravel()[flat])|(prob>=np.float32(.9))]]=True
     mask=np.unpackbits(g['mask_'+arm],count=int(np.prod(shape))).reshape(shape).astype(bool);np.testing.assert_array_equal(mask,rebuilt.reshape(shape));case['arms'][arm]=metrics(mask,annotation,ds)
    immutable(cp,case)
   records.append(case);outputs[str(cp)]=checksum(cp);print(ds,len(records),len(rows),flush=True)
  summary={}
  for arm in ARMS:
   vals=[r['arms'][arm] for r in records];m={k:sum(v[k] for v in vals) for k in ['predicted_pixels','predicted_components','image_predicted_positive']};m.update(mean_predicted_pixels=average([v['predicted_pixels'] for v in vals]),mean_predicted_fraction=average([v['predicted_fraction'] for v in vals]))
   if ds=='inbreast':
    for k in ['points_total','points_exact','points_within1','contours_total','contours_any_overlap','contour_union_pixels','contour_overlap_pixels']:m[k]=sum(v[k] for v in vals)
    m.update(point_exact_hit_rate=m['points_exact']/m['points_total'] if m['points_total'] else None,point_within1_hit_rate=m['points_within1']/m['points_total'] if m['points_total'] else None,contour_annotated_images=sum(v['contour_union_pixels']>0 for v in vals),predicted_pixels_outside_contour_union_not_confirmed_FP=sum(v['predicted_pixels_outside_contour_union_not_confirmed_FP'] or 0 for v in vals),partial_contour_macro_Dice=average([v['partial_contour_Dice'] for v in vals if v['partial_contour_Dice'] is not None]),partial_contour_macro_IoU=average([v['partial_contour_IoU'] for v in vals if v['partial_contour_IoU'] is not None]))
   else:
    for k in ['ROIs_total','ROIs_any_overlap','ROI_union_pixels','ROI_union_overlap_pixels']:m[k]=sum(v[k] for v in vals)
    m.update(ROI_any_overlap_rate=m['ROIs_any_overlap']/m['ROIs_total'] if m['ROIs_total'] else None,ROI_macro_coverage=average([r['coverage'] for v in vals for r in v['ROIs']]),ROI_union_micro_coverage=m['ROI_union_overlap_pixels']/m['ROI_union_pixels'] if m['ROI_union_pixels'] else None,predicted_pixels_outside_ROI_union_unannotated=sum(v['predicted_pixels_outside_ROI_union_unannotated'] or 0 for v in vals))
   summary[arm]=m
  immutable(dest/'complete.json',dict(complete=True,signature=esig,inference_signature=sig,dataset=ds,images=len(records),annotation_summary=ann['summary'],gate_pass_images=sum(r['gate_pass'] for r in records),metrics=summary,endpoints=ENDPOINTS,selection_performed=False,clinical_sensitivity_estimated=False,independent_sparse_mask_reconstruction=True,output_files=outputs))
if __name__=='__main__':cv2.setNumThreads(2);main()

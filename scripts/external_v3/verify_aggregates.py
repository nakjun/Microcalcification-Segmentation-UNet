"""Recount saved per-annotation and per-image original counts independently."""
from common import ROOT,read,verify,checksum,atomic_json
from pathlib import Path
import numpy as np

def main():
 result={}
 for ds in ['inbreast','cbis_test']:
  dest=ROOT/ds/'evaluation';done=verify(dest/'complete.json');cases=[read(p) for p in sorted((dest/'cases').glob('*.json'))];assert len(cases)==done['images'];assert sum(c['gate_pass'] for c in cases)==done['gate_pass_images']
  rows=read(ROOT/'inputs'/f'{ds}.json');assert {c['image_id'] for c in cases}=={r['image_id'] for r in rows}
  for arm in ['imagenet','mammo']:
   m=done['metrics'][arm];vs=[c['arms'][arm] for c in cases]
   for key in ['predicted_pixels','predicted_components','image_predicted_positive']:assert sum(v[key] for v in vs)==m[key]
   if ds=='inbreast':
    ps=[p for v in vs for p in v['point_hits']];cs=[p for v in vs for p in v['contour_hits']]
    assert len(ps)==m['points_total'] and sum(p['exact'] for p in ps)==m['points_exact'] and sum(p['within1'] for p in ps)==m['points_within1']
    assert len(cs)==m['contours_total'] and sum(p['any_overlap'] for p in cs)==m['contours_any_overlap']
    dice=[2*v['contour_overlap_pixels']/(v['contour_union_pixels']+v['predicted_pixels']) for v in vs if v['contour_union_pixels']]
    np.testing.assert_allclose(np.mean(dice),m['partial_contour_macro_Dice'],rtol=0,atol=1e-15)
   else:
    rois=[r for v in vs for r in v['ROIs']];assert len(rois)==m['ROIs_total'] and sum(r['overlap_pixels']>0 for r in rois)==m['ROIs_any_overlap']
    np.testing.assert_allclose(np.mean([r['overlap_pixels']/r['roi_pixels'] for r in rois]),m['ROI_macro_coverage'],rtol=0,atol=1e-15)
    for v in vs:assert v['predicted_pixels_outside_ROI_union_unannotated']==v['predicted_pixels']-v['ROI_union_overlap_pixels']
  result[ds]=dict(images=len(cases),complete_sha256=checksum(dest/'complete.json'),raw_count_reaggregation_exact=True,all_output_SHA_valid=True)
 atomic_json(Path('reports/external_v3/final_verification.json'),dict(complete=True,datasets=result,scope='fixed external annotation-limited evaluation;no selection/training'))
 print(result)
if __name__=='__main__':main()

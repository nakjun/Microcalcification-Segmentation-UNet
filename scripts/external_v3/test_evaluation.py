"""Independent synthetic endpoint checks, no model or external policy selection."""
import tempfile
from pathlib import Path
import numpy as np
from evaluate import metrics
from common import checksum

def test_points_contours_unknown():
 m=np.zeros((7,7),bool);m[0,0]=1;m[3,3]=1
 a=dict(points=[dict(roi_id='a',x=0,y=1),dict(roi_id='b',x=3,y=3),dict(roi_id='c',x=6,y=6)],contours=[dict(roi_id='d',raster_xy=[[2,2],[4,2],[4,4],[2,4]])])
 r=metrics(m,a,'inbreast');assert (r['points_exact'],r['points_within1'])==(1,2)
 assert r['contour_union_pixels']==9 and r['contour_overlap_pixels']==1
 assert r['partial_contour_Dice']==2/11 and r['partial_contour_IoU']==.1
 assert r['predicted_pixels_outside_contour_union_not_confirmed_FP']==1
 r=metrics(m,dict(points=[],contours=[]),'inbreast');assert r['partial_contour_Dice'] is None and r['points_total']==0 and r['predicted_pixels']==2

def test_cbis_coverage():
 with tempfile.TemporaryDirectory() as d:
  p=Path(d)/'roi.npz';np.savez_compressed(p,flat=np.array([0,1,7,8]),shape=np.array([7,7]))
  m=np.zeros((7,7),bool);m[0,0]=1;m[6,6]=1
  r=metrics(m,dict(rois=[dict(roi_id='r',mask_path=str(p),mask_sha256=checksum(p))]),'cbis_test')
  assert r['ROIs_any_overlap']==1 and r['ROIs'][0]['coverage']==.25
  assert r['ROI_union_coverage']==.25 and r['predicted_pixels_outside_ROI_union_unannotated']==1
  r=metrics(m,dict(rois=[]),'cbis_test');assert r['ROI_union_coverage'] is None and r['ROIs_total']==0

if __name__=='__main__':
 test_points_contours_unknown();test_cbis_coverage();print('2 independent endpoint tests passed')

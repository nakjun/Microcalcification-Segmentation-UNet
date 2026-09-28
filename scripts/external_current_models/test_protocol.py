"""CPU synthetic protocol checks; does not open external annotations or train."""
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import common
import evaluate
from common import read_mask, save_mask, sha, write
from infer import starts


def main():
    endpoint=evaluate.legacy_endpoints()
    mask=np.zeros((7,7),bool);mask[0,0]=1;mask[3,3]=1
    ann=dict(points=[dict(roi_id='a',x=0,y=1),dict(roi_id='b',x=3,y=3),dict(roi_id='c',x=6,y=6)],
             contours=[dict(roi_id='d',raster_xy=[[2,2],[4,2],[4,4],[2,4]])])
    values=endpoint.metrics(mask,ann,'inbreast')
    assert (values['points_exact'],values['points_within1'])==(1,2)
    assert values['partial_contour_Dice']==2/11 and values['partial_contour_IoU']==.1
    unknown=endpoint.metrics(mask,dict(points=[],contours=[]),'inbreast')
    assert unknown['partial_contour_Dice'] is None
    summary=evaluate.summarize([values,unknown],'inbreast')
    assert summary['points_total']==3 and summary['partial_contour_macro_Dice']==2/11
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        path=root/'mask.npz'
        probability=np.array([[.0,.49,.5,.9,1]],np.float32)
        save_mask(path,probability>=np.float32(.5))
        np.testing.assert_array_equal(read_mask(path,(1,5)),[[0,0,1,1,1]])
        roi=root/'roi.npz'
        np.savez_compressed(roi,flat=np.array([0,1,7,8]),shape=np.array([7,7]))
        values=endpoint.metrics(mask,dict(rois=[dict(roi_id='r',mask_path=str(roi),mask_sha256=sha(roi))]),'cbis_test')
        assert values['ROIs_any_overlap']==1 and values['ROI_union_coverage']==.25
        assert evaluate.summarize([values],'cbis_test')['ROI_macro_coverage']==.25
        with patch.object(common,'RUN',root):
            record={'condition':'deepmica','fold':0,'checkpoints':{'x':'hash'}}
            with patch.object(evaluate,'read',side_effect=[[],AssertionError('Annotation read reached')]) as read_spy:
                try:
                    evaluate.evaluate('inbreast',record,'signature',{'inventory':{'inbreast':'inventory'}})
                except FileNotFoundError:
                    assert read_spy.call_count == 1
                else:
                    raise AssertionError('Incomplete cohort accepted before annotation access')
            try:
                evaluate.verified_prediction_cohort('inbreast',record,'signature',[])
            except FileNotFoundError:
                pass
            else:
                raise AssertionError('Incomplete cohort accepted')
        first=root/'immutable.json';write(first,{'value':1})
        try:
            write(first,{'value':2})
        except AssertionError:
            pass
        else:
            raise AssertionError('Immutable artifact overwrite accepted')
    for length in (1,255,256,257,500,4096):
        coverage=np.zeros(length,bool)
        for start in starts(length):
            coverage[start:start+256]=True
        assert coverage.all()
    print('PASS: retained point/partial-contour/ROI endpoints; unknown labels; p=.5 packed mask; tile edges; missing cohort rejection; immutable output')


if __name__=='__main__':
    main()

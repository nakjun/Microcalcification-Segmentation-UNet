import tempfile
from pathlib import Path
import numpy as np
from segmentation_metrics import counts, summarize, evaluate

p=np.array([[.8,.8],[.1,.1]])
g=np.array([[1,0],[1,0]])
r=counts(p,g,.5)
assert [r[k] for k in ('TP','FP','FN','TN')]==[1,1,1,1]
assert r['dice']==.5 and r['iou']==1/3
n=counts(p,np.zeros((2,2)),.5)
s=summarize([r,n])
assert s['positive_macro_dice']==.5 and s['positive_pixel_recall']==.5
assert s['negative_fp_pixels']==2 and s['negative_fp_images']==1
assert summarize([n])['positive_macro_dice'] is None
with tempfile.TemporaryDirectory() as folder:
    out=Path(folder)/'eval'
    row={'image':'image_only','mask':'mask_only','patient_id':'test','split':'test'}
    def predict(path):
        assert path=='image_only'
        return p
    evaluate([row],predict,lambda path:g,out,.5,{'device':'cpu','scope':'synthetic'})
    assert (out/'evaluation_complete.json').exists()
print('PASS: exact confusion counts, positive-only macro metrics, negative FP, GT-free inference callback, artifacts')

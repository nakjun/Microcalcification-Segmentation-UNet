import cv2
import numpy as np
from scipy import ndimage
POLICIES=[dict(id='v3',kind='v3',radius=0),*[dict(id=f'{kind}_r{r}',kind=kind,radius=r) for kind in ['bounded','dilation'] for r in [1,2,4]],dict(id='global_hysteresis',kind='global_hysteresis',radius=0)]
def apply(prob,fill,gate,policy):
 assert prob.dtype==np.float32 and prob.shape==fill.shape and np.isfinite(prob).all()
 if not gate:return np.zeros(prob.shape,bool)
 seed=(fill&(prob>=np.float32(.3)))|(~fill&(prob>=np.float32(.9)))
 if policy['kind']=='v3':return seed
 low=prob>=np.float32(.3)
 if policy['kind']=='global_hysteresis':
  return ndimage.binary_propagation(prob>=np.float32(.9),structure=np.ones((3,3),bool),mask=low)
 out=seed.copy()
 for _ in range(policy['radius']):
  grown=cv2.dilate(out.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
  out=out|(grown&low) if policy['kind']=='bounded' else grown
 assert np.all(out[seed]);return out

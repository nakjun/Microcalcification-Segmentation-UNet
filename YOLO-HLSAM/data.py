"""Explicit split manifest and fixed uint8 RGB image input contract."""
import json
from pathlib import Path
import numpy as np
from PIL import Image


def read_image(path):
    with Image.open(path) as im:
        if im.mode not in ('L','RGB'):
            raise ValueError(f'{path}: convert to fixed-normalization uint8 L/RGB first; got {im.mode}')
        return np.asarray(im.convert('RGB')).copy()


def read_manifest(path):
    path=Path(path).resolve()
    rows=json.loads(path.read_text())
    if not isinstance(rows,list) or not rows:
        raise ValueError('Manifest must be a nonempty JSON list')
    ids=set(); patients={}; images=set(); path_splits={}
    for row in rows:
        for key in ('id','patient_id','split','image','mask'):
            if not isinstance(row.get(key),str) or not row[key]:
                raise ValueError(f'Missing/non-string {key}')
        if row['split'] not in ('train','val','test') or row['id'] in ids:
            raise ValueError('Invalid split or duplicate ID')
        ids.add(row['id'])
        old=patients.setdefault(row['patient_id'],row['split'])
        if old!=row['split']:
            raise ValueError('Patient overlap between splits')
        for key in ('image','mask'):
            p=(path.parent/row[key]).resolve()
            if not p.is_file():raise FileNotFoundError(p)
            row[key]=str(p)
            if p in path_splits and path_splits[p] != row['split']:
                raise ValueError('Image/mask path overlap between splits')
            path_splits[p]=row['split']
        if row['image']==row['mask']:raise ValueError('Image and mask must differ')
        if row['image'] in images:raise ValueError('Repeated image path')
        images.add(row['image'])
    return rows


def sample(row):
    image=read_image(row['image'])
    with Image.open(row['mask']) as im:
        mask=np.asarray(im)
    if mask.ndim!=2 or mask.shape!=image.shape[:2]:
        raise ValueError('Mask must be single-channel and match native image dimensions')
    mask=(mask>0).astype(np.float32)
    # Explicit cluster boxes preferred; otherwise one box enclosing all lesions.
    if 'boxes' in row:
        boxes=np.asarray(row['boxes'],dtype=np.float32).reshape(-1,4)
    elif mask.any():
        y,x=np.nonzero(mask)
        boxes=np.array([[x.min(),y.min(),x.max()+1,y.max()+1]],dtype=np.float32)
    else:boxes=np.empty((0,4),dtype=np.float32)
    h,w=mask.shape
    if not np.isfinite(boxes).all() or np.any(boxes[:,2:]<=boxes[:,:2]) or np.any(boxes<0) or np.any(boxes[:,[0,2]]>w) or np.any(boxes[:,[1,3]]>h):
        raise ValueError('Invalid native xyxy boxes')
    if bool(len(boxes))!=bool(mask.any()):
        raise ValueError('Positive masks require boxes; negative masks require empty boxes')
    for box in boxes:
        x1,y1=np.floor(box[:2]).astype(int)
        x2,y2=np.ceil(box[2:]).astype(int)
        if not mask[y1:y2,x1:x2].any():raise ValueError('Box contains no positive mask pixels')
    return image,mask,boxes

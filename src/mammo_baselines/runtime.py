import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

BASE=Path('runs/vindr_baselines').resolve()
DATA=BASE/'data'
PROTOCOL=json.loads(Path('configs/vindr_training_protocol.json').read_text())


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)


def rows_for(split):
    return [r for r in json.loads((DATA/'manifest.json').read_text()) if r['split']==split]


def checksum(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


class BoxMaskDataset(Dataset):
    def __init__(self,rows,augment=False):self.rows=rows;self.augment=augment
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        r=self.rows[index];im=cv2.imread(r['cache_path'],cv2.IMREAD_GRAYSCALE)
        if im is None:raise RuntimeError(r['cache_path'])
        mask=np.zeros((1024,1024),np.float32);valid=np.zeros_like(mask)
        px,py,nw,nh=r['pad_x'],r['pad_y'],r['valid_width'],r['valid_height']
        valid[py:py+nh,px:px+nw]=1
        for x,y,u,v in r['boxes']:
            a,b=int(np.floor(x*r['sx']+px)),int(np.floor(y*r['sy']+py))
            c,d=int(np.ceil(u*r['sx']+px)),int(np.ceil(v*r['sy']+py))
            mask[max(py,b):min(py+nh,d),max(px,a):min(px+nw,c)]=1
        if self.augment:
            scale=random.uniform(1-PROTOCOL['augmentation_scale'],1+PROTOCOL['augmentation_scale'])
            shift=PROTOCOL['augmentation_translate']*1024
            matrix=np.array([[scale,0,(1-scale)*512+random.uniform(-shift,shift)],
                             [0,scale,(1-scale)*512+random.uniform(-shift,shift)]],dtype=np.float32)
            im=cv2.warpAffine(im,matrix,(1024,1024),flags=cv2.INTER_LINEAR,borderValue=114)
            mask=cv2.warpAffine(mask,matrix,(1024,1024),flags=cv2.INTER_NEAREST,borderValue=0)
            valid=cv2.warpAffine(valid,matrix,(1024,1024),flags=cv2.INTER_NEAREST,borderValue=0)
        if self.augment and random.random()<.5:im=im[:,::-1];mask=mask[:,::-1];valid=valid[:,::-1]
        return torch.from_numpy(im.copy()).unsqueeze(0),torch.from_numpy(mask.copy()).unsqueeze(0),torch.from_numpy(valid.copy()).unsqueeze(0),index


def normalize_unet(images,device):
    x=images.to(device,non_blocking=True).float().div_(255).expand(-1,3,-1,-1)
    mean=x.new_tensor([.485,.456,.406]).view(1,3,1,1);std=x.new_tensor([.229,.224,.225]).view(1,3,1,1)
    return (x-mean)/std


def component_boxes(prob,row,threshold):
    px,py,nw,nh=row['pad_x'],row['pad_y'],row['valid_width'],row['valid_height']
    binary=np.zeros(prob.shape,bool);binary[py:py+nh,px:px+nw]=prob[py:py+nh,px:px+nw]>=threshold
    labels,n=ndimage.label(binary,structure=np.ones((3,3)))
    if not n:return np.empty((0,5))
    areas=np.bincount(labels.ravel());sums=np.bincount(labels.ravel(),weights=prob.ravel())
    scores=sums[1:]/areas[1:];top=np.argsort(-scores,kind='stable')[:100]
    objects=ndimage.find_objects(labels)
    return np.asarray([[objects[i][1].start,objects[i][0].start,objects[i][1].stop,objects[i][0].stop,scores[i]] for i in top],dtype=float)

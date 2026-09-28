"""Two-stage local training adapter; separate from the paper's original recipe.

Only train-split records are used. SAM stays frozen; HQ decoder trains using
training-mask box prompts. Detector uses the official one-class CBAM3 network
and anchor-assignment loss (not upstream OTA training). No GT-box validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from data import read_manifest, sample
from model import Segmentor, build_detector, read_state, file_sha256
from upstream import SOURCE


def detector_sample(image,boxes,size,device):
    from utils.datasets import letterbox
    image,ratio,pad=letterbox(image,new_shape=size,auto=False)
    x=torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))).float()[None].to(device)/255
    b=torch.as_tensor(boxes,device=device).clone()
    if len(b):
        b=b*b.new_tensor([ratio[0],ratio[1],ratio[0],ratio[1]])+b.new_tensor([pad[0],pad[1],pad[0],pad[1]])
        center=(b[:,:2]+b[:,2:])/2/size
        wh=(b[:,2:]-b[:,:2])/size
        targets=torch.cat([torch.zeros(len(b),2,device=device),center,wh],1)
    else:targets=torch.zeros(0,6,device=device)
    return x,targets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['detector','segmentor'],required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,required=True)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--detector-size',type=int,default=640)
    p.add_argument('--model-type',choices=['vit_b','vit_l','vit_h'],default='vit_b')
    p.add_argument('--sam-checkpoint')
    p.add_argument('--init-checkpoint',help='Tensor state_dict: detector or decoder, according to stage')
    a=p.parse_args()
    if a.epochs<1 or (not np.isfinite(a.learning_rate) or a.learning_rate<=0) or a.detector_size<64 or a.detector_size%32:p.error('Invalid training settings')
    if a.stage=='segmentor' and not a.sam_checkpoint:p.error('segmentor requires --sam-checkpoint')
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    rows=read_manifest(a.manifest)
    rows=[r for r in rows if r['split']=='train']
    if not rows:raise ValueError('No training records')
    # Validate images/annotations before allocating a model or writing outputs.
    positive=0
    for row in rows:
        _,_,boxes=sample(row)
        positive+=bool(len(boxes))
    if not positive:raise ValueError('Training requires positive images')
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);np.random.seed(a.seed)
    if a.stage=='detector':
        from utils.loss import ComputeLoss
        model=build_detector().to(a.device).train()
        if a.init_checkpoint:model.load_state_dict(read_state(a.init_checkpoint,stage="detector"),strict=True)
        model.hyp=yaml.safe_load((SOURCE/'data/hyp.scratch.p5.yaml').read_text())
        model.gr=1.0
        criterion=ComputeLoss(model)
    else:
        model=Segmentor(a.model_type,a.sam_checkpoint,a.init_checkpoint).to(a.device).train()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.learning_rate)
    out.mkdir(parents=True,exist_ok=False)
    metadata={'settings':vars(a),'scope':'local adapter training; not original paper recipe',
              'sam_sha256':file_sha256(a.sam_checkpoint) if a.stage=='segmentor' else None,
              'manifest_sha256':hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
              'train_ids':[r['id'] for r in rows], 'train_patients':sorted({r['patient_id'] for r in rows})}
    (out/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for epoch in range(1,a.epochs+1):
        total=0.;updates=0;skipped=0
        for index in torch.randperm(len(rows)).tolist():
            image,mask,boxes=sample(rows[index])
            optimizer.zero_grad(set_to_none=True)
            if a.stage=='detector':
                x,targets=detector_sample(image,boxes,a.detector_size,a.device)
                loss,_=criterion(model(x),targets)
            else:
                if not len(boxes):
                    skipped+=1;continue
                # One training annotation box at a time. GT masks only in training.
                box=boxes[np.random.randint(len(boxes))]
                logits=model.logits(image,box[None])
                target_mask=np.zeros_like(mask)
                x1,y1=np.floor(box[:2]).astype(int)
                x2,y2=np.ceil(box[2:]).astype(int)
                target_mask[y1:y2,x1:x2]=mask[y1:y2,x1:x2]
                if not target_mask.any():raise ValueError('Training box contains no positive pixels')
                target=torch.as_tensor(target_mask,device=a.device)[None,None]
                probability=logits.sigmoid()
                dice=1-(2*(probability*target).sum()+1)/(probability.sum()+target.sum()+1)
                loss=F.binary_cross_entropy_with_logits(logits,target)+dice
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError('Non-finite gradient')
            optimizer.step();total+=loss.detach().item();updates+=1
        state=model.state_dict() if a.stage=='detector' else model.decoder.state_dict()
        torch.save({'state_dict':state,'epoch':epoch,'metadata':metadata},out/f'epoch_{epoch:03d}.pt')
        record={'epoch':epoch,'updates':updates,'negative_images_skipped':skipped,'training_loss':total/max(updates,1)}
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    (out/'training_complete.json').write_text(json.dumps({'epochs':a.epochs,'performance_evaluation':False})+'\n')

if __name__=='__main__':main()

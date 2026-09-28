import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
p=argparse.ArgumentParser();p.add_argument('--model',choices=['deepmica','yolo'],required=True);p.add_argument('--device',default='cuda:1');a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(42);assert torch.cuda.is_available()
folder=ROOT/'runs/additional_baselines_20260926/data/fold_0/patches'
imagepath=next(p for p in sorted(folder.glob('f0_train_0000*.png')) if not p.stem.endswith('_mask'))
maskpath=imagepath.with_name(imagepath.stem+'_mask.png')
if a.model=='deepmica':
 sys.path.insert(0,str(ROOT/'DeepMiCa'));from pipeline import build_model,image_tensor,read_array,topk_loss
 model=build_model(1).to(a.device).train();x=image_tensor(imagepath,1,255.)[None].repeat(16,1,1,1).to(a.device);y=torch.from_numpy((read_array(maskpath)>0).astype(np.float32))[None,None].repeat(16,1,1,1).to(a.device)
 loss=topk_loss(model(x),y);loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None)
 optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.99);optimizer.step()
else:
 sys.path.insert(0,str(ROOT/'YOLO-HLSAM'));from model import build_detector,Segmentor;from train import detector_sample;from data import sample;from upstream import SOURCE
 import yaml
 from utils.loss import ComputeLoss
 image,mask,boxes=sample({'image':str(imagepath),'mask':str(maskpath)})
 model=build_detector().to(a.device).train();model.hyp=yaml.safe_load((SOURCE/'data/hyp.scratch.p5.yaml').read_text());model.gr=1.
 x,y=detector_sample(image,boxes,640,a.device);loss,_=ComputeLoss(model)(model(x),y);loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None)
 torch.optim.AdamW(model.parameters(),lr=1e-4).step();del model,x,y,loss;torch.cuda.empty_cache()
 model=Segmentor('vit_b',str(ROOT/'runs/mammo_hypothesis/weights/sam_vit_b.pth')).to(a.device).train()
 logits=model.logits(image,boxes[:1]);target=torch.as_tensor(mask,device=a.device)[None,None];prob=logits.sigmoid();loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)+1-(2*(prob*target).sum()+1)/(prob.sum()+target.sum()+1)
 loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None)
 torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],lr=1e-4).step()
result={'model':a.model,'device':a.device,'GPU':torch.cuda.get_device_name(a.device),'forward_backward_update_finite':True,'scope':'training patch GPU smoke only; no dataset performance','peak_memory_bytes':torch.cuda.max_memory_allocated(a.device)}
out=ROOT/'runs/additional_baselines_20260926/preflight';out.mkdir(parents=True,exist_ok=True);(out/f'{a.model}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

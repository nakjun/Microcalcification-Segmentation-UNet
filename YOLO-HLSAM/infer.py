"""Image-only detection -> segmentation, no ground-truth inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from data import read_image
from model import Detector, Segmentor, predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',required=True)
    p.add_argument('--detector-checkpoint',required=True)
    p.add_argument('--sam-checkpoint',required=True)
    p.add_argument('--decoder-checkpoint',required=True)
    p.add_argument('--model-type',choices=['vit_b','vit_l','vit_h'],default='vit_b')
    p.add_argument('--output',required=True,help='New output directory')
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--detector-size',type=int,default=640)
    p.add_argument('--confidence',type=float,default=.25)
    p.add_argument('--iou',type=float,default=.45)
    p.add_argument('--threshold',type=float,default=.5)
    p.add_argument('--max-detections',type=int,default=100)
    a=p.parse_args()
    if not 0<a.threshold<1:p.error('threshold must be between 0 and 1')
    for checkpoint in (a.detector_checkpoint,a.sam_checkpoint,a.decoder_checkpoint):
        if not Path(checkpoint).is_file():p.error(f'Missing checkpoint: {checkpoint}')
    torch.set_num_threads(a.threads)
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    image=read_image(a.image)
    detector=Detector(a.detector_checkpoint,a.device,a.detector_size,a.confidence,a.iou,a.max_detections)
    segmentor=Segmentor(a.model_type,a.sam_checkpoint,a.decoder_checkpoint).to(a.device).eval()
    prob,boxes,scores=predict(image,detector,segmentor)
    out.mkdir(parents=True,exist_ok=False)
    np.save(out/'probability.npy',prob)
    Image.fromarray((prob>=a.threshold).astype(np.uint8)*255).save(out/'mask.png')
    (out/'prediction.json').write_text(json.dumps({'settings':vars(a),'shape':list(prob.shape),
        'boxes_xyxy':boxes.tolist(),'scores':scores.tolist(),'prompt_source':'predicted_boxes',
        'ground_truth_used':False,'implementation':'official repository decoder + local detector integration'},indent=2)+'\n')

if __name__=='__main__':main()

"""GPU-capable native image evaluation; only detector predictions form prompts."""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image
from data import read_manifest, read_image
from model import Detector, Segmentor, predict as predict_image, file_sha256
sys.path.insert(1,str(Path(__file__).resolve().parents[1]/'scripts/additional_baselines'))
from segmentation_metrics import evaluate, require_device


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--split',choices=['val','test'],required=True)
    p.add_argument('--detector-checkpoint',required=True)
    p.add_argument('--sam-checkpoint',required=True)
    p.add_argument('--decoder-checkpoint',required=True)
    p.add_argument('--model-type',choices=['vit_b','vit_l','vit_h'],default='vit_b')
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--detector-size',type=int,default=640)
    p.add_argument('--confidence',type=float,default=.25)
    p.add_argument('--iou',type=float,default=.45)
    p.add_argument('--max-detections',type=int,default=100)
    p.add_argument('--threshold',type=float,default=.5)
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();require_device(a.device);torch.set_num_threads(a.threads)
    rows=[r for r in read_manifest(a.manifest) if r['split']==a.split]
    detector=Detector(a.detector_checkpoint,a.device,a.detector_size,a.confidence,a.iou,a.max_detections)
    segmentor=Segmentor(a.model_type,a.sam_checkpoint,a.decoder_checkpoint).to(a.device).eval()
    def predict(path):return predict_image(read_image(path),detector,segmentor)[0]
    def load_mask(path):
        with Image.open(path) as im:return np.asarray(im).copy()
    settings=vars(a)|{'manifest_sha256':hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
                     'checkpoint_sha256':{k:file_sha256(getattr(a,k)) for k in ('detector_checkpoint','sam_checkpoint','decoder_checkpoint')},
                     'prompt_source':'predicted_boxes'}
    evaluate(rows,predict,load_mask,a.output,a.threshold,settings)

if __name__=='__main__':main()

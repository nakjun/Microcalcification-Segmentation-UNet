"""GPU-capable native-resolution evaluation of an explicit val/test CSV split."""
import argparse
import hashlib
from pathlib import Path
import sys
import torch
from pipeline import read_manifest, load_checkpoint, image_tensor, tiled_predict, read_array
sys.path.insert(1,str(Path(__file__).resolve().parents[1]/'scripts/additional_baselines'))
from segmentation_metrics import evaluate, require_device


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--split',choices=['val','test'],required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--stride',type=int,default=128)
    p.add_argument('--threshold',type=float,default=.5)
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();require_device(a.device);torch.set_num_threads(a.threads)
    rows=[r for r in read_manifest(a.manifest,require_training=False) if r['split']==a.split]
    model,meta=load_checkpoint(a.checkpoint,a.device)
    def predict(path):
        x=image_tensor(path,meta['channels'],meta['scale'])
        return tiled_predict(model,x,meta['patch_size'],a.stride,a.device)
    settings=vars(a)|{'manifest_sha256':hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),'model_metadata':meta}
    settings['checkpoint_sha256']=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
    evaluate(rows,predict,read_array,a.output,a.threshold,settings)

if __name__=='__main__':main()

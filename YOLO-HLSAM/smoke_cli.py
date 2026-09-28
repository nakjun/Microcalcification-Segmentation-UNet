"""Synthetic CLI integration smoke; uses temporary random weights, no real data."""
import sys, tempfile, json, subprocess, os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
root=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(root/'YOLO-HLSAM'))
from model import Segmentor
python=str(root/'.venv/bin/python')
torch.set_num_threads(4)
with tempfile.TemporaryDirectory(prefix='hlsam_cli_') as directory:
 d=Path(directory)
 image=np.zeros((64,96,3),np.uint8); image[20:25,30:36]=255
 mask=image[:,:,0]
 Image.fromarray(image).save(d/'image.png');Image.fromarray(mask).save(d/'mask.png')
 (d/'manifest.json').write_text(json.dumps([{'id':'synthetic','patient_id':'synthetic','split':'train','image':'image.png','mask':'mask.png'}]))
 model=Segmentor();torch.save(model.sam.state_dict(),d/'sam.pt'); del model
 def run(args):
  result=subprocess.run([python,*args],cwd=root,env=dict(os.environ,MPLCONFIGDIR='/tmp/hlsam_matplotlib'),capture_output=True,text=True)
  if result.returncode:raise RuntimeError(result.stdout+'\n'+result.stderr)
 for stage in ('detector','segmentor'):
  args=['YOLO-HLSAM/train.py','--stage',stage,'--manifest',str(d/'manifest.json'),'--output',str(d/stage),'--epochs','1','--detector-size','128']
  if stage=='segmentor':args+=['--sam-checkpoint',str(d/'sam.pt')]
  run(args)
  assert (d/stage/'training_complete.json').exists()
 run(['YOLO-HLSAM/infer.py','--image',str(d/'image.png'),'--detector-checkpoint',str(d/'detector/epoch_001.pt'),'--sam-checkpoint',str(d/'sam.pt'),'--decoder-checkpoint',str(d/'segmentor/epoch_001.pt'),'--output',str(d/'prediction'),'--detector-size','128'])
 prob=np.load(d/'prediction/probability.npy');assert prob.shape==(64,96) and np.isfinite(prob).all()
 report=json.loads((d/'prediction/prediction.json').read_text());assert report['ground_truth_used'] is False
 Image.fromarray(image[:,::-1].copy()).save(d/'test_image.png');Image.fromarray(mask[:,::-1].copy()).save(d/'test_mask.png')
 (d/'test_manifest.json').write_text(json.dumps([{'id':'synthetic_test','patient_id':'synthetic_test','split':'test','image':'test_image.png','mask':'test_mask.png'}]))
 run(['YOLO-HLSAM/evaluate.py','--manifest',str(d/'test_manifest.json'),'--split','test','--detector-checkpoint',str(d/'detector/epoch_001.pt'),'--sam-checkpoint',str(d/'sam.pt'),'--decoder-checkpoint',str(d/'segmentor/epoch_001.pt'),'--output',str(d/'evaluation'),'--detector-size','128','--device','cpu'])
 assert (d/'evaluation/evaluation_complete.json').exists()
 metrics=json.loads((d/'evaluation/metrics.json').read_text())
 assert metrics['summary']['positive_images']==1 and metrics['summary']['negative_images']==0
 print('PASS: synthetic train, checkpoint reload, image-only inference, and evaluation CLI')
 (root/'YOLO-HLSAM/cli_smoke_result.json').write_text(json.dumps({'scope':'synthetic CPU only; random SAM initialization; no dataset performance', 'detector_one_epoch_cli':True,'segmentor_one_epoch_cli':True,'strict_checkpoint_reload':True,'image_only_inference_cli':True,'evaluation_cli':True,'native_output_shape':[64,96]},indent=2)+'\n')

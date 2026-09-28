"""Synthetic CPU evaluation CLI check; no real dataset performance."""
import sys,tempfile,subprocess,csv,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'DeepMiCa'))
from pipeline import build_model,save_checkpoint,metadata
torch.set_num_threads(4)
with tempfile.TemporaryDirectory() as directory:
 d=Path(directory)
 image=np.zeros((32,32),np.uint8);image[10:12,15:18]=255
 Image.fromarray(image).save(d/'image.png');Image.fromarray(image).save(d/'mask.png')
 with (d/'test.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['image','mask','patient_id','split']);w.writerow(['image.png','mask.png','test','test'])
 save_checkpoint(d/'model.pt',build_model(1),metadata(1,255.,32),0)
 subprocess.run([str(root/'.venv/bin/python'),'DeepMiCa/evaluate.py','--manifest',str(d/'test.csv'),'--split','test','--checkpoint',str(d/'model.pt'),'--output',str(d/'evaluation'),'--device','cpu','--stride','16'],check=True)
 result=json.loads((d/'evaluation/metrics.json').read_text());assert result['summary']['positive_images']==1
 assert (d/'evaluation/evaluation_complete.json').exists()
print('PASS: DeepMiCa test-only CSV evaluation CLI, native predictions, metric artifacts (synthetic CPU)')

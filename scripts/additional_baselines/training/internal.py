"""Fixed-fold native OOF/held32 inference, then metrics; no external work."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(1,str(ROOT/'scripts/additional_baselines'))
from segmentation_metrics import counts,summarize
p=argparse.ArgumentParser();p.add_argument('--model',choices=['deepmica','yolo'],required=True);p.add_argument('--device',default='cuda:1');a=p.parse_args()
torch.set_num_threads(4)
run=ROOT/'runs/additional_baselines_20260926';folds=json.loads((ROOT/'runs/calcification_refinement_v5_seed42/folds.json').read_text())
rows=json.loads((ROOT/folds['source_manifest']).read_text());byid={r['image_id']:r for r in rows};held=[r for r in rows if r['role']!='train'];assert len(held)==32
base=run/a.model;assert (base/'training_complete.json').exists()
all_oof=[];held_folds=[];hashes={};input_hashes={};outputs={}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
for fold in folds['folds']:
 fid=fold['fold'];dest=base/'internal'/f'fold_{fid}';dest.mkdir(parents=True,exist_ok=False)
 if a.model=='deepmica':
  sys.path.insert(0,str(ROOT/'DeepMiCa'));from pipeline import load_checkpoint,image_tensor,tiled_predict
  cp=base/f'fold_{fid}'/'last.pt';model,meta=load_checkpoint(cp,a.device)
  def predict(row):return tiled_predict(model,image_tensor(ROOT/row['native_cache_path'],meta['channels'],meta['scale']),256,192,a.device)
  cps=[cp]
 else:
  sys.path.insert(0,str(ROOT/'YOLO-HLSAM'));from model import Detector,Segmentor,predict as patch_predict
  dc=base/f'fold_{fid}'/'detector'/'epoch_016.pt';sc=base/f'fold_{fid}'/'segmentor'/'epoch_016.pt';sam=ROOT/'runs/mammo_hypothesis/weights/sam_vit_b.pth'
  detector=Detector(dc,a.device,640,.25,.45,100);segmentor=Segmentor('vit_b',sam,sc).to(a.device).eval()
  def predict(row):
   image=np.load(ROOT/row['native_cache_path'],allow_pickle=False);h,w=image.shape;prob=np.zeros((h,w),np.float32);num=np.zeros((h,w),np.uint16)
   def starts(n):
    values=list(range(0,max(n-256,0)+1,192))
    if values[-1]!=max(n-256,0):values.append(max(n-256,0))
    return values
   for y in starts(h):
    for x in starts(w):
     tile=np.repeat(image[y:y+256,x:x+256,None],3,axis=2)
     q,_,_=patch_predict(tile,detector,segmentor);prob[y:y+q.shape[0],x:x+q.shape[1]]+=q;num[y:y+q.shape[0],x:x+q.shape[1]]+=1
   assert np.all(num>0);return prob/num
  cps=[dc,sc,sam]
 for cp in cps:
  h=hashlib.sha256()
  with cp.open('rb') as f:
   for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
  hashes[str(cp)]=h.hexdigest()
 for scope,cohort in [('oof',[byid[i] for i in fold['validation_ids']]),('held32',held)]:
  output=dest/scope;output.mkdir();metadata=[]
  for row in cohort:
   source=ROOT/row['native_cache_path']
   if str(source) not in input_hashes:
    assert sha(source)==row['native_cache_sha256'];input_hashes[str(source)]=row['native_cache_sha256']
   probability=predict(row);assert probability.shape==(row['height'],row['width']) and np.isfinite(probability).all()
   binary=(probability>=.5).astype(np.uint8)
   np.savez_compressed(output/(row['image_id']+'.npz'),packed=np.packbits(binary),shape=np.array(binary.shape))
   metadata.append({'image_id':row['image_id'],'study_id':row['study_id']})
   print(a.model,'fold',fid,scope,'inference',len(metadata),'/',len(cohort),flush=True)
  (output/'inference_complete.json').write_text(json.dumps({'images':len(cohort),'checkpoint_sha256':hashes,'threshold':.5,'GT_read_during_inference':False})+'\n')
  records=[]
  # GT is opened only after all predictions for this fold/scope are saved.
  for row in cohort:
   with np.load(output/(row['image_id']+'.npz')) as z:
    shape=tuple(z['shape']);binary=np.unpackbits(z['packed'],count=int(np.prod(shape))).reshape(shape)
   targetpath=ROOT/row['target_path'];assert sha(targetpath)==row['target_sha256']
   with np.load(targetpath) as z:
    gt=np.zeros(shape,np.uint8);gt[z['y'],z['x']]=1
   record=counts(binary.astype(np.float32),gt,.5)|{'image_id':row['image_id'],'study_id':row['study_id'],'fold':fid};records.append(record)
  (output/'per_image.json').write_text(json.dumps(records,indent=2)+'\n');summary=summarize(records)
  (output/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
  for artifact in output.iterdir():
   if artifact.is_file():outputs[str(artifact)]=sha(artifact)
  if scope=='oof':all_oof.extend(records)
  else:held_folds.append(summary)
 if a.model=='deepmica':del model
 else:del detector,segmentor
 torch.cuda.empty_cache()
assert len(all_oof)==124 and len({r['image_id'] for r in all_oof})==124
keys=['positive_macro_dice','positive_macro_iou','positive_pixel_recall','negative_fp_pixels','negative_fp_images']
summary={'scope':'single seed42; OOF124 and previously exposed development held32 kept separate','oof':summarize(all_oof),'held32_folds':held_folds,'held32_mean_sd':{k:{'mean':float(np.mean([r[k] for r in held_folds])),'sample_sd':float(np.std([r[k] for r in held_folds],ddof=1))} for k in keys},'checkpoint_sha256':hashes,'input_sha256':input_hashes,'output_sha256':outputs,'external_started':False,'review_pass':False}
(base/'internal_complete.json').write_text(json.dumps(summary,indent=2)+'\n')

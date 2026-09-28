"""Materialize fixed native patches inside the original development folds only."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[3]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
 foldpath=ROOT/'runs/calcification_refinement_v5_seed42/folds.json';folds=json.loads(foldpath.read_text());manifest=ROOT/folds['source_manifest']
 assert sha(manifest)==folds['source_manifest_sha256']
 rows=json.loads(manifest.read_text());byid={r['image_id']:r for r in rows if r['role']=='train'};assert len(byid)==124
 provenance={'fold_sha256':sha(foldpath),'manifest_sha256':sha(manifest),'scope':'train124 only; held32/reserved156 targets not read','patches':[],'inputs':{}}
 for fold in folds['folds']:
  fid=fold['fold'];records=[];rng=np.random.default_rng(42+fid)
  for split,ids,count in [('train',fold['train_ids'],2048),('val',fold['validation_ids'],256)]:
   candidates=[byid[i] for i in ids];positives=[r for r in candidates if r['positive']];jobs={i:[] for i in ids}
   for idx in range(count):
    mode='positive' if idx%2==0 else 'uniform'
    pool=positives if mode=='positive' else candidates;r=pool[int(rng.integers(len(pool)))];jobs[r['image_id']].append((idx,mode))
   for ident,items in jobs.items():
    if not items:continue
    row=byid[ident];imagepath=ROOT/row['native_cache_path'];targetpath=ROOT/row['target_path']
    if ident not in provenance['inputs']:
     assert sha(imagepath)==row['native_cache_sha256'];assert sha(targetpath)==row['target_sha256']
     provenance['inputs'][ident]={'image_sha256':row['native_cache_sha256'],'target_sha256':row['target_sha256'],'study_id':row['study_id']}
    image=np.load(imagepath,allow_pickle=False);gt=np.zeros(image.shape,np.uint8)
    with np.load(targetpath,allow_pickle=False) as target:
     yy,xx=target['y'],target['x'];gt[yy,xx]=255
    h,w=image.shape;assert min(h,w)>=256 and image.dtype==np.uint8
    for idx,mode in items:
     if mode=='positive':
      q=int(rng.integers(len(yy)));cy=int(yy[q])+int(rng.integers(-64,65));cx=int(xx[q])+int(rng.integers(-64,65))
     else:cy=int(rng.integers(h));cx=int(rng.integers(w))
     y=int(np.clip(cy-128,0,h-256));x=int(np.clip(cx-128,0,w-256));patch=image[y:y+256,x:x+256];mask=gt[y:y+256,x:x+256]
     if mode=='positive':assert mask.any()
     name=f'f{fid}_{split}_{idx:04d}_{ident}';folder=out/f'fold_{fid}'/'patches';folder.mkdir(parents=True,exist_ok=True)
     ip=folder/(name+'.png');mp=folder/(name+'_mask.png');Image.fromarray(patch).save(ip);Image.fromarray(mask).save(mp)
     records.append({'id':name,'image':str(ip),'mask':str(mp),'patient_id':row['study_id'],'split':split})
     provenance['patches'].append({'id':name,'source_id':ident,'fold':fid,'split':split,'mode':mode,'xywh':[x,y,256,256],'image_sha256':sha(ip),'mask_sha256':sha(mp)})
   print('prepared fold',fid,split,count,flush=True)
  directory=out/f'fold_{fid}';(directory/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
  with (directory/'manifest.csv').open('w') as f:
   writer=csv.DictWriter(f,fieldnames=['image','mask','patient_id','split'],extrasaction='ignore');writer.writeheader();writer.writerows(records)
  assert not {r['patient_id'] for r in records if r['split']=='train'}&{r['patient_id'] for r in records if r['split']=='val'}
 (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 (out/'prepared_complete.json').write_text(json.dumps({'folds':3,'train_patches_per_fold':2048,'val_patches_per_fold':256,'provenance_sha256':sha(out/'provenance.json'),'actual_performance':False})+'\n')
if __name__=='__main__':main()

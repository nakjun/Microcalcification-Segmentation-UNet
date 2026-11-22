"""Image-only frozen new-checkpoint external cohorts; score after full cohort commit."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,ast,fcntl,sys,time
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,context,read,write,sha,digest,disk_guard

def relocate(path):
 source = Path(path)
 if source.is_file():
  return source
 raise FileNotFoundError(f"External inventory DICOM path does not exist: {source}")

def retained_functions():
 """Load only pure endpoint functions, without legacy drivers or path globals."""
 import cv2
 namespace={'np':np,'cv2':cv2,'checksum':sha}
 for file,names in [('scripts/external_v3/evaluate.py',{'average','metrics'}),('scripts/external_current_models/evaluate.py',{'summarize'})]:
  tree=ast.parse((ROOT/file).read_text())
  code=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
  exec(compile(code,str(ROOT/file),'exec'),namespace)
 return namespace['metrics'],namespace['summarize']

def pack(path,prob):
 temp=path.with_suffix('.tmp')
 with temp.open('wb') as f:np.savez_compressed(f,shape=np.array(prob.shape),mask=np.packbits(prob>=.5))
 os.replace(temp,path)

def unpack(path,shape):
 with np.load(path,allow_pickle=False) as z:
  assert tuple(z['shape'])==shape
  return np.unpackbits(z['mask'],count=int(np.prod(shape))).reshape(shape).astype(bool)

def native(row,folder):
 sys.path.insert(0,str(ROOT/'scripts'))
 from calcification_gate_sam_common import native_dicom
 source=relocate(row['dicom_path']);assert sha(source)==row['dicom_sha256']
 path=folder/(row['image_id']+'.npy');marker=path.with_suffix('.json');folder.mkdir(parents=True,exist_ok=True)
 if not marker.exists():
  a=native_dicom(str(source),(row['height'],row['width']))
  temp=path.with_suffix('.tmp')
  with temp.open('wb') as f:np.save(f,a,allow_pickle=False)
  os.replace(temp,path);write(marker,dict(source=str(source),source_sha256=row['dicom_sha256'],cache_sha256=sha(path),preprocessor_sha256=sha(ROOT/'scripts/calcification_gate_sam_common.py')))
 m=read(marker);assert m['source_sha256']==row['dicom_sha256'] and sha(path)==m['cache_sha256']
 assert m['preprocessor_sha256']==sha(ROOT/'scripts/calcification_gate_sam_common.py')
 return np.load(path,mmap_mode='r')

def run(args):
 from export_internal import load_pipeline
 p,_,_,out=context()
 dataset='cbis_test' if args.dataset=='cbis' else args.dataset
 config=next(c for c in p['configurations'] if c==dict(model=args.model,variant=args.variant,seed=args.seed))
 assert any(j['dataset']==args.dataset and all(j[k]==config[k] for k in config) and j['fold']==args.fold for j in p['external_jobs'])
 inventory=ROOT/f'runs/external_v3/inputs/{dataset}.json';rows=read(inventory)
 expected=410 if dataset=='inbreast' else 284;assert len(rows)==expected
 if args.limit:active=rows[:args.limit]
 else:active=rows
 predict,provenance=load_pipeline(args.model,args.variant,args.seed,args.fold,args.device)
 bind=dict(data_signature=p['data_signature'],pipeline=provenance,inventory_sha256=sha(inventory),evaluator_source={x:sha(ROOT/x) for x in ['scripts/external_v3/evaluate.py','scripts/external_current_models/evaluate.py','scripts/csaw312_rerun/export_external.py','scripts/csaw312_rerun/export_internal.py','scripts/calcification_gate_sam_common.py']},threshold=.5,gate=None,smoke=bool(args.limit))
 signature=digest(bind)
 folder=out/'evaluation'/('external_smoke' if args.limit else 'external')/dataset/f'{args.model}_{args.variant}'/f'seed_{args.seed}'/f'fold_{args.fold}'
 folder.mkdir(parents=True,exist_ok=True);lock=(folder/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 protocol=folder/'protocol.json'
 if protocol.exists():
  if read(protocol)!=bind:raise ValueError('Existing evaluation protocol differs')
 else:write(protocol,bind)
 files={}
 for index,row in enumerate(active):
  disk_guard(p,out);path=folder/(row['image_id']+'.npz');marker=path.with_suffix('.json')
  if marker.exists():
   m=read(marker);assert m['signature']==signature and sha(path)==m['prediction_sha256']
  else:
   prob=predict(native(row,out/'cache/external'/dataset))
   assert prob.shape==(row['height'],row['width']) and np.isfinite(prob).all()
   pack(path,prob);write(marker,dict(signature=signature,image_id=row['image_id'],prediction_sha256=sha(path),GT_read=False,source_dicom_sha256=row['dicom_sha256']))
  files[str(path)]=sha(path);files[str(marker)]=sha(marker)
  write(folder/'status.json',dict(phase='image_only_prediction',images=index+1,total=len(active),time=time.time()))
 write(folder/'predictions_complete.json',dict(complete=True,signature=signature,GT_read=False,smoke=bool(args.limit),image_ids=[r['image_id'] for r in active],files=files))
 if args.limit:return
 # Annotation access follows complete prediction commitment and verification.
 assert len(files)==expected*2
 for path,value in files.items():assert sha(path)==value
 annotation_path=ROOT/f'runs/external_v3/annotations/{dataset}.json';annotation=read(annotation_path)
 assert annotation['complete'] and annotation['images']==expected
 for path,value in annotation['source_files'].items():assert sha(relocate(path))==value
 lookup={r['image_id']:r for r in annotation['records']};assert set(lookup)=={r['image_id'] for r in rows}
 metric,summary=retained_functions();records=[];case_hashes={}
 for row in rows:
  a=lookup[row['image_id']]
  for roi in a.get('rois',[]):roi['mask_path']=str(relocate(roi['mask_path']))
  values=metric(unpack(folder/(row['image_id']+'.npz'),(row['height'],row['width'])),a,dataset)
  record=dict(image_id=row['image_id'],study_id=row['study_id'],metrics=values);records.append(record)
  case_path=folder/'cases'/(row['image_id']+'.json');write(case_path,record);case_hashes[str(case_path)]=sha(case_path)
 write(folder/'complete.json',dict(complete=True,signature=signature,annotation_sha256=sha(annotation_path),prediction_complete_sha256=sha(folder/'predictions_complete.json'),pipeline=provenance,model=args.model,variant=args.variant,seed=args.seed,fold=args.fold,dataset=dataset,images=expected,case_sha256=case_hashes,metrics=summary([r['metrics'] for r in records],dataset),annotation_summary=annotation['summary'],selection_performed=False))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--model',required=True);a.add_argument('--variant',required=True);a.add_argument('--seed',type=int,required=True);a.add_argument('--fold',type=int,required=True);a.add_argument('--dataset',choices=['inbreast','cbis'],required=True);a.add_argument('--device',default='cuda:0');a.add_argument('--limit',type=int);args=a.parse_args();run(args)

"""Serialized GPU training/internal queue; stops before external review gate."""
import argparse,fcntl,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/'runs/additional_baselines_20260926'
p=argparse.ArgumentParser();p.add_argument('--model',choices=['deepmica','yolo'],required=True);a=p.parse_args()
base=RUN/a.model;base.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MPLCONFIGDIR='/tmp/hlsam_matplotlib',PYTHONUNBUFFERED='1')
def status(state,**extra):
 value={'model':a.model,'state':state,'pid':os.getpid(),'updated_unix':time.time(),**extra};temp=base/'status.tmp';temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(base/'status.json')
def run(cmd):
 protocol=json.loads((RUN/'protocol.json').read_text())
 for path,digest in protocol['source_sha256'].items():
  if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen source changed: '+path)
 print('RUN',cmd,flush=True);subprocess.run([str(ROOT/'.venv/bin/python'),*cmd],cwd=ROOT,env=env,check=True)
try:
 status('waiting_gpu_lock')
 with (RUN/'gpu1.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  prepared=json.loads((RUN/'data/prepared_complete.json').read_text());assert prepared['folds']==3
  provenance=RUN/'data/provenance.json'
  assert hashlib.sha256(provenance.read_bytes()).hexdigest()==prepared['provenance_sha256']
  for patch in json.loads(provenance.read_text())['patches']:
   directory=RUN/f"data/fold_{patch['fold']}/patches"
   for suffix,key in [('.png','image_sha256'),('_mask.png','mask_sha256')]:
    path=directory/(patch['id']+suffix)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==patch[key],path
  assert json.loads((RUN/f'preflight/{a.model}.json').read_text())['forward_backward_update_finite']
  for fold in range(3):
   folder=base/f'fold_{fold}';data=RUN/f'data/fold_{fold}'
   if a.model=='deepmica':
    status('training',fold=fold,device='cuda:1')
    run(['DeepMiCa/pipeline.py','train','--manifest',str(data/'manifest.csv'),'--output',str(folder),'--epochs','16','--batch-size','16','--channels','1','--lr','.01','--seed','42','--device','cuda:1'])
   else:
    for stage in ['detector','segmentor']:
     status('training',fold=fold,stage=stage,device='cuda:1')
     cmd=['YOLO-HLSAM/train.py','--stage',stage,'--manifest',str(data/'manifest.json'),'--output',str(folder/stage),'--epochs','16','--seed','42','--device','cuda:1']
     if stage=='segmentor':cmd+=['--sam-checkpoint',str(ROOT/'runs/mammo_hypothesis/weights/sam_vit_b.pth')]
     run(cmd)
  (base/'training_complete.json').write_text(json.dumps({'model':a.model,'folds':3,'epochs':16,'seed':42,'selected_epoch':16,'scope':'native patch method adaptation; not original four-backbone loss protocol'})+'\n')
  status('internal_inference_evaluation',device='cuda:1')
  run(['scripts/additional_baselines/training/internal.py','--model',a.model,'--device','cuda:1'])
  status('awaiting_stage1_independent_review',external_started=False)
except BaseException as exc:
 status('failed',error=repr(exc));(base/'failure.txt').write_text(traceback.format_exc());raise

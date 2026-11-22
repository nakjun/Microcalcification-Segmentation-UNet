"""Immutable input bindings, isolated output paths and atomic run records."""
from pathlib import Path
import hashlib,json,os,shutil,time
ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/os.environ.get('CSAW312_PROTOCOL', 'configs/csaw312_rerun_20261105/protocol.json')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
 with tmp.open('w') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def torch_save(p,d):
 import torch
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
 with tmp.open('wb') as f:torch.save(d,f);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def context():
 p=read(PLAN);data=Path(p['data_root']);m=read(data/'manifest.json');s=read(data/'splits.json');done=read(data/'complete.json')
 assert done['technical_complete'] and done['signature']==p['data_signature']==m['signature']==s['signature']
 for name,h in done['file_sha256'].items():assert sha(data/name)==h
 out=Path(p['output_root']).resolve()
 assert out.name=='CSAW312_Rerun_20261105' and 'archive' not in out.parts
 return p,{r['image_id']:r for r in m['images']},s,out

def source_signature(job,smoke=False):
 p,rows,s,out=context()
 names=['common.py','worker.py','sampling.py','backbone.py','specialized.py','metrics.py']
 sources={str(PLAN.relative_to(ROOT)):sha(PLAN)}
 for name in ['manifest.json','splits.json','initial_candidates.json']:
  path=Path(p['data_root'])/name;sources[str(path)]=sha(path)
 sources[p['sam_checkpoint']]=sha(p['sam_checkpoint'])
 for n in names:
  path=Path(__file__).parent/n;sources[str(path.relative_to(ROOT))]=sha(path)
 for folder in ['scripts/comparative_study/models','scripts/improvement_unext_deterministic','DeepMiCa','YOLO-HLSAM']:
  for path in (ROOT/folder).rglob('*.py'):
   if '__pycache__' not in path.parts:sources[str(path.relative_to(ROOT))]=sha(path)
 for folder in ['external/yolo-hlsam','external/DeepMiCa','external/segment-anything']:
  for path in (ROOT/folder).rglob('*'):
   if path.is_file() and path.suffix in ('.py','.yaml','.yml') and '__pycache__' not in path.parts:
    sources[str(path.relative_to(ROOT))]=sha(path)
 for name in ['calcification_refinement_common.py','calcification_refinement_v2_common.py','calcification_refinement_v5_common.py']:
  sources['scripts/'+name]=sha(ROOT/'scripts'/name)
 dependencies={}
 prefix=Path('smoke')/digest(sources)[:12] if smoke else Path('jobs')
 for depid in job['depends_on']:
  dj=next(j for j in p['jobs'] if j['job_id']==depid)
  expected,_=source_signature(dj,smoke=smoke)
  dp=out/prefix/depid;record=read(dp/'complete.json')
  assert record['complete'] and record['signature']==expected and record['job']==dj
  assert Path(record['checkpoint']).name==record['checkpoint']
  assert sha(dp/record['checkpoint'])==record['checkpoint_sha256']
  dependencies[depid]=dict(complete_sha256=sha(dp/'complete.json'),checkpoint_sha256=record['checkpoint_sha256'],selected_epoch=record['selected_epoch'])
 return digest(dict(job=job,data=p['data_signature'],sources=sources,dependencies=dependencies,smoke=smoke)),sources

def disk_guard(p,out):
 if shutil.disk_usage(out).free < p['storage']['min_free_gib']*2**30:raise RuntimeError('Disk free below protocol reserve')

def lr_for_epoch(initial,epoch,p):
 import math
 t=p['training'];return initial*(t['final_lr_multiplier']+(1-t['final_lr_multiplier'])*(.5*(1+math.cos(math.pi*epoch/t['cosine_horizon']))))

def better(candidate,best):
 return best is None or (-candidate['metric'],candidate.get('negative_fp_per_image',0),candidate['epoch'])<(-best['metric'],best.get('negative_fp_per_image',0),best['epoch'])

def should_stop(epoch,last_improvement,p):
 t=p['training'];return epoch>=t['min_epochs'] and epoch-last_improvement>=t['patience_epochs']

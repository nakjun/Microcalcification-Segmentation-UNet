"""Retained selected-model lineage, explicitly not full deleted-history verification."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import torch,numpy as np,cv2
from calcification_gate_common import checksum,atomic_json
from train_calcification_refinement_v2 import read,digest,validate_folds
V2=Path('runs/calcification_refinement_v2_seed42');V3=Path('runs/calcification_refinement_v3_seed42')
ROOT=Path('runs/v3_ideas/idea1')
ARMS=['imagenet','mammo']
def immutable(path,value):
 path=Path(path)
 if path.exists():assert read(path)==value,path
 else:atomic_json(path,value)
def verify(path,expected=None):
 result=read(path);assert result['complete']
 if expected is not None:assert result['signature']==expected
 for p,h in result['output_files'].items():assert checksum(p)==h,p
 return result
def context():
 config=read(V3/'config.json');selected=read(V3/'selection.json')['selected']
 assert (selected['variant'],selected['epoch'],selected['policy'])==('mining_only',15,'two_region@0.3|0.9')
 signature3=read(V3/'signature.json');assert signature3['config']==config
 assert checksum(V2/'signature.json')==signature3['source_signature']
 original=read(V2/'signature.json');assert signature3['selection']==read(V3/'selection.json')
 folds=read(V2/'folds.json');assert checksum(folds['source_manifest'])==folds['source_manifest_sha256']
 rows=read(folds['source_manifest']);validate_folds(rows,folds['folds'])
 parent=read(V2/'complete.json');sources={}
 for p in [V3/'signature.json',V3/'selection.json',V3/'config.json',V2/'complete.json',V2/'folds.json',Path(folds['source_manifest']),Path('reports/refinement_cleanup_manifest.json'),V2/'signature.json',*[Path('scripts/v3_ideas')/name for name in ['baseline.py','cache.py']]]:
  if p.name.startswith('learning'):continue
  sources[str(p)]=checksum(p)
 # Source model implementation files are included directly, never checked through a deleted-epoch list.
 for name in ['calcification_refinement_common.py','calcification_refinement_v2_common.py','calcification_gate_common.py','calcification_gate_sam_common.py','train_calcification_refinement_v2.py','evaluate_calcification_refinement_v3_union.py','calcification_refinement_v6_common.py']:
  p=Path('scripts')/name;current=checksum(p)
  for oldpath,oldsha in original['files'].items():
   if Path(oldpath).resolve()==p.resolve():assert current==oldsha,p
  sources[str(p)]=current
 jobs={}
 for f in folds['folds']:
  j=V2/'jobs/mining_only'/f'fold_{f["fold"]}';donepath=j/'complete.json'
  assert checksum(donepath)==parent['output_files'][str(donepath)]
  done=read(donepath);cp=j/'checkpoints/epoch_15.pt';assert checksum(cp)==done['output_files'][str(cp)]
  sources[str(donepath)]=checksum(donepath);sources[str(cp)]=checksum(cp);jobs[str(f['fold'])]=dict(path=str(j),checkpoint=str(cp),checkpoint_sha256=checksum(cp),marker=done)
  for ident in f['validation_ids']:
   p=j/'epochs/15/cases'/f'{ident}.json';assert checksum(p)==done['output_files'][str(p)];sources[str(p)]=checksum(p)
 for row in rows:
  if row['role']!='train':continue
  assert checksum(row['native_cache_path'])==row['native_cache_sha256']
  sources[row['native_cache_path']]=row['native_cache_sha256']
  for arm in ARMS:
   p=Path(config['development_baseline_root'])/'csaw'/arm/'gated/cases'/f'{row["image_id"]}.json';case=read(p)
   assert checksum(case['mask_path'])==case['mask_sha256'];sources[str(p)]=checksum(p);sources[case['mask_path']]=case['mask_sha256']
 refs={}
 for name,r in [('v2',V2),('v4b',Path('runs/calcification_refinement_v4b_seed42'))]:
  p=r/'selection.json';refs[name]=read(p)['selected']['metrics'];sources[str(p)]=checksum(p)
 protocol=dict(runtime=dict(torch=str(torch.__version__),numpy=np.__version__,opencv=cv2.__version__),cache_contract='FP32 full native probability,original model/input/gate,no targets,no policy search',config=config,selected=selected,reference=refs,sources=sources,validation_scope='retained selected-model/input subset;legacy all-epoch hash verification unavailable')
 sig=digest(protocol)
 if (ROOT/'protocol.json').exists():assert read(ROOT/'protocol.json')==protocol
 return dict(protocol=protocol,signature=sig,config=config,rows=rows,folds=folds['folds'],jobs=jobs)


def evaluation_context():
 c=context();plan=read('configs/v3_ideas/idea1.json')
 inputs={str(ROOT/'protocol.json'):checksum(ROOT/'protocol.json')}
 for name in ['policies.py','evaluate.py','test_idea1.py']:
  p=Path('scripts/v3_ideas')/name;inputs[str(p)]=checksum(p)
 inputs['configs/v3_ideas/idea1.json']=checksum('configs/v3_ideas/idea1.json')
 for row in c['rows']:
  if row['role']=='train':inputs[row['target_path']]=row['target_sha256']
 c['evaluation_protocol']=dict(cache_signature=c['signature'],plan=plan,inputs=inputs)
 c['evaluation_signature']=digest(c['evaluation_protocol'])
 return c

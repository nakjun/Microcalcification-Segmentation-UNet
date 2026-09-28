"""Immutable held32 inputs and retained comparative checkpoint lineage."""
from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

ROOT=Path('runs/comparative_held32_seed42'); PLAN=Path('configs/comparative_held32/protocol.json')
SCRIPTS=Path(__file__).resolve().parents[1]; VARIANTS=('C','L','H','LH'); ARMS=('imagenet','mammo')

def read(p): return json.loads(Path(p).read_text())
def checksum(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def atomic_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));t.replace(p)
def immutable(p,x):
 p=Path(p)
 if p.exists():assert read(p)==x,p
 else:atomic_json(p,x)
def atomic_npz(p,**x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
 with t.open('wb') as f:np.savez_compressed(f,**x)
 t.replace(p)
def verify(p,sig=None):
 x=read(p);assert x['complete'],p
 if sig is not None:assert x['signature']==sig,p
 for q,h in x.get('output_files',{}).items():assert checksum(q)==h,q
 return x

def _external_common():
 p=SCRIPTS/'external_comparative_inbreast/common.py';spec=importlib.util.spec_from_file_location('held_external_checkpoint_common',p);m=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(m);return m
def checkpoints(): return _external_common().checkpoints()
def import_models(): return _external_common().import_models()

def prepare():
 folds=read('runs/calcification_refinement_v5_seed42/folds.json');manifest=Path(folds['source_manifest']);assert checksum(manifest)==folds['source_manifest_sha256'];rows=read(manifest)
 train=[r for r in rows if r['role']=='train'];held=[r for r in rows if r['role']!='train'];assert len(train)==124 and len(held)==32
 assert not {r['study_id'] for r in train}&{r['study_id'] for r in held}
 strip=lambda r:{k:v for k,v in r.items() if k not in ('target_path','target_sha256','positive','gt_pixels','gt_source_path','gt_source_sha256','gt_pixel_sha256')}
 held_inputs=[strip(r) for r in held];smoke_inputs=[strip(train[0])]
 cps=checkpoints();sources={str(PLAN):checksum(PLAN),str(manifest):checksum(manifest),str(Path(__file__)):checksum(__file__),
                              'docs/improvement_team/comparative_held32_protocol.md':checksum('docs/improvement_team/comparative_held32_protocol.md')}
 for name in ('prepare.py','infer.py','verify_inference.py'):
  p=Path(__file__).with_name(name);sources[str(p)]=checksum(p)
 base_root=Path('runs/calcification_refinement_v1/baseline_development');base_done=read(base_root/'complete.json');assert base_done['complete'] and base_done['GT_masks_read'] is False and base_done['GT_box_prompts_used'] is False
 sources[str(base_root/'complete.json')]=checksum(base_root/'complete.json')
 for r in held:
  assert checksum(r['native_cache_path'])==r['native_cache_sha256'];sources[r['native_cache_path']]=r['native_cache_sha256']
  # These image-only gate/fill artifacts exist for all 156 development images; target files are deliberately not opened or hashed.
  for arm in ARMS:
   case=base_root/'csaw'/arm/'gated/cases'/f"{r['image_id']}.json";m=read(case);assert base_done['output_files'][str(case)]==checksum(case) and m['gt_read_for_inference'] is False and m['signature_digest'];assert checksum(m['mask_path'])==m['mask_sha256'];sources[str(case)]=checksum(case);sources[m['mask_path']]=m['mask_sha256']
 for r in smoke_inputs:
  assert checksum(r['native_cache_path'])==r['native_cache_sha256'];sources[r['native_cache_path']]=r['native_cache_sha256']
  for arm in ARMS:
   case=base_root/'csaw'/arm/'gated/cases'/f"{r['image_id']}.json";m=read(case);assert base_done['output_files'][str(case)]==checksum(case) and m['gt_read_for_inference'] is False;assert checksum(m['mask_path'])==m['mask_sha256'];sources[str(case)]=checksum(case);sources[m['mask_path']]=m['mask_sha256']
 for cp in cps:
  state=__import__('torch').load(cp['checkpoint'],map_location='cpu',weights_only=False);job=state['job'];root=Path(cp['root']);proto=read(root/'protocol.json');fold=next(x for x in proto['folds'] if x['fold']==cp['fold']);assert job['train_ids']==fold['train_ids'] and job['validation_ids']==fold['validation_ids'];ids=set(job['train_ids']);assert ids <= {r['image_id'] for r in train};assert not ids&{r['image_id'] for r in held}
  if cp['kind']=='study':assert read(root/'complete.json')['output_files'][cp['complete_path']]==cp['complete_sha256']
  sources[cp['checkpoint']]=cp['checkpoint_sha256'];sources[cp['complete_path']]=cp['complete_sha256']
 for p in [SCRIPTS/'external_comparative_inbreast/common.py',SCRIPTS/'calcification_refinement_common.py',SCRIPTS/'calcification_refinement_v2_common.py',SCRIPTS/'comparative_study/model.py',SCRIPTS/'improvement_unext_deterministic/model.py']:
  sources[str(p)]=checksum(p)
 protocol=dict(plan=read(PLAN),manifest=str(manifest),manifest_sha256=checksum(manifest),held_rows=held_inputs,smoke_rows=smoke_inputs,checkpoints=cps,sources=sources,GT_read=False,held_previously_exposed=True)
 immutable(ROOT/'protocol.json',protocol);return dict(protocol=protocol,signature=digest(protocol),rows=held,checkpoints=cps)
def context():
 p=read(ROOT/'protocol.json');assert p['GT_read'] is False
 for q,h in p['sources'].items():assert checksum(q)==h,q
 assert len(p['held_rows'])==32 and len(p['checkpoints'])==48
 return dict(protocol=p,signature=digest(p),rows=p['held_rows'],smoke_rows=p['smoke_rows'],checkpoints=p['checkpoints'])
def prediction_dir(cp):return ROOT/'predictions'/cp['model']/cp['variant']/f"fold_{cp['fold']}"

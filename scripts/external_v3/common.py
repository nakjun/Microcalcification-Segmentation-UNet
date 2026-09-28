"""Fixed external inference lineage. Never reads annotation targets."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,cv2,torch,pydicom
from calcification_gate_common import checksum,atomic_json
from calcification_gate_sam_common import atomic_npz
from train_calcification_refinement_v2 import digest
ROOT=Path('runs/external_v3')
V3=Path('runs/calcification_refinement_v3_seed42')
PIPE=Path('runs/calcification_gate_retinanet_sam_clean_v1')
ARMS=['imagenet','mammo']
def read(p):return json.loads(Path(p).read_text())
def immutable(p,v):
 p=Path(p)
 if p.exists():assert read(p)==v,p
 else:atomic_json(p,v)
def verify(p,signature=None):
 d=read(p);assert d['complete']
 if signature is not None:assert d['signature']==signature
 for f,h in d['output_files'].items():assert checksum(f)==h,f
 return d
def context(dataset):
 assert dataset in ['inbreast','cbis_test']
 manifest=ROOT/'inputs'/f'{dataset}.json';rows=read(manifest)
 assert len({r['image_id'] for r in rows})==len(rows)
 assert all(r['dataset']==dataset and set(r)<=set('image_id dataset dicom_path dicom_sha256 height width spacing study_id'.split()) for r in rows)
 sources={}
 def add(p):sources[str(p)]=checksum(p)
 parent=read(V3/'complete.json');jobpath=V3/'final_fit/complete.json';job=read(jobpath)
 assert parent['complete'] and job['complete'];assert checksum(jobpath)==parent['output_files'][str(jobpath)]
 cp=V3/'final_fit/checkpoints/epoch_15.pt';assert checksum(cp)==job['output_files'][str(cp)]
 assert job['signature']==parent['signature'] and job['epochs']==15
 for p in [V3/'complete.json',jobpath,cp,V3/'signature.json',V3/'selection.json',V3/'config.json',manifest,Path('configs/external_v3/protocol.json')]:add(p)
 assert checksum(V3/'signature.json')==parent['output_files'][str(V3/'signature.json')]
 sig3=read(V3/'signature.json');assert sig3['config']==read(V3/'config.json')
 assert sig3['selection']==read(V3/'selection.json')
 orig=Path('runs/calcification_refinement_v2_seed42/signature.json');assert checksum(orig)==sig3['source_signature'];add(orig)
 pipesig=read(PIPE/'signature.json');pipedone=read(PIPE/'complete.json')
 assert pipedone['complete'] and not pipedone['smoke']
 assert pipedone['signature_digest']==hashlib.sha256(json.dumps(pipesig,sort_keys=True).encode()).hexdigest()
 historical=[read(orig)['files'],pipesig['files']];add(PIPE/'signature.json');add(PIPE/'complete.json')
 prepared=read(PIPE/'prepared.json');assert prepared['complete'];add(PIPE/'prepared.json')
 # Verify required retained implementation and model files against old SHA when registered.
 names=['calcification_refinement_common.py','calcification_refinement_v2_common.py','calcification_gate_common.py','calcification_gate_sam_common.py','train_calcification_refinement_v2.py','prepare_retinanet_roi.py','train_retinanet_roi.py','train_calcification_gate_finetune.py','extract_calcification_features.py']
 paths=[Path('scripts')/n for n in names]+[Path(__file__),Path(__file__).with_name('infer.py'),Path(__file__).with_name('prepare.py')]
 paths+=list(Path('external/Mammo-CLIP/src/codebase/Detectors/retinanet').rglob('*.py'))
 paths+=[Path('runs/classifier_preflight/report.json')]+list(Path('runs/mammo_hypothesis/weights').glob('*encoder.pt'))
 paths+=[PIPE/'frozen/gate/classifier.pt',PIPE/'frozen/gate/thresholds.json']
 for arm in ARMS:paths += [PIPE/f'frozen/{arm}/best.pt',PIPE/f'frozen/{arm}/snapshot.json']
 for p in paths:
  add(p)
  if str(p).startswith(str(PIPE/'frozen')):assert sources[str(p)]==prepared['files'][str(p)],p
  for old in historical:
   for f,h in old.items():
    if Path(f).resolve()==p.resolve():assert sources[str(p)]==h,p
 # Inventory measures each source SHA; infer checks actual bytes per case, including resume.
 for r in rows:sources[r['dicom_path']]=r['dicom_sha256']
 protocol=dict(plan=read('configs/external_v3/protocol.json'),dataset=dataset,sources=sources,model_checkpoint=str(cp),job=job['job'],config=read(V3/'config.json'),runtime=dict(torch=str(torch.__version__),numpy=np.__version__,opencv=cv2.__version__,pydicom=pydicom.__version__),scope='retained selected model and required inference inputs; not full deleted-history verification')
 if (ROOT/dataset/'protocol.json').exists():assert read(ROOT/dataset/'protocol.json')==protocol
 return dict(rows=rows,protocol=protocol,signature=digest(protocol),checkpoint=cp,config=protocol['config'])

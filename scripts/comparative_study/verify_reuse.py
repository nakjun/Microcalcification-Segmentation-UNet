"""Read-only compatibility check for reuse of the completed fixed UNeXt study."""
import json
from pathlib import Path
from common import context,read,checksum,verify
from calcification_gate_common import atomic_json
c=context();p=Path('runs/improvement_unext_deterministic/unext_v5_losses_deterministic_seed42');old=read(p/'protocol.json');done=verify(p/'complete.json',read(p/'complete.json')['signature'])
assert done['technical_validation_complete'] and done['jobs']==12 and done['epochs']==192 and done['oof_native_images']==496
assert old['folds']==c['protocol']['folds'] and old['initial_pools']==c['protocol']['initial_pools']
a,b=old['config'],c['protocol']['config'];exclude={'root','normalization','backbone','architecture_config'}
assert {k:v for k,v in a.items() if k not in exclude}=={k:v for k,v in b.items() if k not in exclude}
for k in ['seed','variants','epochs','patches_per_epoch','batch_size','primary','secondary','loss','mining']:
 assert old['plan'][k]==c['protocol']['plan'][k],k
assert old['plan']['runtime']==c['protocol']['plan']['runtime']
paths=[p/'protocol.json',p/'complete.json',p/'oof/complete.json']
atomic_json('reports/comparative_study/reuse_unext.json',dict(complete=True,study_signature=c['signature'],reused=True,not_new_training=True,model='UNeXt_author_2ch',same_fixed_training_evaluation_and_runtime=True,model_initialization_specific=True,scope='seed42 development3fold OOF124',verifier_sha256=checksum(__file__),output_files={str(x):checksum(x) for x in paths}))
print('fixed UNeXt study reuse compatibility PASS')

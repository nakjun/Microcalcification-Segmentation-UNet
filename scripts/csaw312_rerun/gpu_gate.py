"""Run actual GPU smoke stages for the configured training protocol."""
import subprocess,sys,time,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import context,read,write,sha,source_signature,digest,ROOT,PLAN
p,_,_,out=context();logs=out/'gpu_preflight';logs.mkdir(parents=True,exist_ok=True)
indices=[18,42,66,90,96,102,103,19]
passed=[]
for i in indices:
 job=p['jobs'][i];write(logs/'status.json',dict(state='gpu_smoke',active_job=job['job_id'],passed=passed,time=time.time()))
 print('GPU smoke '+job['job_id'],flush=True)
 with (logs/f'{i}.log').open('a') as f:
  code=subprocess.call([str(ROOT/'.venv/bin/python'),str(Path(__file__).parent/'worker.py'),'--job-index',str(i),'--device','cuda:0','--smoke'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 if code:
  write(logs/'status.json',dict(state='failed',job_index=i,returncode=code,passed=passed))
  print((logs/f'{i}.log').read_text()[-6000:],flush=True);sys.exit(code)
 sig,sources=source_signature(job,smoke=True);folder=out/'smoke'/digest(sources)[:12]/job['job_id'];d=read(folder/'complete.json')
 assert d['smoke'] and d['signature']==sig and sha(folder/d['checkpoint'])==d['checkpoint_sha256']
 passed.append(i)
_,sources=source_signature(p['jobs'][0])
write(out/'preflight.json',dict(ready=True,protocol_sha256=sha(PLAN),source_sha256=sources,gpu='cuda:0',smoke_jobs=passed,cpu_additional_tests=False,user_requested_gpu_only=True))
write(logs/'status.json',dict(state='passed',passed=passed,time=time.time()))
print('GPU preflight complete.', flush=True)

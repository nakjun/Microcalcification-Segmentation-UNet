"""Serialized, resumable inference/evaluation queue; immutable results and failure status."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from common import CONDITIONS, DATASETS, REPO, RUN, context, selected


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--condition',choices=CONDITIONS)
    p.add_argument('--fold',type=int,choices=(0,1,2))
    p.add_argument('--dataset',choices=tuple(DATASETS))
    a=p.parse_args()
    context(a.condition,a.fold)
    tag=f"{a.device.replace(':','')}_{a.condition or 'all'}_{a.fold if a.fold is not None else 'all'}_{a.dataset or 'all'}"
    status_path=RUN/'queue'/f'{tag}.json'
    status_path.parent.mkdir(parents=True,exist_ok=True)
    def status(state,**extra):
        temp=status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(state=state,pid=os.getpid(),updated_unix=time.time(),device=a.device,**extra),indent=2)+'\n')
        temp.replace(status_path)
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    job={}
    try:
        with (RUN/'queue'/f"{a.device.replace(':','')}.lock").open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for condition,fold in selected(a.condition,a.fold):
                for dataset in ([a.dataset] if a.dataset else DATASETS):
                    job=dict(condition=condition,fold=fold,dataset=dataset)
                    common=['--condition',condition,'--fold',str(fold),'--dataset',dataset]
                    for stage,script,extra in [('inference','infer.py',['--device',a.device]),('evaluation','evaluate.py',[])]:
                        status(stage,**job)
                        cmd=[sys.executable,str(REPO/'scripts/external_current_models'/script),*common,*extra]
                        print('RUN',cmd,flush=True)
                        subprocess.run(cmd,cwd=REPO,env=env,check=True)
            status('complete',scope='selected full cohorts; no ensemble or tuning')
    except BaseException as exc:
        status('failed',error=repr(exc),**job)
        status_path.with_suffix('.failure.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__':
    main()

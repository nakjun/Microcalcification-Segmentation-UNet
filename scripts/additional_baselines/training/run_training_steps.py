"""Finite CPU-source-verified GPU training-cost smoke, measurements and summary."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
SCRIPT=ROOT/'scripts/additional_baselines/training/benchmark_training_steps.py'
OUT=ROOT/'reports/additional_baselines/training_steps_20260927'
NAMES=('proposed','unext','unet2d','utnet')
ARMS=('C','L','H','LH')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def verify(path,model,fold,smoke,gpu):
    r=read(path)
    assert r['complete'] and r['model']==model and r['fold']==fold and r['smoke']==smoke and r['gpu']==gpu
    assert r['checkpoint_unchanged'] and sha(r['checkpoint'])==r['checkpoint_sha256']
    assert r['same_initial_model_and_optimizer_for_all_loss_arms']
    for source,h in r['source_sha256'].items():assert sha(ROOT/source)==h
    assert r['batch']['component_active_patches']>0 and r['batch']['eligible_background_pixels']>0
    assert set(r['results'])==set(ARMS)
    for arm in ARMS:
        assert len(r['results'][arm]['seconds_runs'])==(1 if smoke else 10)
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        for smoke in (True,False):
            for model in NAMES:
                for fold in ((0,) if smoke else (0,1,2)):
                    dest=OUT/('smoke' if smoke else 'measurements')/f'{model}_fold{fold}.json'
                    write(OUT/'status.json',dict(complete=False,stage='smoke' if smoke else 'measurement',model=model,fold=fold))
                    if not dest.exists():
                        command=[sys.executable,str(SCRIPT),'--model',model,'--fold',str(fold),'--gpu',str(a.gpu)]
                        if smoke:command.append('--smoke')
                        subprocess.run(command,cwd=ROOT,check=True)
                    verify(dest,model,fold,smoke,a.gpu)
        records={model:[verify(OUT/'measurements'/f'{model}_fold{fold}.json',model,fold,False,a.gpu) for fold in range(3)] for model in NAMES}
        for fold in range(3):
            assert len({records[m][fold]['batch']['input_target_geometry_sha256'] for m in NAMES})==1
        result={}
        for model in NAMES:
            result[model]={}
            for arm in ARMS:
                fields={k:[r['results'][arm][k] for r in records[model]] for k in ('mean_seconds','mean_seconds_minus_C','mean_seconds_ratio_to_C','peak_allocated_bytes','incremental_peak_allocated_bytes')}
                result[model][arm]={k:dict(values=v,mean=float(np.mean(v)),sample_sd_across_folds=float(np.std(v,ddof=1))) for k,v in fields.items()}
        sources={str(p.relative_to(ROOT)):sha(p) for p in (SCRIPT,Path(__file__).resolve())}
        outputs={str(p.relative_to(ROOT)):sha(p) for p in sorted((OUT/'measurements').glob('*.json'))}
        write(OUT/'complete.json',dict(complete=True,scope='fixed-batch training optimizer-step microbenchmark only; excludes mining/sampling/IO; no full-training cost estimate',
            jobs=12,loss_arms=48,measured_updates=480,warmup_updates=144,smoke_excluded=True,
            same_batch_across_models_within_fold=True,folds=[0,1,2],results=result,source_sha256=sources,output_files=outputs))
        write(OUT/'status.json',dict(complete=True,stage='complete'))
    except Exception as e:
        write(OUT/'status.json',dict(complete=False,stage='failed',error=repr(e)));raise

if __name__=='__main__':main()

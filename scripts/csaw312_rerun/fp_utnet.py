"""Paired UTNet seed-43 rerun on the frozen CSAW312 folds."""
import argparse
import fcntl
from pathlib import Path
from types import SimpleNamespace

import common
import worker
import export_internal as internal

OUT = common.ROOT / 'runs/CSAW312_FP_Rerun_20261119/utnet'
VARIANTS = ('C', 'LH')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan, lookup, splits, original = common.context()
    jobs = []
    for job in plan['jobs']:
        if job['model'] == 'utnet' and job['seed'] == 42 and job['variant'] in VARIANTS:
            jobs.append(dict(job, seed=43, job_id=job['job_id'].replace('seed_42','seed_43'),
                             depends_on=[v.replace('seed_42','seed_43') for v in job['depends_on']]))
    assert len(jobs) == 12 and len({job['job_id'] for job in jobs}) == 12
    plan = dict(plan, jobs=jobs, output_root=str(OUT))
    _, sources = common.source_signature(dict(jobs[0], depends_on=[]))
    for file in [Path(__file__), Path(internal.__file__)]:
        sources[str(file.resolve().relative_to(common.ROOT))] = common.sha(file)

    def signature(job, smoke=False):
        dependencies = {}
        for dep in job['depends_on']:
            dep_job = next(j for j in jobs if j['job_id'] == dep)
            folder = OUT / 'jobs' / dep
            marker = common.read(folder / 'complete.json')
            assert marker['complete'] and marker['job'] == dep_job
            assert marker['signature'] == signature(dep_job)[0]
            assert common.sha(folder / marker['checkpoint']) == marker['checkpoint_sha256']
            dependencies[dep] = common.sha(folder / 'complete.json')
        return common.digest(dict(job=job, sources=sources, dependencies=dependencies, smoke=smoke)), sources

    worker.context = internal.context = lambda: (plan, lookup, splits, OUT)
    worker.source_signature = internal.source_signature = signature
    worker.torch.set_num_threads(4)
    worker.cv2.setNumThreads(1)
    for variant in VARIANTS:
        index = next(i for i,j in enumerate(jobs) if j['phase']=='selection' and j['fold']==0 and j['variant']==variant)
        worker.run(SimpleNamespace(job_index=index, device=args.device, smoke=True))
    if args.smoke_only:
        return
    common.write(OUT/'preflight.json',dict(ready=True,protocol_sha256=common.sha(common.PLAN),source_sha256=sources))
    paths = {}
    for fold in range(3):
        for variant in VARIANTS:
            for phase in ('selection','refit'):
                index = next(i for i,j in enumerate(jobs) if (j['fold'], j['variant'], j['phase'])==(fold,variant,phase))
                worker.run(SimpleNamespace(job_index=index, device=args.device, smoke=False))
            pipe = internal._load_pipeline('utnet',variant,43,fold,args.device)
            for split in ('oof','test'):
                internal.export_split(pipe, split)
                folder = OUT/'evaluation/internal'/f'utnet_{variant}'/'seed_43'/f'fold_{fold}'/split
                paths.setdefault(variant,{}).setdefault(str(fold),{})[split] = str(folder/'counts.json')
            del pipe
    common.write(OUT/'evaluations.json', paths)


if __name__ == '__main__':
    main()

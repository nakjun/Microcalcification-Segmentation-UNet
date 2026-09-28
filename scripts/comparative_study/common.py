"""Controlled architecture-by-loss comparative study; preserved historical sources."""
import sys, os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'v3_ideas'))
from baseline import read, checksum, verify, digest, immutable, ARMS
from train_calcification_refinement_v2 import validate_folds
BACKBONE=os.environ.get('STUDY_BACKBONE','proposed')
assert BACKBONE in ('proposed','unet2d','utnet')
ROOT=Path('runs/comparative_study')/(BACKBONE+'_seed42')
OUT=Path('reports/comparative_study')/(BACKBONE+'_seed42')
OLD=Path('runs/calcification_refinement_v5_seed42')
SCRIPTS=Path('scripts/comparative_study')
PLAN=Path('configs/comparative_study/plan.json')
VARIANTS=('C','L','H','LH')

def spec(c,fold):return next(x for x in c['folds'] if x['fold']==fold)

def context(variant='C'):
    assert variant in VARIANTS
    plan=read(PLAN);previous=read(OLD/'signature.json');old_done=read(OLD/'complete.json')
    assert old_done['complete'] and old_done['selected'] is None
    sources={str(PLAN):checksum(PLAN)}
    for name in ['common.py','train.py','miner.py','evaluate.py','model.py']:
        assert (SCRIPTS/name).is_file(),name
    def bind(p,expected=None):
        p=str(p);actual=checksum(p)
        if expected is not None:assert actual==expected,p
        sources[p]=actual
    for name in ['signature','plan','config','folds','complete','selection','oof_candidates']:
        p=OLD/f'{name}.json';bind(p,old_done['output_files'].get(str(p)))
    folds_record=read(OLD/'folds.json');bind(folds_record['source_manifest'],folds_record['source_manifest_sha256'])
    rows=read(folds_record['source_manifest']);validate_folds(rows,folds_record['folds'])
    cfg=read(OLD/'config.json');cfg.update(runtime_deterministic_algorithms=True,cublas_workspace_config=':4096:8',root=str(ROOT),normalization=plan['backbones'][BACKBONE]['normalization'],backbone=BACKBONE,gpu_memory_fraction=.4,gpu_min_free_mib=8000)
    cfg['architecture_config']=plan['backbones'][BACKBONE]
    for p in SCRIPTS.glob('*.py'):
        if p.name.startswith('verify_') or p.name.startswith('render_'):continue
        bind(p)
    for p in (SCRIPTS/'models').rglob('*'):
        if p.is_file() and '__pycache__' not in str(p):bind(p)
    for p in ['docs/improvement_team/comparative_study_candidate.md','docs/improvement_team/comparative_study_review.md']:
        bind(p)
    for p in ['scripts/calcification_refinement_common.py','scripts/calcification_refinement_v2_common.py','scripts/calcification_refinement_v5_common.py','scripts/train_calcification_refinement_v2.py','scripts/calcification_refinement_coverage_common.py','scripts/calcification_gate_common.py','scripts/calcification_gate_sam_common.py','scripts/calcification_refinement_v6_common.py','scripts/evaluate_calcification_refinement_v3_union.py','scripts/train_calcification_refinement.py','scripts/v3_ideas/baseline.py']:
        bind(p,previous['files'].get(p))
    for row in rows:
        if row['role']!='train':continue
        bind(row['native_cache_path'],row['native_cache_sha256']);bind(row['target_path'],row['target_sha256'])
        for arm in ARMS:
            p=Path(cfg['development_baseline_root'])/'csaw'/arm/'gated/cases'/f'{row["image_id"]}.json'
            assert str(p) in previous['files'];bind(p,previous['files'][str(p)])
            m=read(p);bind(m['mask_path'],m['mask_sha256'])
    initial={}
    for f in folds_record['folds']:
        fi=f['fold'];job=OLD/'jobs/C'/f'fold_{fi}';marker=job/'complete.json'
        bind(marker,old_done['output_files'][str(marker)]);done=read(marker)
        assert done['job']['train_ids']==f['train_ids'] and done['job']['validation_ids']==f['validation_ids']
        path=job/'data/sampling_centers.npz';meta=path.with_suffix('.json')
        bind(path,done['output_files'][str(path)]);bind(meta,done['output_files'][str(meta)])
        initial[str(fi)]=dict(path=str(path),sha256=checksum(path),training_ids=f['train_ids'])
    old_candidates=read(OLD/'oof_candidates.json')
    historical={v:next(x for x in old_candidates['candidates'] if x['candidate_id']==f'{v}/epoch_16/full_image@0.9')['metrics'] for v in VARIANTS}
    matched_path=Path('runs/improvement_ranking/small_relative_logit_seed42/oof/complete.json');bind(matched_path)
    matched=read(matched_path)['pipeline']['control']
    protocol=dict(plan=plan,config=cfg,folds=folds_record['folds'],sources=sources,initial_pools=initial,historical_v5=historical,references=old_candidates['references'],historical_matched_control=matched,scope='CSAW-S subject3fold OOF124 seed42; no held/test/selection',selection=None,held_allowed=False)
    if (ROOT/'protocol.json').exists():assert read(ROOT/'protocol.json')==protocol
    active=next(x for x in plan['variants'] if x['id']==variant)
    return dict(protocol=protocol,signature=digest(protocol),rows=rows,folds=folds_record['folds'],variant=variant,learning_config=dict(cfg,**active),initial_pools=initial,historical_v5=historical,references=old_candidates['references'],historical_matched_control=matched)

def job_signature(c,fold,smoke=False):
    return digest(dict(source=c['signature'],fold=spec(c,fold),variant=c['variant'],model=BACKBONE,smoke=bool(smoke)))

def initial_pool(c,fold):
    import numpy as np
    m=c['initial_pools'][str(fold)];assert checksum(m['path'])==m['sha256']
    with np.load(m['path']) as z:return {k:z[k] for k in z.files}

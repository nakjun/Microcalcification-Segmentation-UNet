"""Independent sparse-set/8-neighbour audit; never imports evaluator helpers."""
from pathlib import Path
import argparse
import numpy as np
from common import ROOT, ARMS, read, checksum, context, verify, atomic_json, prediction_dir

POLICIES = ('primary', 'secondary_v3', 'ungated_p5')
BUCKETS = ('1-4', '5-16', '17-64', '>64')


def components(ys, xs, width):
    remaining = set(zip(map(int, ys), map(int, xs)))
    assert len(remaining) == len(ys)
    result = []
    while remaining:
        seed = remaining.pop()
        stack, points = [seed], []
        while stack:
            y, x = stack.pop()
            points.append(y * width + x)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (y + dy, x + dx)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
        result.append(np.asarray(points, np.int64))
    return result


def divide(a, b):
    return float(a / b) if b else None


def independent_counts(pred, gt, groups, fill):
    overlap = np.intersect1d(pred, gt, assume_unique=True)
    tp, total, truth = len(overlap), len(pred), len(gt)
    inside = int(fill[overlap].sum())
    predicted_inside = int(fill[pred].sum())
    m = dict(TP=tp, FP=total-tp, FN=truth-tp, gt_pixels=truth,
             predicted_pixels=total, Dice=divide(2*tp, total+truth),
             IoU=divide(tp, total+truth-tp), TP_inside_boxes=inside,
             TP_outside_boxes=tp-inside, FP_inside_boxes=predicted_inside-inside,
             FP_outside_boxes=total-tp-predicted_inside+inside)
    bins = {k: dict(total=0, any_overlap=0, half_overlap=0) for k in BUCKETS}
    for group in groups:
        size = len(group)
        key = '1-4' if size <= 4 else '5-16' if size <= 16 else '17-64' if size <= 64 else '>64'
        hit = len(np.intersect1d(pred, group, assume_unique=True))
        bins[key]['total'] += 1
        bins[key]['any_overlap'] += int(hit > 0)
        bins[key]['half_overlap'] += int(hit*2 >= size)
    m['components'] = bins
    return m


def summary(cases):
    pos = [c for c in cases if c['gt_pixels']]
    neg = [c for c in cases if not c['gt_pixels']]
    tp, fp, fn = [sum(c[k] for c in cases) for k in ('TP', 'FP', 'FN')]
    nfp = sum(c['FP'] for c in neg)
    nimg = sum(c['FP'] > 0 for c in neg)
    m = dict(positive_images=len(pos), negative_images=len(neg),
             positive_macro_Dice=float(np.mean([c['Dice'] for c in pos])),
             positive_macro_IoU=float(np.mean([c['IoU'] for c in pos])),
             all_micro_Dice=divide(2*tp, 2*tp+fp+fn), all_micro_IoU=divide(tp, tp+fp+fn),
             TP=tp, FP=fp, FN=fn, pixel_recall=divide(tp,tp+fn), pixel_precision=divide(tp,tp+fp),
             negative_image_FP_rate=divide(nimg,len(neg)), negative_FP_pixels_per_image=divide(nfp,len(neg)),
             negative_FP_pixels=nfp, negative_images_with_FP=nimg,
             positive_images_with_TP=sum(c['TP']>0 for c in pos),
             TP_inside_boxes=sum(c['TP_inside_boxes'] for c in cases),
             TP_outside_boxes=sum(c['TP_outside_boxes'] for c in cases))
    m['components'] = {b: {k: sum(c['components'][b][k] for c in cases)
                          for k in ('total','any_overlap','half_overlap')} for b in BUCKETS}
    return m


def compare(expected, actual, where='root'):
    if isinstance(expected, dict):
        for key, value in expected.items():
            compare(value, actual[key], where+'.'+key)
    elif expected is None:
        assert actual is None, (where, expected, actual)
    elif isinstance(expected, float):
        assert np.isclose(expected,actual,rtol=0,atol=1e-12), (where,expected,actual)
    else:
        assert expected == actual, (where,expected,actual)


def flatten(value, prefix=''):
    if isinstance(value, dict):
        out = {}
        for k,v in value.items(): out.update(flatten(v,prefix+'.'+k if prefix else k))
        return out
    return {prefix:value}


def self_test():
    ys=np.asarray([0,1,3,3]); xs=np.asarray([0,1,0,1])
    groups=components(ys,xs,5)
    assert sorted(map(len,groups))==[2,2]
    m=independent_counts(np.asarray([0,6,19]),ys*5+xs,groups,np.zeros(25,bool))
    assert (m['TP'],m['FP'],m['FN'])==(2,1,2)
    assert m['components']['1-4']==dict(total=2,any_overlap=1,half_overlap=1)
    print('independent sparse audit self-test PASS')


def bootstrap_self_test():
    from report_math import bootstrap, FIELDS
    studies=['a','a','b','c']
    x=np.zeros((2,3,4,6),float)
    for condition in range(2):
        for fold in range(3):
            for image,truth in enumerate((7,0,11,0)):
                tp=min(truth,1+fold+condition*(fold+1)) if truth else 0
                fp=3+fold+image-condition
                x[condition,fold,image]=[int(truth>0),2*tp/(truth+tp+fp) if truth else 0,
                                        tp,truth-tp,fp if not truth else 0,int(not truth and fp>0)]
    reps,seed=123,93271
    actual=bootstrap(x,studies,replicates=reps,seed=seed)
    rng=np.random.default_rng(seed);ids=sorted(set(studies));deltas=[]
    for draw in rng.integers(len(ids),size=(reps,len(ids))):
        selected=[i for s in draw for i,name in enumerate(studies) if name==ids[s]]
        fold_differences=[]
        for fold in range(3):
            values=[]
            for condition in range(2):
                total=x[condition,fold,selected].sum(axis=0)
                values.append([total[1]/total[0] if total[0] else np.nan,
                               total[2]/(total[2]+total[3]) if total[2]+total[3] else np.nan,
                               total[4],total[5]])
            fold_differences.append(np.asarray(values[1])-values[0])
        deltas.append(np.mean(fold_differences,axis=0))
    deltas=np.asarray(deltas)
    for i,key in enumerate(FIELDS):
        finite=deltas[np.isfinite(deltas[:,i]),i]
        compare(dict(lower=float(np.percentile(finite,2.5)),upper=float(np.percentile(finite,97.5)),
                     valid_replicates=len(finite)),actual[key],key)
    assert actual['positive_macro_Dice']['valid_replicates']<reps
    print('independent scalar paired bootstrap audit PASS')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    self_test();bootstrap_self_test()
    if args.self_test:return
    c=context();done=verify(ROOT/'evaluation/complete.json');assert done['inference_signature']==c['signature']
    assert done['checkpoints']==48 and done['images_per_fold']==32 and done['ensemble'] is False
    full=read(c['protocol']['manifest']);held={r['image_id']:r for r in full if r['role']!='train'}
    truth={}
    for ident,r in held.items():
        assert checksum(r['target_path'])==r['target_sha256']
        with np.load(r['target_path']) as z:ys,xs=z['y'],z['x']
        truth[ident]=(np.sort(ys*r['width']+xs),components(ys,xs,r['width']))
    results={};checked=0
    for cp in c['checkpoints']:
        raw={(a,p):[] for a in ARMS for p in POLICIES}
        for ident,r in held.items():
            shape=(r['height'],r['width']);size=int(np.prod(shape));gt,groups=truth[ident]
            predpath=prediction_dir(cp)/'predictions'/f'{ident}.npz'
            casepath=ROOT/'evaluation/cases'/cp['model']/cp['variant']/f"fold_{cp['fold']}"/f'{ident}.json'
            case=read(casepath);assert case['signature']==done['signature']
            assert case['checkpoint_sha256']==cp['checkpoint_sha256'] and case['GT_sha256']==r['target_sha256']
            assert case['probability_sha256']==checksum(predpath)
            with np.load(predpath) as z:
                flat,prob=z['flat'],z['probability']
                assert tuple(z['shape'])==shape and flat.dtype==np.int64 and prob.dtype==np.float32
                assert flat.shape==prob.shape and np.all(flat[1:]>flat[:-1]) and np.all((flat>=0)&(flat<size))
                assert np.all(np.isfinite(prob)) and np.all((prob>=np.float32(.3))&(prob<=1))
                for arm in ARMS:
                    base=read(Path('runs/calcification_refinement_v1/baseline_development/csaw')/arm/'gated/cases'/f'{ident}.json')
                    assert not base['gt_read_for_inference'] and checksum(base['mask_path'])==base['mask_sha256']
                    with np.load(base['mask_path']) as b:
                        fill=np.unpackbits(b['box_fill'],count=size).astype(bool)
                    assert np.array_equal(fill,np.unpackbits(z['fill_'+arm],count=size).astype(bool))
                    gate=bool(z['gate_'+arm]);assert gate==base['gate_pass']
                    policies=dict(primary=flat[prob>=np.float32(.9)] if gate else np.empty(0,np.int64),
                                  secondary_v3=flat[fill[flat]|(prob>=np.float32(.9))] if gate else np.empty(0,np.int64),
                                  ungated_p5=flat[prob>=np.float32(.5)])
                    for policy,pred in policies.items():
                        original=case['arms'][arm][policy];mp=original['mask_path']
                        assert checksum(mp)==original['mask_sha256']
                        with np.load(mp) as mask:
                            assert tuple(mask['shape'])==shape
                            actual=np.flatnonzero(np.unpackbits(mask['mask'],count=size))
                        np.testing.assert_array_equal(actual,pred)
                        expected=independent_counts(pred,gt,groups,fill)
                        compare(expected,original,f'{cp["model"]}/{cp["variant"]}/{cp["fold"]}/{ident}/{arm}/{policy}')
                        raw[arm,policy].append(expected);checked+=1
        for (arm,policy),records in raw.items():
            expected=summary(records)
            condition=done['conditions'][cp['model']][cp['variant']]
            fold=next(f for f in condition['folds'] if f['fold']==cp['fold'])
            compare(expected,fold['metrics'][policy][arm])
            results[cp['model'],cp['variant'],cp['fold'],arm,policy]=expected
        print(f'independent masks/counts {checked}/9216',flush=True)
    for model,conditions in done['conditions'].items():
        for variant,condition in conditions.items():
            for arm in ARMS:
                for policy in POLICIES:
                    rows=[flatten(results[model,variant,f,arm,policy]) for f in range(3)]
                    for key in rows[0]:
                        values=[x[key] for x in rows if x[key] is not None]
                        if not values:continue
                        expected=dict(mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)) if len(values)>1 else None)
                        compare(expected,condition['fold_mean_sample_sd'][policy][arm][key])
    assert checked==9216
    out=Path('reports/comparative_held32/independent_verification.json')
    atomic_json(out,dict(complete=True,verified_masks=checked,checkpoints=48,images_per_checkpoint=32,
                        independent_sparse_counts=True,independent_eight_neighbor_components=True,
                        fold_aggregates_and_sample_sd=True,inference_signature=c['signature'],
                        evaluation_signature=done['signature'],evaluation_sha256=checksum(ROOT/'evaluation/complete.json'),
                        verifier_sha256=checksum(__file__)))
    print(out)


if __name__=='__main__':main()

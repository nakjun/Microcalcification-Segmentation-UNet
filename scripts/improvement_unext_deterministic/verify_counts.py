"""Parent independent native masks, connected components and aggregate verification."""
import numpy as np
from scipy import ndimage
from common import ROOT,OUT,VARIANTS,context,read,verify,checksum
from calcification_gate_common import atomic_json
POLICIES=('ungated','primary:imagenet','primary:mammo','secondary_v3:imagenet','secondary_v3:mammo')

def verify_effects(metrics, reported, field):
    """Recompute contrasts from verified raw aggregates, without evaluator helpers."""
    scalar_keys = ('positive_macro_Dice', 'pixel_recall', 'TP',
                   'negative_FP_pixels', 'negative_images_with_FP')
    buckets = ('1-4', '5-16', '17-64', '>64')
    metric_keys = set(scalar_keys) | {'component_' + bucket for bucket in buckets}
    contrasts = {'LH_minus_C', 'LH_minus_H', 'LH_minus_L',
                 'LH_minus_L_minus_H_plus_C'}
    assert set(metrics) == {'C', 'L', 'H', 'LH'}, field
    assert set(reported) == {'imagenet', 'mammo'}, field
    for arm in ('imagenet', 'mammo'):
        assert set(reported[arm]) == contrasts, (field, arm)
        values = {}
        for variant in ('C', 'L', 'H', 'LH'):
            raw = metrics[variant][arm]
            values[variant] = {key: raw[key] for key in scalar_keys}
            for bucket in buckets:
                values[variant]['component_' + bucket] = raw['components'][bucket]['any_overlap']
            assert all(np.isfinite(value) for value in values[variant].values()), (field, arm, variant)
        for name in contrasts:
            assert set(reported[arm][name]) == metric_keys, (field, arm, name)
            for key in metric_keys:
                if name == 'LH_minus_L_minus_H_plus_C':
                    # Preserve the declared FP arithmetic order for exact comparison.
                    expected = values['LH'][key] - values['L'][key]
                    expected -= values['H'][key]
                    expected += values['C'][key]
                else:
                    baseline = name.rsplit('_', 1)[1]
                    expected = values['LH'][key] - values[baseline][key]
                assert reported[arm][name][key] == expected, (field, arm, name, key)

def counts(c):
    done = verify(ROOT / 'oof/complete.json', c['signature'])
    lookup = {r['image_id']: r for r in c['rows']}
    comparisons = 0
    files = {}
    verified_records = {(m, p): [] for m in VARIANTS
                        for p in POLICIES}
    for f in c['folds']:
        for ident in f['validation_ids']:
            row = lookup[ident]
            assert checksum(row['target_path']) == row['target_sha256']
            shape = (row['height'], row['width'])
            gt = np.zeros(shape, bool)
            with np.load(row['target_path']) as z:
                gt[z['y'], z['x']] = True
            labels, n = ndimage.label(gt, np.ones((3, 3)))
            sizes = np.bincount(labels.ravel(), minlength=n+1)[1:]
            cp = ROOT / 'oof/cases' / f'{ident}.json'
            case = read(cp)
            assert case['signature'] == c['signature']
            for model in VARIANTS:
                path = ROOT / 'jobs' / model / f'fold_{f["fold"]}' / 'predictions' / f'{ident}.npz'
                with np.load(path) as z:
                    data = {k: z[k] for k in z.files}
                prob = np.zeros(np.prod(shape), np.float32)
                prob[data['flat']] = data['probability']
                prob = prob.reshape(shape)
                for policy in POLICIES:
                    if policy == 'ungated':
                        mask = prob >= np.float32(.5)
                        expected = case['ungated'][model]
                    else:
                        kind,arm=policy.split(':')
                        fill=np.unpackbits(data['fill_'+arm],count=prob.size).reshape(shape).astype(bool)
                        mask=(prob>=np.float32(.9) if kind=='primary' else np.where(fill,prob>=np.float32(.3),prob>=np.float32(.9))) & bool(data['gate_'+arm])
                        expected=case[kind][arm][model]
                    tp, fp, ng = int((mask & gt).sum()), int((mask & ~gt).sum()), int(gt.sum())
                    raw = dict(TP=tp, FP=fp, FN=ng-tp, gt_pixels=ng, predicted_pixels=tp+fp,
                               Dice=2*tp/(tp+fp+ng) if tp+fp+ng else None,
                               IoU=tp/(ng+fp) if ng+fp else None)
                    for k, v in raw.items():
                        assert expected[k] == v, (ident, model, policy, k)
                    region = np.zeros(shape, bool) if policy == 'ungated' else fill
                    for key, value in [('TP_inside_boxes', mask & gt & region),
                                       ('TP_outside_boxes', mask & gt & ~region),
                                       ('FP_inside_boxes', mask & ~gt & region),
                                       ('FP_outside_boxes', mask & ~gt & ~region)]:
                        assert expected[key] == int(value.sum()), (ident, model, policy, key)
                    overlap = np.bincount(labels[mask], minlength=n+1)[1:]
                    for key, sel in [('1-4', sizes<=4), ('5-16', (sizes>4)&(sizes<=16)),
                                     ('17-64', (sizes>16)&(sizes<=64)), ('>64', sizes>64)]:
                        actual = dict(total=int(sel.sum()), any_overlap=int((overlap[sel]>0).sum()),
                                      half_overlap=int((2*overlap[sel]>=sizes[sel]).sum()))
                        assert expected['components'][key] == actual, (ident, model, policy, key)
                    verified_records[(model, policy)].append(expected)
                    comparisons += 1
            files[str(cp)] = checksum(cp)
            print('independent masks', comparisons, '/2480', flush=True)
    assert comparisons == 2480
    for (model, policy), records in verified_records.items():
        aggregate = done['ungated_p5'][model] if policy == 'ungated' else done['pipeline' if policy.split(':')[0]=='primary' else 'secondary_v3'][model][policy.split(':')[1]]
        positive = [r for r in records if r['gt_pixels'] > 0]
        negative = [r for r in records if r['gt_pixels'] == 0]
        tp, fp, fn = (sum(r[k] for r in records) for k in ('TP', 'FP', 'FN'))
        values = dict(TP=tp, FP=fp, FN=fn, positive_images=len(positive), negative_images=len(negative),
                      positive_macro_Dice=float(np.mean([r['Dice'] for r in positive])),
                      positive_macro_IoU=float(np.mean([r['IoU'] for r in positive])),
                      all_micro_Dice=2*tp/(2*tp+fp+fn), all_micro_IoU=tp/(tp+fp+fn),
                      pixel_recall=tp/(tp+fn), pixel_precision=tp/(tp+fp) if tp+fp else None,
                      negative_FP_pixels=sum(r['FP'] for r in negative),
                      negative_images_with_FP=sum(r['FP']>0 for r in negative),
                      positive_images_with_TP=sum(r['TP']>0 for r in positive))
        for key, value in values.items():
            assert aggregate[key] == value, (model, policy, key)
        for bucket in ('1-4', '5-16', '17-64', '>64'):
            for key in ('total', 'any_overlap', 'half_overlap'):
                assert aggregate['components'][bucket][key] == sum(r['components'][bucket][key] for r in records)
    assert done['historical_v5']==c['historical_v5']
    verify_effects(done['pipeline'], done['paired_effects'], 'paired_effects')
    verify_effects(c['historical_v5'], done['historical_v5_effects'], 'historical_v5_effects')
    def local(m,b):
        checks={'Dice_not_strictly_above_control':m['positive_macro_Dice']>b['positive_macro_Dice'],'TP_below_control':m['TP']>=b['TP'],'positive_images_with_TP_below_control':m['positive_images_with_TP']>=b['positive_images_with_TP'],'negative_FP_pixels_above_control':m['negative_FP_pixels']<=b['negative_FP_pixels'],'negative_images_with_FP_above_control':m['negative_images_with_FP']<=b['negative_images_with_FP']}
        checks.update({f'component_{k}_below_control':m['components'][k]['any_overlap']>=b['components'][k]['any_overlap'] for k in b['components']})
        return {k for k,v in checks.items() if not v}
    for variant in VARIANTS:
        for arm in ('imagenet','mammo'):
            m=done['pipeline'][variant][arm];secondary=done['secondary_v3'][variant][arm]
            if variant!='C':assert set(done['within_backbone_failures'][variant][arm])==local(m,done['pipeline']['C'][arm])
            assert set(done['historical_local_failures'][variant][arm])==local(m,c['historical_matched_control'][arm])
            assert set(done['secondary_v3_local_failures'][variant][arm])==local(secondary,c['historical_matched_control'][arm])
            for field,value in [('global_constraint_failures',m),('secondary_v3_global_constraint_failures',secondary)]:
                v2,v4=c['references']['v2'][arm],c['references']['v4b'][arm]
                check={'recall_below_v2':value['TP']>=v2['TP'],'Dice_not_above_v2':value['positive_macro_Dice']>v2['positive_macro_Dice'],'positive_overlap_below_v2':value['positive_images_with_TP']>=v2['positive_images_with_TP'],'negative_FP_area_above_v4b':value['negative_FP_pixels']<=v4['negative_FP_pixels'],'negative_FP_images_above_v4b':value['negative_images_with_FP']<=v4['negative_images_with_FP'],'Dice_not_above_v4b':value['positive_macro_Dice']>v4['positive_macro_Dice']}
                check.update({f'component_{k}_below_v2':value['components'][k]['any_overlap']>=v2['components'][k]['any_overlap'] for k in ('1-4','5-16')})
                assert set(done[field][variant][arm])=={k for k,v in check.items() if not v}
    atomic_json(OUT/'count_verification.json',dict(complete=True,signature=c['signature'],mask_comparisons=comparisons,all_pixel_and_component_counts_exact=True,independent_aggregates_exact=True,success_conditions_independently_checked=True,oof_sha256=checksum(ROOT/'oof/complete.json'),verifier_sha256=checksum(__file__),output_files=files))

if __name__=='__main__':
    counts(context());print('INDEPENDENT 2480 masks PASS',flush=True)

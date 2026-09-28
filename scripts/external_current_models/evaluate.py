"""Read external labels only after a complete image-only prediction cohort is verified."""
import argparse
import fcntl
import sys
from pathlib import Path

import numpy as np
from common import (CONDITIONS, DATASETS, REPO, RUN, SCRIPTS, context, digest,
                    load_module, prediction_dir, read, read_mask, sha, verify, write)


def legacy_endpoints():
    """Reuse exact retained endpoint functions without calling their gated driver."""
    previous = sys.modules.get('common')
    try:
        module = load_module('current_retained_v3_common',SCRIPTS/'external_v3/common.py')
        sys.modules['common'] = module
        return load_module('current_retained_v3_evaluate',SCRIPTS/'external_v3/evaluate.py')
    finally:
        if previous is None:
            sys.modules.pop('common',None)
        else:
            sys.modules['common']=previous


def summarize(values,dataset):
    def mean(items):
        return float(np.mean(items)) if items else None
    result={k:sum(v[k] for v in values) for k in ('predicted_pixels','predicted_components','image_predicted_positive')}
    result.update(mean_predicted_pixels=mean([v['predicted_pixels'] for v in values]),
                  mean_predicted_fraction=mean([v['predicted_fraction'] for v in values]))
    if dataset=='inbreast':
        for k in ('points_total','points_exact','points_within1','contours_total','contours_any_overlap','contour_union_pixels','contour_overlap_pixels'):
            result[k]=sum(v[k] for v in values)
        result.update(point_exact_hit_rate=result['points_exact']/result['points_total'] if result['points_total'] else None,
                      point_within1_hit_rate=result['points_within1']/result['points_total'] if result['points_total'] else None,
                      contour_annotated_images=sum(v['contour_union_pixels']>0 for v in values),
                      partial_contour_macro_Dice=mean([v['partial_contour_Dice'] for v in values if v['partial_contour_Dice'] is not None]),
                      partial_contour_macro_IoU=mean([v['partial_contour_IoU'] for v in values if v['partial_contour_IoU'] is not None]),
                      predicted_pixels_outside_contour_union_not_confirmed_FP=sum(v['predicted_pixels_outside_contour_union_not_confirmed_FP'] or 0 for v in values))
    else:
        for k in ('ROIs_total','ROIs_any_overlap','ROI_union_pixels','ROI_union_overlap_pixels'):
            result[k]=sum(v[k] for v in values)
        result.update(ROI_any_overlap_rate=result['ROIs_any_overlap']/result['ROIs_total'] if result['ROIs_total'] else None,
                      ROI_macro_coverage=mean([r['coverage'] for v in values for r in v['ROIs']]),
                      ROI_union_micro_coverage=result['ROI_union_overlap_pixels']/result['ROI_union_pixels'] if result['ROI_union_pixels'] else None,
                      predicted_pixels_outside_ROI_union_unannotated=sum(v['predicted_pixels_outside_ROI_union_unannotated'] or 0 for v in values))
    return result


def verified_prediction_cohort(dataset,record,signature,rows):
    folder=prediction_dir(dataset,record['condition'],record['fold'])
    done=verify(folder/'complete.json',signature)
    assert not done['smoke'] and done['GT_read'] is False and done['threshold']==.5 and done['gate'] is None
    assert done['image_ids']==[r['image_id'] for r in rows] and done['checkpoints']==record['checkpoints']
    expected={}
    for row in rows:
        path=folder/(row['image_id']+'.npz')
        marker=path.with_suffix('.json')
        item=verify(marker,signature)
        assert item['image_id']==row['image_id'] and item['dicom_sha256']==row['dicom_sha256']
        assert item['checkpoints']==record['checkpoints'] and item['GT_read'] is False and not item['smoke']
        assert item['threshold']==.5 and item['gate'] is None and item['output_files']=={str(path):sha(path)}
        assert item['dataset']==dataset and item['condition']==record['condition'] and item['fold']==record['fold']
        expected[str(path)]=sha(path);expected[str(marker)]=sha(marker)
    assert done['output_files']==expected
    return folder


def evaluate(dataset,record,signature,protocol):
    rows=read(protocol['inventory'][dataset])
    # Every image must be committed before annotation JSON or masks are opened.
    folder=verified_prediction_cohort(dataset,record,signature,rows)
    annotation_path=REPO/f'runs/external_v3/annotations/{dataset}.json'
    annotation=read(annotation_path)
    assert annotation['complete'] and annotation['images']==len(rows) and annotation['dataset']==dataset
    for path,digest_ in annotation['source_files'].items():
        assert sha(path)==digest_,path
    if dataset=='inbreast':
        assert annotation['summary']['valid_points']==4339 and annotation['summary']['valid_contours']==2801
        assert annotation['summary']['missing_xml']==67 and annotation['summary']['xml_images']==343
    else:
        assert annotation['summary']['valid_ROIs']==324 and annotation['summary']['excluded_ROI_series']==2
    lookup={r['image_id']:r for r in annotation['records']}
    assert set(lookup)=={r['image_id'] for r in rows}
    endpoints=legacy_endpoints()
    destination=RUN/dataset/'evaluation'/record['condition']/f"fold_{record['fold']}"
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        evaluation_protocol=dict(inference_signature=signature,annotation_path=str(annotation_path),annotation_sha256=sha(annotation_path),
                                 prediction_complete_sha256=sha(folder/'complete.json'),endpoints=endpoints.ENDPOINTS,
                                 policy='current methods ungated p>=0.5; no external_v3 gate or detector union',
                                 evaluator_sha256=sha(__file__),retained_evaluator_sha256=sha(SCRIPTS/'external_v3/evaluate.py'))
        esig=digest(evaluation_protocol)
        write(destination/'protocol.json',evaluation_protocol)
        records=[];outputs={str(destination/'protocol.json'):sha(destination/'protocol.json')}
        for index,row in enumerate(rows):
            path=folder/(row['image_id']+'.npz')
            casepath=destination/'cases'/(row['image_id']+'.json')
            if casepath.exists():
                case=read(casepath)
                assert case['signature']==esig and case['prediction_sha256']==sha(path)
            else:
                a=lookup[row['image_id']]
                shape=(row['height'],row['width'])
                assert (a['height'],a['width'])==shape
                case=dict(signature=esig,dataset=dataset,image_id=row['image_id'],fold=record['fold'],condition=record['condition'],
                          prediction_sha256=sha(path),annotation_status=a['status'],annotation_issues=a['issues'],
                          metrics=endpoints.metrics(read_mask(path,shape),a,dataset))
                write(casepath,case)
            records.append(case['metrics']);outputs[str(casepath)]=sha(casepath)
            print('evaluate',dataset,record['condition'],record['fold'],index+1,len(rows),flush=True)
        write(destination/'complete.json',dict(complete=True,signature=esig,inference_signature=signature,dataset=dataset,
                                              condition=record['condition'],fold=record['fold'],images=len(rows),seed=42,epoch=16,
                                              checkpoints=record['checkpoints'],metrics=summarize(records,dataset),
                                              annotation_summary=annotation['summary'],policy='ungated_p5',gate=None,
                                              selection_performed=False,ensemble=False,clinical_sensitivity_estimated=False,
                                              output_files=outputs))


def aggregate(dataset,signature):
    conditions={}
    for condition in CONDITIONS:
        paths=[RUN/dataset/'evaluation'/condition/f'fold_{fold}/complete.json' for fold in range(3)]
        if not all(p.exists() for p in paths):
            continue
        folds=[read(path) for path in paths]
        assert all(r['complete'] and r['inference_signature']==signature and r['fold']==i for i,r in enumerate(folds))
        keys=[k for k,v in folds[0]['metrics'].items() if isinstance(v,(float,int)) and not isinstance(v,bool)]
        conditions[condition]=dict(folds=folds,fold_mean_sample_sd={k:dict(mean=float(np.mean([r['metrics'][k] for r in folds])),sample_sd=float(np.std([r['metrics'][k] for r in folds],ddof=1))) for k in keys},
                                   fold_complete_sha256={str(p):sha(p) for p in paths})
        write(RUN/dataset/'evaluation'/condition/'complete.json',dict(complete=True,inference_signature=signature,**conditions[condition]))
    if len(conditions)==len(CONDITIONS):
        write(RUN/dataset/'evaluation/complete.json',dict(complete=True,inference_signature=signature,dataset=dataset,images=DATASETS[dataset],
                                                       models=6,conditions=conditions,folds=3,seed=42,epoch=16,threshold=.5,gate=None,
                                                       ensemble=False,selection_performed=False))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=tuple(DATASETS),required=True)
    p.add_argument('--condition',choices=CONDITIONS)
    p.add_argument('--fold',type=int,choices=(0,1,2))
    p.add_argument('--aggregate-only',action='store_true')
    args=p.parse_args()
    protocol,records,signature=context(args.condition,args.fold)
    if not args.aggregate_only:
        for record in records:
            evaluate(args.dataset,record,signature,protocol)
    aggregate(args.dataset,signature)


if __name__=='__main__':
    main()

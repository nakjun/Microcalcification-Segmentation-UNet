"""Locked epoch16 OOF comparisons; no threshold/epoch/held selection."""
import fcntl
from pathlib import Path
import numpy as np
from common import ROOT,VARIANTS,ARMS,context,read,verify,checksum,immutable,job_signature
from calcification_refinement_v2_common import component_pixel_groups
from calcification_refinement_common import baseline_masks
from evaluate_calcification_refinement_v3_union import counts
from train_calcification_refinement_v2 import aggregate
from calcification_refinement_v6_common import failures

def local_failures(metrics,reference):
    result={}
    for arm in ARMS:
        m,r=metrics[arm],reference[arm]
        checks={'Dice_not_strictly_above_control':m['positive_macro_Dice']>r['positive_macro_Dice'],'TP_below_control':m['TP']>=r['TP'],'positive_images_with_TP_below_control':m['positive_images_with_TP']>=r['positive_images_with_TP'],'negative_FP_pixels_above_control':m['negative_FP_pixels']<=r['negative_FP_pixels'],'negative_images_with_FP_above_control':m['negative_images_with_FP']<=r['negative_images_with_FP']}
        checks.update({f'component_{k}_below_control':m['components'][k]['any_overlap']>=r['components'][k]['any_overlap'] for k in r['components']})
        result[arm]=[k for k,v in checks.items() if not v]
    return result

def verify_predictions(c):
    lookup={r['image_id']:r for r in c['rows']}
    for variant in VARIANTS:
        active=dict(c,variant=variant)
        for fold in c['folds']:
            fi=fold['fold'];job=ROOT/'jobs'/variant/f'fold_{fi}';sig=job_signature(active,fi)
            done=verify(job/'complete.json',sig);pred=verify(job/'predictions/complete.json',c['signature'])
            assert done['epochs']==16 and not done['smoke'] and done['job']['train_ids']==fold['train_ids'] and done['job']['validation_ids']==fold['validation_ids']
            assert pred['image_ids']==fold['validation_ids'] and pred['variant']==variant and not pred['GT_read']
            assert pred['checkpoint_sha256']==checksum(job/'checkpoint.pt') and pred['job_signature']==sig
            for ident in fold['validation_ids']:
                path=job/'predictions'/f'{ident}.json';m=verify(path,c['signature']);assert m['image_id']==ident and m['variant']==variant and m['fold']==fi and not m['GT_read']
                assert m['checkpoint_sha256']==pred['checkpoint_sha256'] and m['job_signature']==sig and m['native_sha256']==lookup[ident]['native_cache_sha256']

def sparse(z,shape):
    f,p=z['flat'],z['probability'];assert f.dtype==np.int64 and p.dtype==np.float32 and f.shape==p.shape
    assert np.all(f[1:]>f[:-1]) and np.all((f>=0)&(f<np.prod(shape))) and np.isfinite(p).all() and np.all((p>=.3)&(p<=1))
    return f,p

def effects(metrics):
    keys=('positive_macro_Dice','pixel_recall','TP','negative_FP_pixels','negative_images_with_FP')
    result={}
    for arm in ARMS:
        vals={v:dict({k:metrics[v][arm][k] for k in keys},**{f'component_{k}':m['any_overlap'] for k,m in metrics[v][arm]['components'].items()}) for v in VARIANTS}
        contrasts={f'LH_minus_{v}':{k:vals['LH'][k]-vals[v][k] for k in vals['LH']} for v in ('C','H','L')}
        contrasts['LH_minus_L_minus_H_plus_C']={k:vals['LH'][k]-vals['L'][k]-vals['H'][k]+vals['C'][k] for k in vals['LH']};result[arm]=contrasts
    return result

def main():
    c=context();out=ROOT/'oof';out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify_predictions(c)
        lookup={r['image_id']:r for r in c['rows']};records=[];outputs={str(ROOT/'protocol.json'):checksum(ROOT/'protocol.json')}
        for v in VARIANTS:
            for f in c['folds']:
                p=ROOT/'jobs'/v/f'fold_{f["fold"]}'/'predictions/complete.json';outputs[str(p)]=checksum(p)
        for fold in c['folds']:
            fi=fold['fold']
            for ident in fold['validation_ids']:
                path=out/'cases'/f'{ident}.json'
                if path.exists():case=read(path);assert case['signature']==c['signature']
                else:
                    row=lookup[ident];shape=(row['height'],row['width']);assert checksum(row['target_path'])==row['target_sha256']
                    with np.load(row['target_path']) as target:ys,xs=target['y'],target['x']
                    groups=component_pixel_groups(ys,xs,shape) if len(ys) else []
                    case=dict(signature=c['signature'],image_id=ident,fold=fi,shape=list(shape),GT_sha256=row['target_sha256'],primary={a:{} for a in ARMS},secondary_v3={a:{} for a in ARMS},ungated={})
                    for variant in VARIANTS:
                        prediction=ROOT/'jobs'/variant/f'fold_{fi}'/'predictions'/f'{ident}.npz'
                        with np.load(prediction) as z:
                            assert tuple(z['shape'])==shape;flat,p=sparse(z,shape)
                            mask=np.zeros(np.prod(shape),bool);mask[flat[p>=np.float32(.5)]]=True
                            case['ungated'][variant]=counts(mask.reshape(shape),ys,xs,np.zeros(shape,bool),groups)
                            for arm in ARMS:
                                fill=np.unpackbits(z['fill_'+arm],count=int(np.prod(shape))).reshape(shape).astype(bool);gate=bool(z['gate_'+arm]);_,expected,base=baseline_masks(c['learning_config']['development_baseline_root'],arm,ident)
                                assert np.array_equal(fill,expected) and gate==bool(base['gate_pass'])
                                primary=np.zeros(np.prod(shape),bool);secondary=np.zeros_like(primary)
                                if gate:
                                    primary[flat[p>=np.float32(.9)]]=True;secondary[flat[fill.ravel()[flat]|(p>=np.float32(.9))]]=True
                                case['primary'][arm][variant]=counts(primary.reshape(shape),ys,xs,fill,groups)
                                case['secondary_v3'][arm][variant]=counts(secondary.reshape(shape),ys,xs,fill,groups)
                    immutable(path,case)
                records.append(case);outputs[str(path)]=checksum(path);print(f'Study OOF {len(records)}/124',flush=True)
        def agg(policy):return {v:{a:aggregate([{v:r[policy][a][v]} for r in records],v) for a in ARMS} for v in VARIANTS}
        primary=agg('primary');secondary=agg('secondary_v3');ungated={v:aggregate([{v:r['ungated'][v]} for r in records],v) for v in VARIANTS}
        result=dict(complete=True,signature=c['signature'],images=124,policy='epoch16/gate/full_image@0.9',pipeline=primary,secondary_v3=secondary,ungated_p5=ungated,historical_v5=c['historical_v5'],secondary_v3_local_failures={v:local_failures(secondary[v],c['historical_matched_control']) for v in VARIANTS},secondary_v3_global_constraint_failures={v:failures(secondary[v],c['references']['v2'],c['references']['v4b']) for v in VARIANTS},within_backbone_failures={v:local_failures(primary[v],primary['C']) for v in VARIANTS if v!='C'},historical_local_failures={v:local_failures(primary[v],c['historical_matched_control']) for v in VARIANTS},global_constraint_failures={v:failures(primary[v],c['references']['v2'],c['references']['v4b']) for v in VARIANTS},paired_effects=effects(primary),historical_v5_effects=effects(c['historical_v5']),selected=None,held_allowed=False,output_files=outputs)
        immutable(out/'complete.json',result)
if __name__=='__main__':main()

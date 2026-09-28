"""Common original-coordinate COCO AP/AR and score-swept box FROC."""
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def rounded(value):
    if isinstance(value,dict):return {str(k):rounded(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [rounded(v) for v in value]
    if isinstance(value,(float,np.floating)):return round(float(value),3) if np.isfinite(value) else None
    if isinstance(value,np.integer):return int(value)
    return value


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(rounded(value),ensure_ascii=False,indent=2));tmp.replace(path)


def ious(a,b):
    a=np.asarray(a,dtype=float).reshape(-1,4);b=np.asarray(b,dtype=float).reshape(-1,4)
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.maximum(hi-lo,0).prod(2)
    aa=np.maximum(a[:,2:]-a[:,:2],0).prod(1);bb=np.maximum(b[:,2:]-b[:,:2],0).prod(1)
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-12)


def original_boxes(pred,row):
    p=np.asarray(pred,dtype=float).reshape(-1,5).copy()
    if not len(p):return p
    p[:,[0,2]]=(p[:,[0,2]]-row['pad_x'])/row['sx']
    p[:,[1,3]]=(p[:,[1,3]]-row['pad_y'])/row['sy']
    p[:,[0,2]]=p[:,[0,2]].clip(0,row['width']);p[:,[1,3]]=p[:,[1,3]].clip(0,row['height'])
    p=p[(p[:,2]>p[:,0])&(p[:,3]>p[:,1])&np.isfinite(p).all(1)]
    return p[np.argsort(-p[:,4],kind='stable')][:100]


def coco_metrics(rows,predictions):
    dataset={'images':[],'annotations':[],'categories':[{'id':1,'name':'석회화'}],'info':{}}
    detection=[];ann_id=1
    for i,r in enumerate(rows,1):
        dataset['images'].append({'id':i,'height':r['height'],'width':r['width']})
        for b in r['boxes']:
            x,y,u,v=b
            dataset['annotations'].append(dict(id=ann_id,image_id=i,category_id=1,bbox=[x,y,u-x,v-y],area=(u-x)*(v-y),iscrowd=0));ann_id+=1
        for x,y,u,v,s in predictions.get(r['image_id'],[]):
            detection.append(dict(image_id=i,category_id=1,bbox=[float(x),float(y),float(u-x),float(v-y)],score=float(s)))
    names=['AP50_95','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large']
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO();gt.dataset=dataset;gt.createIndex()
        if detection:dt=gt.loadRes(detection)
        else:
            dt=COCO();dt.dataset={**dataset,'annotations':[]};dt.createIndex()
        ev=COCOeval(gt,dt,'bbox');ev.evaluate();ev.accumulate();ev.summarize()
    return {k:float(v) if v>=0 else None for k,v in zip(names,ev.stats)}


def matched_records(rows,predictions):
    records=[]
    for r in rows:
        pred=np.asarray(predictions.get(r['image_id'],[]),dtype=float).reshape(-1,5)
        pred=pred[np.argsort(-pred[:,4],kind='stable')][:100]
        overlaps=ious(pred[:,:4],r['boxes']);used=set()
        for j,p in enumerate(pred):
            eligible=[k for k in range(len(r['boxes'])) if k not in used and overlaps[j,k]>=0.5]
            tp=bool(eligible)
            if tp:used.add(max(eligible,key=lambda k:overlaps[j,k]))
            records.append((float(p[4]),int(tp),r['study_id']))
    return records


def operating_point(rows,predictions,budget=1.0):
    rec=sorted(matched_records(rows,predictions),key=lambda x:-x[0])
    ng=sum(len(r['boxes']) for r in rows);n=len(rows)
    tp=fp=0;best={'threshold':1.001,'sensitivity':0.,'fp_per_image':0.}
    i=0
    while i<len(rec):
        score=rec[i][0]
        while i<len(rec) and rec[i][0]==score:
            tp+=rec[i][1];fp+=1-rec[i][1];i+=1
        if fp/n<=budget and tp/max(ng,1)>best['sensitivity']:
            best={'threshold':score,'sensitivity':tp/max(ng,1),'fp_per_image':fp/n}
    return best


def fixed_metrics(rows,predictions,threshold):
    rec=matched_records(rows,predictions);kept=[r for r in rec if r[0]>=threshold]
    tp=sum(r[1] for r in kept);fp=len(kept)-tp;ng=sum(len(r['boxes']) for r in rows)
    precision=tp/max(tp+fp,1);recall=tp/max(ng,1)
    return dict(precision=precision,recall=recall,F1=2*precision*recall/max(precision+recall,1e-12),
                TP=tp,FP=fp,FN=ng-tp,FP_per_image=fp/len(rows),proposals_per_image=len(kept)/len(rows))


def bootstrap_operating_ci(rows,predictions,threshold,n_boot=2000):
    groups=sorted({r['study_id'] for r in rows});index={s:i for i,s in enumerate(groups)}
    counts=np.zeros((len(groups),4)) # TP,FP,GT,images
    for r in rows:
        counts[index[r['study_id']],2]+=len(r['boxes']);counts[index[r['study_id']],3]+=1
    for score,tp,s in matched_records(rows,predictions):
        if score>=threshold:counts[index[s],0 if tp else 1]+=1
    rng=np.random.default_rng(2026);samples=[]
    for _ in range(n_boot):
        tp,fp,ng,ni=counts[rng.integers(0,len(groups),len(groups))].sum(0)
        if ng:samples.append([tp/ng,fp/ni,tp/max(tp+fp,1)])
    ci=np.percentile(samples,[2.5,97.5],axis=0)
    return {k:[float(ci[0,i]),float(ci[1,i])] for i,k in enumerate(['recall','FP_per_image','precision'])}


def union_area(rectangles):
    if not rectangles:return 0.
    xs=sorted({x for b in rectangles for x in (b[0],b[2])});area=0.
    for a,b in zip(xs,xs[1:]):
        ys=sorted((r[1],r[3]) for r in rectangles if r[0]<b and r[2]>a)
        total=0.;end=-float('inf')
        for lo,hi in ys:
            total+=max(0,hi-max(lo,end));end=max(end,hi)
        area+=(b-a)*total
    return area


def coverage(rows,predictions,threshold):
    values=[]
    for r in rows:
        ps=[p for p in predictions.get(r['image_id'],[]) if p[4]>=threshold]
        for x,y,u,v in r['boxes']:
            clipped=[]
            for a,b,c,d,_ in ps:
                aa,bb,cc,dd=max(a,x),max(b,y),min(c,u),min(d,v)
                if cc>aa and dd>bb:clipped.append((aa,bb,cc,dd))
            values.append(union_area(clipped)/((u-x)*(v-y)))
    return float(np.mean(values)) if values else None


def evaluate(rows,predictions,threshold=None,bootstrap=False):
    result=coco_metrics(rows,predictions)
    ops={str(b):operating_point(rows,predictions,b) for b in [0.5,1.,2.,4.]}
    if threshold is None:threshold=ops['1.0']['threshold']
    result.update(FROC={b:v['sensitivity'] for b,v in ops.items()},threshold=threshold,
                  operating=fixed_metrics(rows,predictions,threshold),box_union_coverage=coverage(rows,predictions,threshold))
    if bootstrap:result['operating_CI95']=bootstrap_operating_ci(rows,predictions,threshold)
    return result

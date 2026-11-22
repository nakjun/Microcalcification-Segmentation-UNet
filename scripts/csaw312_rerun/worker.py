"""One immutable selection/refit stage; isolated process, no legacy output writes."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,fcntl,json,time,random,sys
from pathlib import Path
import cv2,numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import PLAN,ROOT,read,write,sha,digest,torch_save,context,source_signature,disk_guard,lr_for_epoch,better,should_stop
from sampling import PatchSampler,breast_centers
import backbone,specialized
from metrics import segmentation_counts,summarize_segmentation,detection_ap

def image(row):
 a=cv2.imread(row.get('native_cache_path',row['source_image']),cv2.IMREAD_UNCHANGED) if not row.get('native_cache_path') else np.load(row['native_cache_path'],mmap_mode='r')
 if a is None or a.dtype!=np.uint8 or a.shape!=(row['height'],row['width']):raise ValueError('Invalid image '+row['image_id'])
 return a

def mask(row):
 a=cv2.imread(row['source_mask'],cv2.IMREAD_UNCHANGED)
 if a is None or a.shape!=(row['height'],row['width']):raise ValueError('Invalid mask '+row['image_id'])
 return a>0

def cache_rows(rows,out):
 cached=[]
 for row in rows:
  r=dict(row);path=out/'cache/images'/f'{r["image_id"]}.npy';marker=path.with_suffix('.json')
  if not marker.exists():
   assert sha(r['source_image'])==r['image_sha256']
   path.parent.mkdir(parents=True,exist_ok=True)
   tmp=path.with_suffix('.tmp')
   with tmp.open('wb') as f:np.save(f,image(r),allow_pickle=False)
   os.replace(tmp,path);write(marker,dict(source_sha256=r['image_sha256'],sha256=sha(path)))
  m=read(marker);assert m['source_sha256']==r['image_sha256'] and sha(path)==m['sha256']
  assert sha(r['source_mask'])==r['mask_sha256']
  r['native_cache_path']=str(path);r['role']='train';cached.append(r)
 return cached

def initial_centers(rows,p):
 rec=read(Path(p['data_root'])/'initial_candidates.json');assert rec['data_signature']==p['data_signature']
 result={}
 for r in rows:
  iid=r['image_id'];a=image(r);result[iid+'_breast']=breast_centers(a)
  union=np.zeros(a.shape,bool)
  info=rec['images'][iid]
  for m in info['masks']:
   assert sha(m['path'])==m['sha256']
   with np.load(m['path'],allow_pickle=False) as z:
    shape=tuple(z['shape'].tolist());assert shape==a.shape
    union|=np.unpackbits(z['mask'],count=a.size).reshape(shape).astype(bool)
  pts=np.column_stack(np.nonzero(union)).astype(np.int32)
  if len(pts)>4096:pts=pts[np.sort(np.random.default_rng(42).choice(len(pts),4096,replace=False))]
  result[iid+'_hard']=pts
 return result

def state_dict(model,stage):return model.decoder.state_dict() if stage=='segmentor' else model.state_dict()
def load_weights(model,stage,state):
 (model.decoder if stage=='segmentor' else model).load_state_dict(state,strict=True)

def predictor(job,model,device,detector):
 if job['model'] in ('deepmica','yolo'):
  stage='deepmica' if job['model']=='deepmica' else job['stage']
  return lambda a:specialized.predict(stage,model,a,device,detector=detector)
 return lambda a:backbone.predict(model,a,device)

def validate(job,model,rows,device,detector,heartbeat):
 records=[];predict=None if job['stage']=='detector' else predictor(job,model,device,detector)
 for i,row in enumerate(rows):
  a=image(row)
  if job['stage']=='detector':
   boxes,scores=specialized.predict_boxes(model,a,device)
   gt=mask(row);record=dict(image_id=row['image_id'],boxes=boxes.tolist(),scores=scores.tolist(),gt_boxes=specialized.boxes_from_mask(gt.astype(np.uint8)).tolist())
  else:
   prob=predict(a);gt=mask(row);record=dict(image_id=row['image_id'],**segmentation_counts(prob,gt,.5))
  records.append(record);heartbeat(dict(phase='validation',images=i+1,total=len(rows)))
 if job['stage']=='detector':
  summary=detection_ap(records)
  metric=summary['AP50_95'];fp=0.
 else:
  summary=summarize_segmentation(records);metric=summary['positive_macro_dice'];fp=summary['negative_fp_per_image']
 return dict(metric=metric,negative_fp_per_image=fp,summary=summary,records=records)

def mine(model,rows,device,heartbeat):
 pools={}
 for ri,row in enumerate(rows):
  prob=backbone.predict(model,image(row),device);gt=mask(row);h,w=gt.shape;points=[]
  for y in range(0,h,128):
   for x in range(0,w,128):
    crop=prob[y:y+128,x:x+128];dy,dx=np.unravel_index(np.argmax(crop),crop.shape);cy,cx=y+dy,x+dx
    if prob[cy,cx]<.3:continue
    yy=int(np.clip(cy-128,0,max(0,h-256)));xx=int(np.clip(cx-128,0,max(0,w-256)))
    if not gt[yy:yy+256,xx:xx+256].any():points.append((float(prob[cy,cx]),int(cy),int(cx)))
  points.sort(key=lambda v:(-v[0],v[1],v[2]));pools[row['image_id']]=np.asarray([(y,x) for _,y,x in points[:4096]],np.int32).reshape(-1,2)
  heartbeat(dict(phase='mining',images=ri+1,total=len(rows)))
 return pools

def run(args):
 p,lookup,splits,out=context();job=p['jobs'][args.job_index];is_smoke=args.smoke
 signature,sources=source_signature(job,smoke=is_smoke)
 smoke_prefix=Path('smoke')/digest(sources)[:12]
 folder=out/(smoke_prefix if is_smoke else Path('jobs'))/job['job_id'];folder.mkdir(parents=True,exist_ok=True)
 lock=(folder/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (folder/'complete.json').exists():
  d=read(folder/'complete.json');assert d['signature']==signature
  assert sha(folder/d['checkpoint'])==d['checkpoint_sha256'];return
 if not is_smoke:
  ready=read(out/'preflight.json');assert ready['ready'] and ready['protocol_sha256']==sha(PLAN)
  assert ready['source_sha256']==sources
 device=torch.device(args.device);torch.set_num_threads(4);cv2.setNumThreads(1)
 torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
 torch.backends.cudnn.deterministic=True
 seed=job['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if device.type=='cuda':torch.cuda.set_device(device);torch.cuda.manual_seed_all(seed)
 fold=next(f for f in splits['folds'] if f['fold']==job['fold'])
 ids=fold['inner_train_ids'] if job['phase']=='selection' else fold['outer_train_ids']
 vals=[lookup[i] for i in fold['inner_val_ids']]
 if is_smoke:ids=[i for i in ids if lookup[i]['positive']][:2]+[i for i in ids if not lookup[i]['positive']][:1];vals=[r for r in vals if r['positive']][:1]+[r for r in vals if not r['positive']][:1]
 rows=cache_rows([lookup[i] for i in ids],out);stage=job['stage'];special=job['model'] in ('deepmica','yolo')
 if job['phase']=='selection' or is_smoke:
  for r in vals:
   assert sha(r['source_image'])==r['image_sha256'] and sha(r['source_mask'])==r['mask_sha256']
 if special:model,opt,criterion=specialized.build('deepmica' if job['model']=='deepmica' else stage,device,p['sam_checkpoint'])
 else:model,opt=backbone.build(job['model'],device);criterion=None
 initial_lr=opt.param_groups[0]['lr'];detector=None
 base=Path(job['job_id']).parents[1]
 if stage=='segmentor':
  detector,_,_=specialized.build('detector',device)
  dpath=out/(smoke_prefix if is_smoke else Path('jobs'))/base/job['phase']/'detector'
  d=read(dpath/'complete.json');assert sha(dpath/d['checkpoint'])==d['checkpoint_sha256']
  load_weights(detector,'detector',torch.load(dpath/d['checkpoint'],map_location=device,weights_only=False)['model']);detector.eval()
 budget=p['training']['max_epochs'];target=None
 if job['phase']=='refit':
  d=read(out/(smoke_prefix if is_smoke else Path('jobs'))/base/'selection'/stage/'complete.json');target=d['selected_epoch'];budget=target
 if is_smoke:budget=1
 centers=initial_centers(rows,p);sampler=PatchSampler(rows,centers,cache_size=8)
 train_ids=set(ids);assert not train_ids & set(fold['outer_val_ids']);assert not train_ids & set(splits['test_ids'])
 if job['phase']=='selection':assert not train_ids & set(fold['inner_val_ids'])
 meta=dict(signature=signature,job=job,data_signature=p['data_signature'],sources=sources,train_ids=ids,selection_ids=[r['image_id'] for r in vals] if job['phase']=='selection' else [],target_epoch=target)
 if (folder/'job.json').exists():assert read(folder/'job.json')==meta
 else:write(folder/'job.json',meta)
 start=1;best=None;best_significant=-float('inf');last_improvement=0
 last=folder/'last.pt'
 if last.exists():
  state=torch.load(last,map_location=device,weights_only=False);assert state['signature']==signature
  load_weights(model,stage,state['model']);opt.load_state_dict(state['optimizer']);start=state['epoch']+1
  best=state['best'];best_significant=state['best_significant'];last_improvement=state['last_improvement']
  sampler.centers=state['centers'];random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'].cpu())
  if device.type=='cuda':torch.cuda.set_rng_state(state['cuda_rng'].cpu(),device)
  write(folder/'epochs'/f'{state["epoch"]:03d}.json',state['epoch_report'])
  if state.get('stop'):budget=state['epoch']
 def heartbeat(info):write(folder/'status.json',dict(state='running',pid=os.getpid(),job_id=job['job_id'],time=time.time(),**info))
 heartbeat(dict(phase='starting',epoch=start))
 for epoch in range(start,budget+1):
  disk_guard(p,out);tic=time.monotonic()
  for group in opt.param_groups:group['lr']=lr_for_epoch(initial_lr,epoch,p)
  rng=np.random.default_rng(seed+epoch*1009)
  draws=32 if is_smoke else p['training']['patches_per_epoch']
  examples=sampler.epoch(rng,specialist=special,draws=draws)
  def progress(info):
   if info.get('updates',0)%16==0:heartbeat(dict(phase='training',epoch=epoch,**info))
  if special:stats=specialized.train_one_epoch('deepmica' if job['model']=='deepmica' else stage,model,opt,criterion,examples,device,on_update=progress)
  else:stats=backbone.train_one_epoch(model,opt,examples,job['variant'],epoch,device,on_update=progress)
  stats.update(sampler.exposure())
  report=dict(epoch=epoch,train=stats,lr=opt.param_groups[0]['lr'],training_seconds=time.monotonic()-tic,validation=None)
  if (job['phase']=='selection' and epoch%p['training']['val_every']==0) or is_smoke:
   val=validate(job,model,vals,device,detector,heartbeat);report['validation']=val
   cand=dict(epoch=epoch,metric=val['metric'],negative_fp_per_image=val['negative_fp_per_image'])
   if better(cand,best):
    name=f'best_epoch_{epoch:03d}.pt';torch_save(folder/name,dict(signature=signature,epoch=epoch,model=state_dict(model,stage)))
    best=dict(cand,checkpoint=name,checkpoint_sha256=sha(folder/name))
   if val['metric']>best_significant+p['training']['min_delta']:best_significant=val['metric'];last_improvement=epoch
  stop=job['phase']=='selection' and epoch%p['training']['val_every']==0 and should_stop(epoch,last_improvement,p)
  if epoch==16:torch_save(folder/'epoch16.pt',dict(signature=signature,epoch=epoch,model=state_dict(model,stage)))
  if not special and epoch in p['training']['mining_epochs'] and epoch<budget and not stop and not is_smoke:
   sampler.set_hard_centers(mine(model,rows,device,heartbeat))
  state=dict(signature=signature,epoch=epoch,model=state_dict(model,stage),optimizer=opt.state_dict(),best=best,best_significant=best_significant,last_improvement=last_improvement,centers=sampler.centers,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None,epoch_report=report,stop=stop)
  torch_save(last,state);write(folder/'epochs'/f'{epoch:03d}.json',report)
  print(json.dumps(dict(job=job['job_id'],epoch=epoch,train_loss=stats['training_loss'],best=best,stop=stop)),flush=True)
  if stop:break
 state=torch.load(last,map_location='cpu',weights_only=False)
 if job['phase']=='selection' or is_smoke:
  assert best is not None;chosen=best['checkpoint'];selected=best['epoch']
 else:
  chosen='final.pt';selected=state['epoch'];torch_save(folder/chosen,dict(signature=signature,epoch=selected,model=state['model']))
 complete=dict(complete=True,smoke=is_smoke,extended=state['epoch']>p['training']['base_epochs'],last_improvement_epoch=last_improvement,signature=signature,job=job,epochs=state['epoch'],selected_epoch=selected,checkpoint=chosen,checkpoint_sha256=sha(folder/chosen),convergence_status='smoke_only' if is_smoke else ('fixed_refit' if job['phase']=='refit' else ('plateau_stop' if state['stop'] else 'cap_hit_unresolved')),best=best)
 write(folder/'complete.json',complete);write(folder/'status.json',dict(state='complete',time=time.time(),**complete))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--job-index',type=int,required=True);a.add_argument('--device',default='cuda:0');a.add_argument('--smoke',action='store_true');args=a.parse_args()
 try:run(args)
 except Exception as e:
  import traceback
  traceback.print_exc();sys.exit(1)

"""Parent barrier: exact complete epoch3 state for all four losses per fold."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
from common import ROOT,OUT,VARIANTS,context,read,verify,checksum,immutable,job_signature

def tensor_digest(value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            y=x.detach().cpu().contiguous();h.update(str((str(y.dtype),tuple(y.shape))).encode());h.update(y.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        else:h.update(json.dumps(x,sort_keys=True).encode())
    visit(value);return h.hexdigest()

def main():
    c=context();records=[];outputs={};torch.set_num_threads(2)
    for f in c['folds']:
        fi=f['fold'];expected=None
        for variant in VARIANTS:
            active=dict(c,variant=variant);job=ROOT/'jobs'/variant/f'fold_{fi}';sig=job_signature(active,fi);m=verify(job/'resume.json',sig)
            assert m['epoch']==3
            st=torch.load(job/'resume.pt',map_location='cpu',weights_only=False);assert st['epoch']==3 and st['signature']==sig
            assert st['job']['train_ids']==f['train_ids'] and st['job']['validation_ids']==f['validation_ids']
            assert {int(v['step']) for v in st['optimizer']['state'].values()}=={384}
            stage=verify(job/'mining/after_03/complete.json',sig);assert stage['image_ids']==f['train_ids'] and stage['training_only'] and not stage['smoke']
            with np.load(stage['pool_path']) as z:pool={k:torch.from_numpy(z[k].copy()) for k in z.files}
            epochs=[read(job/'epochs'/f'{e:02d}.json') for e in (1,2,3)]
            item=dict(model_sha256=tensor_digest(st['model']),optimizer_sha256=tensor_digest(st['optimizer']),rng_cpu_sha256=tensor_digest(st['rng_cpu']),rng_cuda_sha256=tensor_digest(st['rng_cuda']),pool_sha256=tensor_digest(pool),input_geometry=[(e['input_target_sha256'],e['geometry_sha256']) for e in epochs])
            if expected is None:expected=item
            else:assert expected==item,(fi,variant,'warmup state differs')
            target=job/'warmup_checkpoint.pt'
            if target.exists():assert checksum(target)==checksum(job/'resume.pt')
            else:shutil.copyfile(job/'resume.pt',target)
            marker=job/'warmup_complete.json';immutable(marker,dict(complete=True,signature=sig,variant=variant,fold=fi,epoch=3,evidence=item,output_files={str(target):checksum(target),str(job/'mining/after_03/complete.json'):checksum(job/'mining/after_03/complete.json')}))
            outputs[str(marker)]=checksum(marker);outputs[str(target)]=checksum(target)
            records.append(dict(variant=variant,fold=fi,**item))
    result=dict(complete=True,signature=c['signature'],jobs=12,all_model_optimizer_pool_equal=True,all_warmup_input_geometry_equal=True,records=records,verifier_sha256=checksum(__file__),output_files=outputs)
    immutable(ROOT/'warmup_verified.json',result);immutable(OUT/'warmup_verification.json',result);print('INDEPENDENT all12 warmup exact PASS',flush=True)
if __name__=='__main__':main()

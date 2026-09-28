"""Execute unchanged upstream class ASTs independently of the port."""
import ast,math,hashlib
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from common import context,OUT,checksum
from model import UNeXtOfficial3ch,UNeXtAuthor2ch
from calcification_gate_common import atomic_json

torch.set_num_threads(2)
p=Path('scripts/improvement_unext_deterministic/vendor_unext/archs.py');source=p.read_text()
assert checksum(p)=='7486a70d6398484da496727f49f7b0d0c14aac0b49bda0cd34b4f877737831ba'
tree=ast.parse(source);tree.body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
namespace=dict(torch=torch,nn=nn,F=F,math=math,to_2tuple=lambda x:(x,x) if isinstance(x,int) else x,trunc_normal_=nn.init.trunc_normal_)
exec(compile(tree,str(p),'exec'),namespace)
torch.manual_seed(42);original=namespace['UNext'](num_classes=1,img_size=256)
torch.manual_seed(42);ported=UNeXtOfficial3ch()
assert list(original.state_dict())==list(ported.state_dict())
for k,v in original.state_dict().items():assert torch.equal(v,ported.state_dict()[k]),k
checks=[]
for training in (False,True):
    original.train(training);ported.train(training)
    x=torch.randn(2,3,256,256,requires_grad=True);xx=x.detach().clone().requires_grad_();a=original(x);b=ported(xx)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    ga=torch.autograd.grad(a.square().mean(),(x,original.encoder2.weight));gb=torch.autograd.grad(b.square().mean(),(xx,ported.encoder2.weight))
    for u,v in zip(ga,gb):torch.testing.assert_close(u,v,rtol=0,atol=0)
    for k,v in original.state_dict().items():assert torch.equal(v,ported.state_dict()[k]),k
    checks.append(dict(training=training,output_exact=True,input_and_encoder2_gradient_exact=True,BN_buffers_exact=True))
c=context();m=UNeXtAuthor2ch(c['learning_config'])
atomic_json(OUT/'model_source_verification.json',dict(complete=True,signature=c['signature'],upstream_sha256=checksum(p),verifier_sha256=checksum(__file__),upstream_class_AST_unchanged=True,only_imports_excluded=True,helper_compatibility='torch.nn.init.trunc_normal_ and scalar to_2tuple; defaultzeroDropPath isIdentity',three_channel_initial_tensors_exact=True,parameters=sum(p.numel() for p in m.parameters()),checks=checks))
print('INDEPENDENT upstream model PASS',flush=True)

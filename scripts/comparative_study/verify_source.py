"""Parent binding of direct-source parity and transparent original PatchUNet reuse."""
import ast,json
from pathlib import Path
import torch
from common import context,OUT,checksum
from model import StudyModel
from calcification_refinement_common import PatchUNet
from calcification_gate_common import atomic_json
c=context();cfg=c['learning_config'];torch.manual_seed(42);m=StudyModel(cfg);b=cfg['backbone'];p=Path('reports/comparative_study/model_source_verification.json');r=json.loads(p.read_text())
for vendor in ['vendor_attention','vendor_utnet']:
 folder=Path('scripts/comparative_study/models')/vendor
 for f in (folder/'original').glob('*.py'):
  def tree(path):
   a=ast.parse(path.read_text());a.body=[n for n in a.body if not isinstance(n,(ast.Import,ast.ImportFrom))];return ast.dump(a,include_attributes=False)
  assert tree(f)==tree(folder/'compat'/f.name),str(f)
if b=='proposed':
 assert type(m) is PatchUNet
 torch.manual_seed(42);other=PatchUNet(cfg)
 assert all(torch.equal(v,other.state_dict()[k]) for k,v in m.state_dict().items())
else:
 assert 'CPU upstream parity passed' in Path('logs/comparative_study/model_source.log').read_text()
assert sum(v.numel() for v in m.parameters())=={'proposed':487857,'unet2d':2465185,'utnet':10018725}[b]
atomic_json(OUT/'model_source_verification.json',dict(complete=True,signature=c['signature'],backbone=b,parameters=sum(v.numel() for v in m.parameters()),vendor_non_import_ast_exact=True,original_PatchUNet_direct_reuse=b=='proposed',source_parity_report_sha256=checksum(p),parent_parity_log_sha256=checksum('logs/comparative_study/model_source.log'),verifier_sha256=checksum(__file__),performance_evaluation=False))
print('parent source binding PASS',b)

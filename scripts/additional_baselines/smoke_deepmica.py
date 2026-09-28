"""Synthetic CPU execution check, not a dataset performance evaluation."""
import json
from pathlib import Path
import torch
from deepmica import build_model, source_sha256

torch.set_num_threads(4)
records=[]
for channels in (1, 2):
    torch.manual_seed(42)
    model=build_model(channels)
    x=torch.randn(1, channels, 256, 256)
    y=model(x)
    assert y.shape == (1, 1, 256, 256) and torch.isfinite(y).all()
    loss=torch.nn.functional.binary_cross_entropy_with_logits(y, torch.zeros_like(y))
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    records.append({'in_channels':channels, 'output_shape':list(y.shape),
                    'parameters':sum(p.numel() for p in model.parameters()),
                    'finite_output_and_gradients':True})
    del model,x,y,loss,grads
result={'scope':'synthetic CPU forward/backward only; no training or clinical metrics',
        'torch':torch.__version__, 'source_sha256':source_sha256(), 'checks':records}
p=Path(__file__).resolve().parents[2]/'reports/additional_baselines/deepmica_smoke.json'
p.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

"""Transparent factory; no wrapper prefix or architecture operation changes."""
from calcification_refinement_common import PatchUNet

def StudyModel(config):
    if config['backbone']=='proposed':return PatchUNet(config)
    from models import build_model
    return build_model(dict(config,**config['architecture_config']))

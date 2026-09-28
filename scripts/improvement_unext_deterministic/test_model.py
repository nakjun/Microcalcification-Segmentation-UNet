"""CPU-only provenance and interface checks for UNeXtAuthor2ch."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import torch


SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from improvement_unext_deterministic.model import ADAPTATION_MODULES, UNeXtAuthor2ch, UNeXtOfficial3ch, parameter_count


CONFIG = {'input_channels': 2, 'patch_size': 256}


def _align_for_two_channel_semantics(reference, adapted):
    """Copy every common state tensor and map the reference third input to zero."""
    reference_state, adapted_state = reference.state_dict(), adapted.state_dict()
    for name in adapted_state:
        if name in {'encoder1.weight', 'final.weight', 'final.bias'}:
            continue
        adapted_state[name].copy_(reference_state[name])
    with torch.no_grad():
        adapted.encoder1.weight.copy_(reference.encoder1.weight[:, :2])
        adapted.encoder1.bias.copy_(reference.encoder1.bias)
        adapted.final.weight.copy_(reference.final.weight)
        adapted.final.bias.copy_(reference.final.bias)
    adapted.load_state_dict(adapted_state, strict=True)
    for name, tensor in adapted.state_dict().items():
        if name not in {'encoder1.weight', 'final.weight', 'final.bias'}:
            torch.testing.assert_close(reference_state[name], tensor, rtol=0, atol=0)


class UNeXtAuthorTests(unittest.TestCase):
    def test_adaptation_contract_and_cpu_shape_reload_gradient(self):
        torch.manual_seed(42)
        model = UNeXtAuthor2ch(CONFIG)
        self.assertEqual(model.adaptation_modules, ADAPTATION_MODULES)
        self.assertGreater(parameter_count(model), 0)
        output = model(torch.randn(1, 2, 256, 256))
        self.assertEqual(tuple(output.shape), (1, 1, 256, 256))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
        restored = UNeXtAuthor2ch(CONFIG)
        restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        self.assertEqual(parameter_count(restored), parameter_count(model))

    def test_three_channel_zero_plane_semantics_and_selected_gradient(self):
        torch.manual_seed(42)
        reference = UNeXtOfficial3ch(num_classes=1)
        torch.manual_seed(42)
        adapted = UNeXtAuthor2ch(CONFIG)
        _align_for_two_channel_semantics(reference, adapted)
        reference.eval(); adapted.eval()
        x2 = torch.randn(1, 2, 256, 256, requires_grad=True)
        x3_base = x2.detach().clone().requires_grad_()
        x3 = torch.cat((x3_base, torch.zeros(1, 1, 256, 256)), dim=1)
        out2, out3 = adapted(x2), reference(x3)
        torch.testing.assert_close(out2, out3, rtol=0, atol=0)
        out2.square().mean().backward(); out3.square().mean().backward()
        torch.testing.assert_close(x2.grad, x3_base.grad, rtol=0, atol=0)
        torch.testing.assert_close(adapted.encoder2.weight.grad, reference.encoder2.weight.grad, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()

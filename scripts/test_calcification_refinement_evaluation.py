"""Focused output/provenance test; synthetic fixtures, no study data."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from calcification_gate_common import atomic_json, checksum
from calcification_gate_sam_common import load_masks, save_masks
import evaluate_calcification_refinement as evaluation


class ConstantLogit(torch.nn.Module):
    def forward(self, value):
        return torch.zeros_like(value[:, :1])


class LockedEvaluationTests(unittest.TestCase):
    def test_native_box_score_schema_and_GT_order_and_resume(self):
        config = json.loads(Path('configs/calcification_refinement.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config.update(root=str(root), frozen_pipeline_root=str(root / 'baseline'), disk_reserve_gib=0)
            image = np.full((17, 23), 128, np.uint8)
            source = root / 'tiny.png'
            cv2.imwrite(str(source), image)
            gt = np.zeros_like(image)
            gt[8, 11] = 255
            cv2.imwrite(str(root / 'tiny_calcifications.png'), gt)
            row = {'image_id': 'tiny', 'study_id': 'reserved_group', 'split': 'reserved',
                   'source_path': str(source), 'source_sha256': checksum(source), 'height': 17, 'width': 23}
            boxes = np.asarray([[0, 0, 23, 17, .618]], np.float32)
            for arm in config['arms']:
                output = root / 'baseline/csaw' / arm / 'gated'
                path = output / 'masks/tiny.npz'
                sha = save_masks(path, image > 0, image > 0, boxes)
                atomic_json(output / 'cases/tiny.json', {'mask_path': str(path), 'mask_sha256': sha,
                    'gate_pass': True, 'gate_logit': 1.0, 'prediction_boxes': boxes.tolist()})
            selection = {'locked_before_reserved_GT': True, 'development_groups': ['train_group'],
                         'constraint': 'predicted_boxes', 'threshold': .5, 'method': 'predicted_boxes@0.5'}
            output = root / 'final'
            original = evaluation.read_target
            def guarded_read(row, reserved=False):
                self.assertTrue((output / 'probabilities/tiny.npz').exists())
                for arm in config['arms']:
                    mask, _, saved_boxes = load_masks(output / arm / 'masks/tiny.npz')
                    np.testing.assert_array_equal(saved_boxes, boxes)
                    self.assertTrue(mask.all())
                return original(row, reserved)
            with patch.object(evaluation, 'read_target', side_effect=guarded_read):
                result = evaluation.evaluate(ConstantLogit(), [row], config, torch.device('cpu'),
                    output, 'synthetic_signature', 1, lambda *a, **kw: None, selection=selection, reserved=True)
            self.assertTrue(result['complete'])
            for arm in config['arms']:
                self.assertEqual(set(result['metrics'][arm]), {'SAM_baseline', 'box_fill_control', 'predicted_boxes@0.5'})
                self.assertEqual(result['metrics'][arm]['predicted_boxes@0.5']['TP'], 1)
            resumed = evaluation.evaluate(ConstantLogit(), [row], config, torch.device('cpu'),
                output, 'synthetic_signature', 1, lambda *a, **kw: None, selection=selection, reserved=True)
            self.assertEqual(result, resumed)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()

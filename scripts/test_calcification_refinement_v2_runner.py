"""Integration checks for OOF pooling, role isolation and resumable training."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from calcification_refinement_common import PatchUNet
from calcification_refinement_coverage_common import measure_probability
from train_calcification_refinement_v2 import (aggregate, component_stats, epoch_train,
    fresh_model, state_digest, validate_folds, evaluate)


class RunnerTests(unittest.TestCase):
    def test_real_fold_partition_and_roles(self):
        rows = json.loads(Path('runs/calcification_refinement_v1/data/manifest.json').read_text())
        folds = json.loads(Path('configs/calcification_refinement_v2_folds.json').read_text())['folds']
        self.assertEqual(len(validate_folds(rows, folds)), 124)
        bad = copy.deepcopy(folds)
        bad[0]['validation_ids'][0] = next(r['image_id'] for r in rows if r['role'] != 'train')
        with self.assertRaises(AssertionError):
            validate_folds(rows, bad)

    def test_pool_counts_not_mean_fold_metrics(self):
        inference = {'threshold_grid': [.3], 'inside_thresholds': [], 'outside_thresholds': []}
        records = []
        for score, target in [(.9, True), (.1, True), (.9, False)]:
            probability = np.array([[score]], np.float32)
            fill = np.ones((1, 1), bool)
            y = x = np.array([0] if target else [], dtype=np.int64)
            value = measure_probability(probability, fill, fill, y, x, True, inference)
            for metric in value.values():
                metric['components'] = component_stats(np.array([metric['TP'] > 0]), [np.array([0])] if target else [])
            records.append(value)
        total = aggregate(records, 'boxes@0.3')
        self.assertEqual((total['TP'], total['FP'], total['FN']), (1, 1, 1))
        self.assertEqual(total['positive_macro_Dice'], .5)
        self.assertEqual(total['negative_images_with_FP'], 1)
        self.assertEqual(total['components']['1-4']['total'], 2)

    def test_checkpoint_optimizer_resume_matches_continuous(self):
        c = json.loads(Path('configs/calcification_refinement.json').read_text())
        c.update(training_patches_per_epoch=2, training_batch_size=2, overlap_loss='tversky')
        class Sampler:
            def sample(self, rng):
                x = rng.normal(size=(2, 32, 32)).astype(np.float32)
                y = np.zeros((1, 32, 32), np.float32)
                y[0, 16, 16] = 1
                return x, y, 'positive'
        device = torch.device('cpu')
        model, optimizer = fresh_model(c, device)
        epoch_train(model, optimizer, Sampler(), c, 1, device, lambda *a, **k: None, {})
        saved = copy.deepcopy({'model': model.state_dict(), 'optimizer': optimizer.state_dict()})
        epoch_train(model, optimizer, Sampler(), c, 2, device, lambda *a, **k: None, {})
        restored, opt = fresh_model(c, device)
        restored.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        epoch_train(restored, opt, Sampler(), c, 2, device, lambda *a, **k: None, {})
        self.assertEqual(state_digest(model), state_digest(restored))

    def test_evaluation_gate_rejection_retains_false_negative(self):
        c = json.loads(Path('configs/calcification_refinement.json').read_text())
        inference = {'threshold_grid': [.3], 'inside_thresholds': [], 'outside_thresholds': []}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            np.save(root / 'image.npy', np.zeros((16, 16), np.uint8))
            np.savez(root / 'gt.npz', y=np.array([5]), x=np.array([5]))
            rows = [{'image_id': 'test', 'study_id': 'test', 'role': 'validation',
                'native_cache_path': str(root / 'image.npy'), 'target_path': str(root / 'gt.npz'),
                'source_sha256': 'test', 'target_sha256': 'test'}]
            empty = np.zeros((16, 16), bool)
            with patch('train_calcification_refinement_v2.tiled_probability', return_value=(np.ones((16, 16), np.float32), 1)), \
                 patch('train_calcification_refinement_v2.baseline_masks', return_value=(empty, empty, {'gate_pass': False})):
                result = evaluate(None, rows, c, inference, torch.device('cpu'), root / 'out',
                    'smoke_signature', lambda *a, **k: None, {})
            for arm in c['arms']:
                m = result['metrics'][arm]['full_image@0.3']
                self.assertEqual((m['TP'], m['FN'], m['FP']), (0, 1, 0))
                self.assertEqual(m['positive_images'], 1)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()

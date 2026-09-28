import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from calcification_refinement_common import (PatchUNet, TrainSampler, candidate_metrics,
    input_patch, reflected_indices, select_candidate, tiled_probability, training_loss)

C = json.loads(Path('configs/calcification_refinement.json').read_text())


class RefinementTests(unittest.TestCase):
    def test_global_contrast_patch_equivalence(self):
        image = np.random.default_rng(42).integers(0, 256, (531, 417), dtype=np.uint8)
        raw = image.astype(np.float32) / 255
        smooth = cv2.GaussianBlur(raw, (19, 19), 3, borderType=cv2.BORDER_REFLECT_101)
        full = np.stack((2 * raw - 1, np.clip((raw - smooth) * 16, -1, 1)))
        for y, x in [(0, 0), (75, 37), (275, 161)]:
            np.testing.assert_allclose(input_patch(image, y, x, C), full[:, y:y + 256, x:x + 256], atol=1e-6)
        self.assertEqual(reflected_indices(-2, 7, 3).tolist(), [2, 1, 0, 1, 2, 1, 0])

    def test_full_tiling_and_probability_mean(self):
        class IdentityLogit(torch.nn.Module):
            def forward(self, x):
                return x[:, :1]
        image = np.random.default_rng(1).integers(0, 256, (501, 379), dtype=np.uint8)
        result, count = tiled_probability(IdentityLogit(), image, C, torch.device('cpu'))
        expected = 1 / (1 + np.exp(-(image.astype(np.float32) / 255 * 2 - 1)))
        np.testing.assert_allclose(result, expected, atol=1e-7)
        self.assertEqual(count, 6)
        small, _ = tiled_probability(IdentityLogit(), image[:73, :61], C, torch.device('cpu'))
        np.testing.assert_allclose(small, expected[:73, :61], atol=1e-7)

    def test_sparse_metrics_exact_threshold_and_empty_gate(self):
        p = np.array([[.1, .2], [.3, .5]], np.float32)
        sam = np.array([[True, False], [False, True]])
        fill = np.ones((2, 2), bool)
        result = candidate_metrics(p, sam, fill, np.array([0, 1]), np.array([0, 1]), C)
        self.assertEqual(result['methods']['predicted_boxes@0.5']['TP'], 1)
        self.assertEqual(result['methods']['SAM_intersection@0.1']['TP'], 2)
        empty = candidate_metrics(p, fill & False, fill & False, np.array([0]), np.array([0]), C)
        self.assertEqual(empty['methods']['predicted_boxes@0.1']['FN'], 1)
        self.assertEqual(empty['methods']['predicted_boxes@0.1']['FP'], 0)

    def test_negative_loss_and_train_only_sampler(self):
        logits = torch.zeros((2, 1, 16, 16), requires_grad=True)
        loss, info = training_loss(logits, torch.zeros_like(logits), C)
        loss.backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        self.assertEqual(info['positive_soft_Dice_loss'], 0)
        with self.assertRaises(AssertionError):
            TrainSampler([{'role': 'validation'}], C, {})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            np.save(path / 'image.npy', np.full((300, 300), 100, np.uint8))
            np.savez(path / 'target.npz', y=np.array([3]), x=np.array([3]))
            row = {'role': 'train', 'image_id': 'example', 'positive': True,
                   'native_cache_path': str(path / 'image.npy'), 'target_path': str(path / 'target.npz')}
            c = dict(C, sampling_probabilities={'positive': 0, 'hard_negative': 1, 'random_breast': 0})
            sampler = TrainSampler([row], c, {'example_hard': np.array([[170, 170]]),
                                             'example_breast': np.array([[170, 170]])})
            _, target, mode = sampler.sample(np.random.default_rng(42))
            self.assertEqual(mode, 'hard_negative')
            self.assertFalse(target.any())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()

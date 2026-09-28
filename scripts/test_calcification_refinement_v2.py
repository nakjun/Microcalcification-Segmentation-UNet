"""Behavioral checks for proposed recall-aware training and selection."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from calcification_refinement_common import TrainSampler, training_loss as original_loss
from calcification_refinement_v2_common import (RecallAwareSampler, choose_positive_pixel,
    choose_recall_constrained, component_pixel_groups, mine_negative_centers, positive_overlap_loss, training_loss)


C = json.loads(Path('configs/calcification_refinement.json').read_text())


class RecallTrainingTests(unittest.TestCase):
    def test_dice_control_and_negative_gradient(self):
        logits = torch.tensor([[[[0., 2.], [-1., 3.]]], [[[1., -1.], [0., 0.]]]], requires_grad=True)
        target = torch.tensor([[[[1., 0.], [0., 1.]]], [[[0., 0.], [0., 0.]]]])
        config = dict(C, overlap_loss='dice')
        self.assertTrue(torch.equal(training_loss(logits, target, config)[0], original_loss(logits, target, C)[0]))
        symmetric = positive_overlap_loss(logits, target, 'tversky', .5, .5)
        dice = positive_overlap_loss(logits, target, 'dice')
        torch.testing.assert_close(symmetric, dice)
        config['overlap_loss'] = 'tversky'
        loss, info = training_loss(logits, torch.zeros_like(target), config)
        loss.backward()
        self.assertEqual(info['positive_overlap_loss'], 0.)
        self.assertTrue(torch.all(logits.grad > 0))  # Gradient descent lowers all-negative logits.

    def test_false_negative_gradient_increases_with_FN_weight(self):
        targets = torch.tensor([[[[1., 0.]]]])
        gradients = []
        for fp, fn in [(.5, .5), (.3, .7)]:
            logits = torch.zeros_like(targets, requires_grad=True)
            positive_overlap_loss(logits, targets, 'tversky', fp, fn).backward()
            gradients.append(logits.grad.detach())
        self.assertGreater(abs(float(gradients[1][0, 0, 0, 0])), abs(float(gradients[0][0, 0, 0, 0])))
        self.assertLess(abs(float(gradients[1][0, 0, 0, 1])), abs(float(gradients[0][0, 0, 0, 1])))

    def test_small_component_gets_equal_draws_without_pixel_dilation(self):
        gy = np.array([1] + [4] * 8)
        gx = np.array([1] + list(range(2, 10)))
        groups = component_pixel_groups(gy, gx, (12, 12))
        self.assertEqual([len(g) for g in groups], [1, 8])
        for strategy, expected in [('pixel_uniform', 1 / 9), ('component_uniform', .5)]:
            rng = np.random.default_rng(42)
            rate = np.mean([choose_positive_pixel(rng, groups, 9, strategy) == 0 for _ in range(10000)])
            self.assertLess(abs(rate - expected), .02)
        self.assertEqual(sum(map(len, groups)), 9)

    def test_sampler_control_exact_and_train_only(self):
        with self.assertRaises(AssertionError):
            RecallAwareSampler([{'role': 'validation'}], dict(C, positive_strategy='component_uniform'), {})
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            np.save(p / 'image.npy', np.full((300, 300), 128, np.uint8))
            np.savez(p / 'gt.npz', y=np.array([150]), x=np.array([150]))
            rows = [{'role': 'train', 'positive': True, 'image_id': 'tiny',
                     'native_cache_path': str(p / 'image.npy'), 'target_path': str(p / 'gt.npz')}]
            config = dict(C, positive_strategy='pixel_uniform',
                          sampling_probabilities={'positive': 1, 'hard_negative': 0, 'random_breast': 0})
            old, new = TrainSampler(rows, config, {}), RecallAwareSampler(rows, config, {})
            a, b = np.random.default_rng(42), np.random.default_rng(42)
            for _ in range(5):
                left, right = old.sample(a), new.sample(b)
                np.testing.assert_array_equal(left[0], right[0])
                np.testing.assert_array_equal(left[1], right[1])
                self.assertEqual(left[2], right[2])

    def test_selection_rejects_high_Dice_with_missed_pixels_or_images(self):
        ref = {'TP': 60, 'FN': 40, 'positive_images': 10, 'negative_images': 10,
               'positive_images_with_TP': 8, 'negative_FP_pixels': 1000, 'negative_images_with_FP': 5}
        references = {arm: copy.deepcopy(ref) for arm in ['imagenet', 'mammo']}
        good = {'candidate_id': 'good', 'metrics': {arm: dict(ref, positive_macro_Dice=.3)
                                                  for arm in references}}
        high_dice = copy.deepcopy(good)
        high_dice['candidate_id'] = 'fails_recall'
        high_dice['metrics']['mammo'].update(TP=59, FN=41, positive_macro_Dice=.9)
        self.assertEqual(choose_recall_constrained([good, high_dice], references)['candidate_id'], 'good')
        self.assertIsNone(choose_recall_constrained([high_dice], references))
        missed_images = copy.deepcopy(good)
        missed_images['metrics']['imagenet']['positive_images_with_TP'] = 7
        self.assertIsNone(choose_recall_constrained([missed_images], references))
        extra_alarms = copy.deepcopy(good)
        extra_alarms['metrics']['imagenet']['negative_images_with_FP'] = 6
        self.assertIsNone(choose_recall_constrained([extra_alarms], references))

    def test_model_false_positive_mining_checks_whole_patch_and_training_role(self):
        probability = np.zeros((512, 512), np.float32)
        probability[150, 150] = .99  # Center misses GT, but the 256 patch contains it.
        probability[450, 450] = .9   # Entire patch is GT empty.
        gy, gx = np.array([160]), np.array([160])
        centers, scores = mine_negative_centers({'role': 'train'}, probability, gy, gx)
        np.testing.assert_array_equal(centers, [[450, 450]])
        np.testing.assert_array_equal(scores, np.array([.9], np.float32))
        with self.assertRaises(AssertionError):
            mine_negative_centers({'role': 'validation'}, probability, gy, gx)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()

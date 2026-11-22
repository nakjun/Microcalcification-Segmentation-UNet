import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("csaw312_metrics", Path(__file__).with_name("metrics.py"))
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class SegmentationTests(unittest.TestCase):
    def test_positive_macro_and_pooled_recall_have_different_denominators(self):
        first = metrics.segmentation_counts(np.array([[0.5, 0.1], [0.8, 0.1]]), np.array([[1, 1], [0, 0]]))
        second = metrics.segmentation_counts(np.ones((2, 2)), np.ones((2, 2)))
        negative = metrics.segmentation_counts(np.array([[1, 1], [0, 0]]), np.zeros((2, 2)))
        clean = metrics.segmentation_counts(np.zeros((2, 2)), np.zeros((2, 2)))
        self.assertEqual([first[key] for key in ("TP", "FP", "FN", "TN")], [1, 1, 1, 1])
        self.assertEqual(first["dice"], 0.5)
        result = metrics.summarize_segmentation([first, second, negative, clean])
        self.assertEqual(result["positive_macro_dice"], 0.75)
        self.assertAlmostEqual(result["pixel_recall"], 5 / 6)
        self.assertEqual(result["negative_fp_per_image"], 1)
        self.assertEqual(result["negative_fp_images"], 1)

    def test_empty_gt_and_no_positive_predictions_are_explicit(self):
        negative = metrics.segmentation_counts(np.zeros((1, 1)), np.zeros((1, 1)))
        self.assertIsNone(negative["dice"])
        summary = metrics.summarize_segmentation([negative])
        self.assertIsNone(summary["positive_macro_dice"])
        self.assertIsNone(summary["pixel_recall"])
        missed = metrics.segmentation_counts(np.zeros((1, 1)), np.ones((1, 1)))
        self.assertEqual(missed["dice"], 0)

    def test_nonfinite_and_invalid_probabilities_rejected(self):
        for value in (float("nan"), -0.1, 1.1):
            with self.assertRaises(ValueError):
                metrics.segmentation_counts(np.array([[value]]), np.ones((1, 1)))


class DetectionTests(unittest.TestCase):
    def record(self, ident="a", boxes=None, scores=None, truths=None):
        return dict(image_id=ident, boxes=[] if boxes is None else boxes,
                    scores=[] if scores is None else scores, gt_boxes=[] if truths is None else truths)

    def test_perfect_multiple_images_and_halfopen_singlepixel_boxes(self):
        record = self.record(boxes=[[0, 0, 1, 1]], scores=[0.9], truths=[[0, 0, 1, 1]])
        other = self.record("b", [[5, 5, 6, 6]], [0.4], [[5, 5, 6, 6]])
        result = metrics.detection_ap([record, other])
        self.assertEqual(result["AP50_95"], 1)
        self.assertEqual(result["per_iou"]["0.95"]["TP"], 2)

    def test_empty_predictions_and_empty_gt_cohort(self):
        missing = metrics.detection_ap([self.record(truths=[[0, 0, 1, 1]])])
        self.assertEqual(missing["AP50_95"], 0)
        unmeasured = metrics.detection_ap([self.record(boxes=[[0, 0, 1, 1]], scores=[0.9])])
        self.assertIsNone(unmeasured["AP50_95"])
        self.assertEqual(unmeasured["per_iou"]["0.50"]["FP"], 1)

    def test_duplicate_fp_before_second_true_positive(self):
        record = self.record(boxes=[[0, 0, 1, 1], [0, 0, 1, 1], [3, 3, 4, 4]], scores=[0.9, 0.8, 0.7],
                             truths=[[0, 0, 1, 1], [3, 3, 4, 4]])
        result = metrics.detection_ap([record])
        self.assertAlmostEqual(result["AP50"], (51 + 50 * 2 / 3) / 101)
        self.assertEqual(result["per_iou"]["0.50"], dict(AP=result["AP50"], TP=2, FP=1, FN=0))

    def test_false_positive_on_negative_image_enters_pooled_precision(self):
        positive = self.record("positive", [[0, 0, 1, 1]], [0.6], [[0, 0, 1, 1]])
        negative = self.record("negative", [[0, 0, 1, 1]], [0.9])
        result = metrics.detection_ap([positive, negative])
        self.assertEqual(result["AP50"], 0.5)

    def test_iou_thresholds_and_cross_image_matching(self):
        partial = self.record(boxes=[[0, 0, 6, 10]], scores=[0.8], truths=[[0, 0, 10, 10]])
        result = metrics.detection_ap([partial])
        self.assertAlmostEqual(result["AP50_95"], 0.3)
        self.assertEqual(result["AP_per_IoU"]["0.60"], 1)
        self.assertEqual(result["AP_per_IoU"]["0.65"], 0)
        wrong_image = metrics.detection_ap([self.record("a", [[0, 0, 1, 1]], [0.9]),
                                            self.record("b", truths=[[0, 0, 1, 1]])])
        self.assertEqual(wrong_image["AP50_95"], 0)

    def test_score_ties_are_deterministic_under_record_reordering(self):
        a = self.record("a", [[0, 0, 1, 1]], [0.5], [[0, 0, 1, 1]])
        b = self.record("b", [[0, 0, 1, 1]], [0.5])
        self.assertEqual(metrics.detection_ap([a, b]), metrics.detection_ap([b, a]))


if __name__ == "__main__":
    unittest.main()

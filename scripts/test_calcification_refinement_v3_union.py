import unittest
import numpy as np
from evaluate_calcification_refinement_v3_union import union_masks, counts


class UnionTests(unittest.TestCase):
    def test_exhaustive_recall_preservation(self):
        for a in range(16):
            old = np.array([bool(a & (1 << i)) for i in range(4)]).reshape(2, 2)
            for b in range(16):
                new = np.array([bool(b & (1 << i)) for i in range(4)]).reshape(2, 2)
                union = union_masks(old, new)
                self.assertTrue(np.all(union[old]))
                self.assertTrue(np.all(union[new]))
                self.assertTrue(np.array_equal(union, union_masks(new, old)))

    def test_union_adds_tp_and_fp_as_reported(self):
        old = np.array([[True, False], [False, False]])
        new = np.array([[False, True], [True, False]])
        union = union_masks(old, new)
        m = counts(union, np.array([0, 0]), np.array([0, 1]), old, [np.array([0]), np.array([1])])
        self.assertEqual((m['TP'], m['FP'], m['FN']), (2, 1, 0))
        self.assertEqual((m['TP_inside_boxes'], m['TP_outside_boxes']), (1, 1))


if __name__ == '__main__':
    unittest.main()

"""Synthetic fixed-policy checks; they do not load checkpoints or external data."""
import numpy as np

from infer import primary_mask, ungated_mask


def test_policy_boundaries_and_sparse_reconstruction():
    probability = np.asarray([[.299, .3, .499, .5, .899, .9]], np.float32)
    np.testing.assert_array_equal(primary_mask(probability, True), [[0, 0, 0, 0, 0, 1]])
    assert not primary_mask(probability, False).any()
    np.testing.assert_array_equal(ungated_mask(probability), [[0, 0, 0, 1, 1, 1]])
    flat = np.flatnonzero(probability.ravel() >= np.float32(.3))
    sparse = np.zeros_like(probability)
    sparse.ravel()[flat] = probability.ravel()[flat]
    np.testing.assert_array_equal(primary_mask(probability, True), primary_mask(sparse, True))
    np.testing.assert_array_equal(ungated_mask(probability), ungated_mask(sparse))


if __name__ == '__main__':
    test_policy_boundaries_and_sparse_reconstruction()
    print('inference synthetic test passed')

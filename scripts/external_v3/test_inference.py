"""CPU-only fixed policy/storage checks, no data-dependent model evaluation."""
import unittest
from infer import *
class FixedInferenceTests(unittest.TestCase):
 def test_boundaries_and_gate(self):
  p=np.array([[.299,.3,.899,.9]],np.float32);f=np.array([[1,1,0,0]],bool)
  np.testing.assert_array_equal(policy(p,f,True),[[0,1,0,1]])
  self.assertFalse(policy(p,f,False).any())
 def test_sparse_is_exact(self):
  rng=np.random.default_rng(42);p=rng.random((71,53),dtype=np.float32);f=rng.random(p.shape)>.5
  flat=np.flatnonzero(p.ravel()>=np.float32(.3));q=np.zeros_like(p);q.ravel()[flat]=p.ravel()[flat]
  np.testing.assert_array_equal(policy(p,f,True),policy(q,f,True))
 def test_box_raster_and_packing(self):
  boxes=np.array([[1.3,2.1,4.2,5.8,.8]],np.float32);f=box_fill_mask((9,7),boxes)
  expected=np.zeros((9,7),bool);expected[2:6,1:5]=1
  np.testing.assert_array_equal(f,expected)
  np.testing.assert_array_equal(np.unpackbits(np.packbits(f),count=f.size).reshape(f.shape).astype(bool),f)
if __name__=='__main__':unittest.main()

import numpy as np
from evaluate import masks,stats,flatten
def test_policies():
 p=np.asarray([[.299,.3,.499,.5,.899,.9]],np.float32);f=np.asarray([[1,1,0,0,0,0]],bool);m=masks(p,f,True)
 np.testing.assert_array_equal(m['primary'],[[0,0,0,0,0,1]]);np.testing.assert_array_equal(m['secondary_v3'],[[0,1,0,0,0,1]]);np.testing.assert_array_equal(m['ungated_p5'],[[0,0,0,1,1,1]])
 assert not masks(p,f,False)['primary'].any() and not masks(p,f,False)['secondary_v3'].any()
def test_summary():
 assert stats([1.,2.,3.])==dict(mean=2.,sample_sd=1.,n_folds=3);assert flatten(dict(a=1,b=dict(c=.5),d=True))=={'a':1,'b.c':.5}
if __name__=='__main__':test_policies();test_summary();print('held32 synthetic tests passed')

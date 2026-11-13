import unittest,numpy as np
import jax,jax.numpy as j
from bayesify_jax import transformers as T,gaussian_text as G
class GaussianText(unittest.TestCase):
 def test_vadam_frozen_and_sampled_updates(self):
  for arch,classes in [('gpt2',2),('roberta',3)]:
   cfg=T.Config(architecture=arch,width=4,heads=2,layers=1,intermediate=6,vocabulary=16,positions=10,rank=2,classes=classes)
   base=T.random_base(cfg);ids=j.array([[2,3],[3,2]]);mask=j.ones_like(ids);labels=j.array([0,1]);targets=j.array([0,1])
   c,p,state=G.initialize(cfg,base,optimizer='vadam',N=100,temperature=.01,likelihood_normalization=True)
   for mode in ['sampled','moment']:
    new,loss=G.make_step(c,base,targets,p,100,mode=mode,rank=2,samples=2,optimizer='vadam',temperature=.01)(state,ids,mask,labels)
    self.assertTrue(np.isfinite(float(loss)))
    self.assertTrue(all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(new)))
    self.assertEqual(len(new['optimizer']),4)
    for n,q in new['belief'].items():
     expected=1/(100/.01*new['optimizer'][2][n]+1/p[n].variance)
     np.testing.assert_allclose(q.variance,expected,rtol=1e-6)
 def test_each_original_family_update(self):
  cfg=T.Config(architecture='roberta',width=4,heads=2,layers=1,intermediate=6,vocabulary=16,positions=10,rank=2,classes=3)
  base=T.random_base(cfg);ids=j.array([[2,3],[3,2]]);mask=j.ones_like(ids);labels=j.array([0,2]);targets=j.array([0,1])
  for full in [False,True]:
   c,p,state=G.initialize(cfg,base,full_weight=full)
   for method in ['mfvi','ep']:
    for mode in ['moment','sampled']:
     new,loss=G.make_step(c,base,targets,p,100,method,mode,rank=2,samples=2)(state,ids,mask,labels)
     self.assertTrue(np.isfinite(float(loss)))
     self.assertTrue(all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(new)))
     self.assertTrue(all(np.all(np.asarray(q.variance)>0) for q in new['belief'].values()))
if __name__=='__main__':unittest.main()

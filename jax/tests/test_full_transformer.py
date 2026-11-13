import unittest,numpy as np
from dataclasses import replace
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as j
from bayesify_jax import transformers as T,heads

class FullTransformer(unittest.TestCase):
 def test_full_weight_embedding_and_reverse(self):
  c=T.Config(architecture='roberta',width=4,heads=2,layers=1,intermediate=8,vocabulary=16,positions=10,classes=3,family='full')
  base=T.random_base(c,dtype=j.float64);q=T.full_weight_beliefs(base,1e-6)
  ids=j.array([[2,3,4],[2,3,4]]);mask=j.ones_like(ids);targets=j.array([0,1]);key=jax.random.key(2);y=j.array([0,2]);rank=3
  out,states=T.tangent(base,q,ids,mask,c,targets,key,rank,True)
  self.assertGreater(float(j.linalg.norm(states[0].factors)),0)
  np.testing.assert_allclose(states[0].factors[:,0],states[0].factors[:,1],atol=1e-14)
  means={n:p.mean for n,p in q.items()};ks=jax.random.split(key,len(q))
  u={n:j.sqrt(p.variance)*jax.random.rademacher(k,(rank,)+p.mean.shape,dtype=p.mean.dtype) for (n,p),k in zip(sorted(q.items()),ks)}
  direct=jax.vmap(lambda d:jax.jvp(lambda w:T.point(base,w,ids,mask,c,targets),(means,),(d,))[1])(u)
  np.testing.assert_allclose(out.factors,direct,atol=1e-12,rtol=1e-10)
  for method in ['ep','mfvi']:
   value,g,_,_=T.moment_likelihood(base,q,ids,mask,c,targets,key,y,rank,method)
   def full(q):
    z=T.tangent(base,q,ids,mask,c,targets,key,rank)
    return heads.categorical_objective(j.concatenate((z.mean[None]+z.factors,z.mean[None]-z.factors),0),y,method)
   ref,grad=jax.value_and_grad(full)(q);np.testing.assert_allclose(value,ref,atol=1e-12)
   for a,b in zip(jax.tree_util.tree_leaves(g),jax.tree_util.tree_leaves(grad)):np.testing.assert_allclose(a,b,atol=1e-9,rtol=1e-7)
if __name__=='__main__':unittest.main()

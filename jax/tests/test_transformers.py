import unittest,numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as j
from bayesify_jax import transformers as T
from bayesify_jax.covariance import spherical_carrier

class Transformers(unittest.TestCase):
 def fixture(self,architecture):
  c=T.Config(architecture=architecture,width=8,heads=2,layers=2,intermediate=16,vocabulary=31,positions=20,rank=2,classes=3 if architecture=='roberta' else 2)
  b=T.random_base(c,dtype=j.float64);q=T.initial_beliefs(c,variance=.0002,dtype=j.float64)
  return c,b,q,j.array([[2,3,4,5],[3,2,1,1]]),j.array([[1,1,1,1],[1,1,0,0]]),j.array([6,7])
 def test_jvp_local_matches_complete_model(self):
  for arch in ['gpt2','roberta']:
   c,b,q,ids,mask,tokens=self.fixture(arch);key=jax.random.key(4);R=3
   out=T.tangent(b,q,ids,mask,c,tokens,key,R)
   means={n:p.mean for n,p in q.items()};ks=jax.random.split(key,len(q))
   u={n:j.sqrt(p.variance)*jax.random.rademacher(k,(R,)+p.mean.shape,dtype=p.mean.dtype) for (n,p),k in zip(sorted(q.items()),ks)}
   f=lambda w:T.point(b,w,ids,mask,c,tokens)
   direct=jax.vmap(lambda u:jax.jvp(f,(means,),(u,))[1])(u)
   np.testing.assert_allclose(out.mean,f(means),atol=1e-10)
   np.testing.assert_allclose(out.factors,direct,atol=1e-10)
 def test_causal_prefix(self):
  c,b,q,ids,mask,tokens=self.fixture('gpt2');w={n:p.mean for n,p in q.items()}
  x=T.embeddings(b,ids,j.ones_like(mask),c);a=T.block(x,j.ones_like(mask),b,w,c,0);changed=x.at[:,3].add(100)
  z=T.block(changed,j.ones_like(mask),b,w,c,0)
  np.testing.assert_allclose(a[:,:3],z[:,:3],atol=1e-10)
 def test_probability_average(self):
  c,b,q,ids,mask,tokens=self.fixture('gpt2');s=T.tangent(b,q,ids,mask,c,tokens,jax.random.key(2),4)
  p=T.gaussian_prediction(s);np.testing.assert_allclose(p.sum(-1),1,atol=1e-10)
  self.assertTrue(np.all(np.asarray(p)>0))
 def test_spherical_moments(self):
  plus=j.array([[1.,2.],[3.,-1.],[-2.,.5]]);minus=j.array([[-1.,.2],[1.,1.],[.5,1.]])
  draws=jax.jit(jax.vmap(lambda k:spherical_carrier(plus,minus,k)))(jax.random.split(jax.random.key(7),30000))
  m=((plus+minus)/2).mean(0);odd=(plus-minus)/2;even=(plus+minus)/2-m
  np.testing.assert_allclose(draws.mean(0),m,atol=.025)
  np.testing.assert_allclose(np.cov(np.asarray(draws).T),(odd.T@odd+even.T@even)/3,atol=.04)
 def test_nonlinear_variance_gradient(self):
  c,b,q,ids,mask,tokens=self.fixture('gpt2')
  f=lambda q:j.sum(T.nonlinear_carrier(b,q,ids,mask,c,tokens,jax.random.key(4),pairs=2)**2)
  value,g=jax.value_and_grad(f)(q)
  self.assertTrue(np.isfinite(value));self.assertTrue(all(np.all(np.isfinite(a)) for a in jax.tree_util.tree_leaves(g)))
if __name__=='__main__':unittest.main()

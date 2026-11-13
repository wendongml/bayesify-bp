import unittest,numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as j
from bayesify_jax import transformers as T,text_training as TT,heads

class TextUpdates(unittest.TestCase):
 def test_scalar_contrast_variance_and_reverse(self):
  c=T.Config(width=4,heads=2,layers=1,intermediate=6,vocabulary=16,positions=10,rank=2)
  base=T.random_base(c,dtype=j.float64);q=T.initial_beliefs(c,dtype=j.float64);ids=j.array([[2,3],[3,4]]);mask=j.ones_like(ids);targets=j.array([2,3]);labels=j.array([0,1])
  z,J=T.scalar_moments(base,q,ids,mask,c,targets)
  f=lambda w:j.diff(T.point(base,w,ids,mask,c,targets),axis=-1)[:,0]
  reference=jax.jacrev(f)({n:a.mean for n,a in q.items()})
  for n in J:np.testing.assert_allclose(J[n],reference[n],atol=1e-10)
  for method in ['mfvi','ep']:
   value,g=T.scalar_likelihood(base,q,ids,mask,c,targets,labels,method)
   objective=lambda p:heads.binary_gaussian(T.scalar_moments(base,p,ids,mask,c,targets)[0],labels,method)
   ref=jax.grad(objective)(q)
   for a,b in zip(jax.tree_util.tree_leaves(g),jax.tree_util.tree_leaves(ref)):np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-8)
 def test_global_weight_bma(self):
  c=T.Config(architecture='roberta',width=4,heads=2,layers=1,intermediate=6,vocabulary=16,positions=12,rank=2,classes=3,family='a')
  base=T.random_base(c);state,_=TT.make_state(c,base);ids=j.array([[2,3,4],[4,3,2]]);mask=j.ones_like(ids);key=jax.random.key(31);targets=j.array([0,1]);draws=3
  actual=TT.prediction(c,base,targets,'global',draws)(state,ids,mask,key)
  probabilities=[]
  for k in jax.random.split(key,draws):
   source,_=jax.random.split(k);names=tuple(sorted(state['rho']));w=dict(state['means'])
   for n,draw in zip(names,jax.random.split(source,len(names))):w[n]=w[n]+state['rho'][n]**2*jax.random.normal(draw,w[n].shape,dtype=w[n].dtype)
   probabilities.append(jax.nn.softmax(T.point(base,w,ids,mask,c,targets),-1))
  np.testing.assert_allclose(actual,j.stack(probabilities).mean(0),atol=1e-7,rtol=1e-6)
 def test_local_transformer_reverse(self):
  c=T.Config(width=4,heads=2,layers=2,intermediate=8,vocabulary=16,positions=10,rank=2)
  base=T.random_base(c,dtype=j.float64);q=T.initial_beliefs(c,dtype=j.float64);ids=j.array([[2,3,4],[3,5,6]]);mask=j.ones_like(ids);targets=j.array([2,3]);key=jax.random.key(2);labels=j.array([0,1])
  for method in ['ep','mfvi']:
   value,g,_,_=T.moment_likelihood(base,q,ids,mask,c,targets,key,labels,rank=3,method=method)
   def full(q):
    z=T.tangent(base,q,ids,mask,c,targets,key,3);return heads.categorical_objective(j.concatenate((z.mean[None]+z.factors,z.mean[None]-z.factors),0),labels,method)
   v,ref=jax.value_and_grad(full)(q);np.testing.assert_allclose(value,v,atol=1e-10)
   for a,b in zip(jax.tree_util.tree_leaves(g),jax.tree_util.tree_leaves(ref)):np.testing.assert_allclose(a,b,atol=1e-9,rtol=1e-7)
 def test_source_split_update_paths(self):
  c=T.Config(architecture='roberta',width=4,heads=2,layers=1,intermediate=6,vocabulary=16,positions=12,rank=2,classes=3,family='a')
  base=T.random_base(c);ids=j.array([[2,3,4],[4,3,2]]);mask=j.ones_like(ids);labels=j.array([0,2])
  for mode in ['ordinary','sampled','hybrid']:
   state,opt=TT.make_state(c,base);initial=state['means']['classifier.out.weight'];step=TT.make_step(c,base,j.array([0,1]),mode,100,3,opt)
   state,stats=step(state,ids,mask,labels)
   self.assertTrue(np.isfinite(float(stats[0])));self.assertGreater(float(j.linalg.norm(state['means']['classifier.out.weight']-initial)),0)
   self.assertTrue(all(np.isfinite(a).all() for a in jax.tree_util.tree_leaves(state['rho'])))
if __name__=='__main__':unittest.main()

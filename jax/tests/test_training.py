import unittest
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as j
from dataclasses import replace
from bayesify_jax import models,layers,heads,covariance
from bayesify_jax.gaussian import Moments,FullMoments,RootMoments,encode
from bayesify_jax.training import protocol,initialize,make_step,make_predictor

class Training(unittest.TestCase):
 def test_shared_relu_tail_derivatives(self):
  # Exact analytic JVP vs ordinary derivative where well-conditioned; both
  # mean and shared-factor adjoints are checked, including nonzero means.
  rng=np.random.default_rng(177)
  q=covariance.Shared(j.array(rng.normal(size=(2,7))),j.array(rng.normal(size=(4,2,7))))
  def raw(q):
   y,terms=layers.relu_terms(Moments(q.mean,q.variance));pos,s,P,*_=terms
   return covariance.Shared(y.mean,q.factors*j.where(pos,P,(q.mean>0).astype(q.mean.dtype)))
  a=covariance.Shared(j.array(rng.normal(size=(2,7))),j.array(rng.normal(size=(4,2,7))))
  scalar=lambda f,q:j.sum(f(q).mean*a.mean)+j.sum(f(q).factors*a.factors)
  for x,y in zip(jax.tree_util.tree_leaves(jax.grad(lambda q:scalar(raw,q))(q)),jax.tree_util.tree_leaves(jax.grad(lambda q:scalar(covariance.relu_shared,q))(q))):
   np.testing.assert_allclose(x,y,atol=1e-10,rtol=1e-8)
  for s in [1e-8,1e-12,1e-16,1e-18]:
   q=covariance.Shared(j.array([1.,-1.,0.],j.float32),j.full((1,3),s,j.float32))
   g=jax.jit(jax.grad(lambda q:covariance.relu_shared(q).mean.sum()+covariance.relu_shared(q).factors.sum()))(q)
   self.assertTrue(all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(g)))
   np.testing.assert_allclose(g.factors,j.array([[1.,0.,.5+1/np.sqrt(2*np.pi)]]),atol=2e-6)
 def test_root_and_covariance_heads_agree(self):
  rng=np.random.default_rng(701);root=j.array(rng.normal(size=(2,4,6)));mean=j.array(rng.normal(size=(2,4)));labels=j.array([0,2])
  a=RootMoments(mean,root);b=FullMoments(mean,a.covariance)
  np.testing.assert_allclose(heads.categorical_nodes(a),heads.categorical_nodes(b),atol=1e-10,rtol=1e-9)
  for method in ['mfvi','ep']:
   f=lambda R:heads.categorical_objective(heads.categorical_nodes(RootMoments(mean,R)),labels,method)
   g=lambda R:heads.categorical_objective(heads.categorical_nodes(FullMoments(mean,R@R.swapaxes(-1,-2))),labels,method)
   np.testing.assert_allclose(jax.grad(f)(root),jax.grad(g)(root),atol=1e-9,rtol=1e-8)
  # A small positive diagonal is represented faithfully even if a dense
  # float32 Gram matrix would round it away. No covariance jitter is added.
  root=j.concatenate((j.ones((1,4,1),j.float32)*100,j.eye(4,dtype=j.float32)[None]*1e-4),-1)
  q=RootMoments(j.zeros((1,4),j.float32),root)
  self.assertTrue(np.isfinite(jax.jit(heads.categorical_nodes)(q)).all())
  self.assertTrue(np.isfinite(jax.grad(lambda R:heads.categorical_objective(heads.categorical_nodes(RootMoments(q.mean,R)),j.array([0])))(root)).all())
 def test_correlated_readout_float32_precision(self):
  # Actual six readouts from the first ResNet epoch. Contrast covariance is
  # SPD; low-precision GPU contraction previously made its Cholesky fail.
  from pathlib import Path
  a=np.load(Path(heads.__file__).parent/'assets/readout_precision_fixture.npz')
  q=FullMoments(j.asarray(a['mean']),j.asarray(a['covariance']))
  f=jax.jit(heads.categorical_nodes);nodes=f(q)
  self.assertTrue(np.isfinite(nodes).all())
  D=np.concatenate((np.eye(9),-np.ones((9,1))),1)
  C=D@a['covariance'].astype(np.float64)@D.T
  self.assertGreater(float(np.linalg.eigvalsh(C).min()),0)
  m=a['mean'].astype(np.float64)@D.T;L=np.linalg.cholesky(C)
  delta=3*np.moveaxis(L,-1,0)
  expected=np.pad(np.concatenate((m[None]+delta,m[None]-delta)),((0,0),(0,0),(0,1)))
  np.testing.assert_allclose(nodes,expected,atol=2e-4,rtol=2e-4)
  g=jax.jit(jax.grad(lambda p:heads.categorical_objective(heads.categorical_nodes(p),j.arange(6)%10)))(q)
  self.assertTrue(all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(g)))
 def test_full_readout_literal(self):
  rng=np.random.default_rng(12);q=lambda sh:Moments(j.array(rng.normal(size=sh)),j.ones(sh)*.1)
  args=(q((3,4)),q((4,3)),q((3,)));y=layers.readout(*args);a=j.array(rng.normal(size=y.mean.shape));G=j.array(rng.normal(size=y.covariance.shape));G=(G+G.swapaxes(-1,-2))/2
  signal=encode(y,FullMoments(a,G))
  for x,y in zip(jax.tree_util.tree_leaves(layers.readout_backward(*args,signal)),jax.tree_util.tree_leaves(layers.projected(layers.readout,args,signal))):np.testing.assert_allclose(x,y,atol=1e-9,rtol=1e-8)
 def test_shared_reverse(self):
  m=models.mlp(inputs=3,width=4,classes=2,dtype=j.float64);q=m.snapshot();x=j.array([[.1,.3,.8],[-.2,.3,.6]]);y=j.array([0,1]);key=jax.random.PRNGKey(4)
  for method in ['mfvi','ep']:
   value,g,_,_=covariance.shared_likelihood(m.nodes,q,x,y,key,3,method=method)
   f=lambda q:heads.categorical_objective(heads.categorical_nodes(covariance.shared_records(m.nodes,q,x,key,3)[0]['readout']),y,method)
   v,ref=jax.value_and_grad(f)(q);np.testing.assert_allclose(value,v,atol=1e-10)
   for a,b in zip(jax.tree_util.tree_leaves(g),jax.tree_util.tree_leaves(ref)):np.testing.assert_allclose(a,b,atol=1e-9,rtol=1e-8)
 def test_five_mnist_update_paths(self):
  # Full 784-400-400-10 architecture, actual compiled updates and belief changes.
  x=j.array(np.random.default_rng(3).uniform(size=(4,28,28,1)),dtype=j.float32);y=j.array([0,1,2,3])
  for method in ['ordinary','sampled_mfvi','moment_mfvi','sampled_ep','moment_ep']:
   cfg=protocol('mlp',method);m,p,s,e=initialize(cfg,55000);step=make_step(m,p,55000,cfg,e)
   before=s['belief']['dense1.weight'].mean
   for _ in range(2):s,loss=step(s,x,y,j.float32(1.))
   self.assertTrue(np.isfinite(float(loss)),method);self.assertGreater(float(j.linalg.norm(s['belief']['dense1.weight'].mean-before)),0)
   prediction=make_predictor(m,cfg)(s,x,jax.random.PRNGKey(9));self.assertTrue(np.isfinite(prediction).all(),method);np.testing.assert_allclose(prediction.sum(-1),1,atol=1e-5)
   jax.clear_caches()
if __name__=='__main__':unittest.main()

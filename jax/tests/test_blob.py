import unittest
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as j
import optax
from bayesify_jax import blob,transformers as T

class BLoB(unittest.TestCase):
 def test_coupled_draw_identity(self):
  a=j.array([.6,.8]);xi=j.array([[.3,-.5],[.8,.9]]);nodes=blob.coupled_nodes(xi,a,jax.random.key(3))
  np.testing.assert_allclose(j.einsum('r,rij->ij',a,nodes),xi,atol=1e-12)
 def test_rank_one_output_and_update_identity(self):
  for arch in ['gpt2','roberta']:
   c=T.Config(architecture=arch,width=4,heads=2,layers=1,intermediate=6,vocabulary=16,positions=12,rank=2,classes=3,family='a')
   base=T.random_base(c,dtype=j.float64);q=T.initial_beliefs(c,dtype=j.float64);means={n:p.mean+.1 for n,p in q.items()};rho=blob.initial_rho(means,jax.random.key(6))
   ids=j.array([[2,3,4]]);mask=j.ones_like(ids);targets=j.array([4,5]);key=jax.random.key(7);aux=jax.random.key(9)
   f=lambda m,r,ref:blob.forward(base,m,r,ids,mask,c,targets,key,aux,pairs=1,reference=ref,rematerialize=False)
   np.testing.assert_allclose(f(means,rho,True),f(means,rho,False),atol=1e-10)
   a=jax.grad(lambda m,r:j.sum(f(m,r,True)**2),argnums=(0,1))(means,rho);b=jax.grad(lambda m,r:j.sum(f(m,r,False)**2),argnums=(0,1))(means,rho)
   for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):np.testing.assert_allclose(x,y,atol=1e-10,rtol=1e-7)
 def test_exact_kl_limit(self):
  m={'a.A':j.array([.1,.2]),'a.B':j.array([.7])};r={'a.A':j.array([.2,.3])}
  ref=.5*j.sum((r['a.A']**4+m['a.A']**2)/.04-1+j.log(.04/r['a.A']**4))
  np.testing.assert_allclose(blob.source_kl(m,r,epsilon=0),ref,atol=1e-12)
 def test_integer_power_schedule(self):
  # Same mathematical schedule, evaluated independently with NumPy exp2.
  # Check eager and compiled device arithmetic without relaxing tolerances.
  for cycle in (3,8,32):
   compiled=jax.jit(lambda step:blob.cycle_weight(step,cycle))
   for step in (0,1,cycle-1,cycle,cycle+1):
    exponent=np.float32(step%cycle-cycle)
    expected=np.exp2(exponent)/(np.float32(1)-np.exp2(np.float32(-cycle-1)))
    np.testing.assert_array_equal(blob.cycle_weight(step,cycle),expected)
    np.testing.assert_array_equal(compiled(j.int32(step)),expected)
 def test_split_update_against_independent_recurrence(self):
  # Independently expand AdamW, then the source's post-Adam KL derivative.
  # Three updates check bias correction/state, not just a first-step sign.
  means={'a.A':j.array([.12,-.08]),'a.B':j.array([.3,-.2])}
  rho={'a.A':j.array([.04,.05])};decay={'a.A':0.,'a.B':.01}
  optimizer=optax.scale_by_adam();state=optimizer.init(means)
  m={n:np.asarray(x).copy() for n,x in means.items()};r=np.asarray(rho['a.A']).copy()
  first={n:np.zeros_like(x) for n,x in m.items()};second={n:np.zeros_like(x) for n,x in m.items()}
  lr,vlr,prior,eps=.003,.002,.2,1e-6
  for step in range(3):
   gm={n:j.array([.2,-.1])*(step+1) for n in means};gr={'a.A':j.array([.03,-.02])}
   after={}
   for n in m:
    first[n]=.9*first[n]+.1*np.asarray(gm[n])
    second[n]=.999*second[n]+.001*np.asarray(gm[n])**2
    direction=(first[n]/(1-.9**(step+1)))/(np.sqrt(second[n]/(1-.999**(step+1)))+1e-8)
    after[n]=m[n]-lr*direction-lr*decay[n]*m[n]
   # The production schedule is explicitly float32, including in x64 tests.
   pi=np.exp2(np.float32(step-3))/(np.float32(1)-np.exp2(np.float32(-4)))
   K=np.sum(np.log(prior+eps)-np.log(r*r+eps)+(r**4+after['a.A']**2)/(2*prior**2+eps)-.5)
   dm=2*after['a.A']/(2*prior**2+eps)
   dr=-2*r/(r*r+eps)+4*r**3/(2*prior**2+eps)
   expected={**after,'a.A':after['a.A']-vlr*pi*dm}
   expected_r=r-vlr*(np.asarray(gr['a.A'])+pi*dr)
   means,rho,state,value,weight=blob.split_update(means,rho,gm,gr,state,optimizer,lr,vlr,step,3,decay,prior)
   for n in means:np.testing.assert_allclose(means[n],expected[n],atol=1e-10,rtol=1e-8)
   np.testing.assert_allclose(rho['a.A'],expected_r,atol=1e-10,rtol=1e-8)
   np.testing.assert_allclose(value,K,atol=1e-10,rtol=1e-8)
   np.testing.assert_allclose(weight,pi,atol=1e-8)
   m,r=expected,expected_r
if __name__=='__main__':unittest.main()

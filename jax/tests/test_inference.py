import unittest,numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as j
from bayesify_jax import inference as I,models,heads,messages
from bayesify_jax.gaussian import Moments
from bayesify_jax.gaussian_reference import ep_site

class Inference(unittest.TestCase):
 def test_price_readout_against_quadrature_autodiff(self):
  # Independent 2-D Gaussian integration, not the candidate cubature head.
  from numpy.polynomial.hermite import hermgauss
  t,w=hermgauss(40);eps=j.array(np.stack(np.meshgrid(t,t),-1).reshape(-1,2)*np.sqrt(2));weight=j.array(np.outer(w,w).ravel()/np.pi)
  h=j.array([.3,-.7]);m=j.array([[.2,-.3],[.1,.4]]);v=j.array([[.2,.1],[.3,.15]]);label=1
  logits=lambda m,v:h@m+eps*j.sqrt((h*h)@v)
  for method in ['mfvi','ep']:
   def objective(m,v):
    lp=jax.nn.log_softmax(logits(m,v),-1)[:,label]
    return j.sum(weight*lp) if method=='mfvi' else j.log(j.sum(weight*j.exp(lp)))
   reference=jax.grad(objective,argnums=(0,1))(m,v)
   value,g=heads.sampled_readout_adjoint(j.broadcast_to(h,(len(eps),2)),logits(m,v),label,method)
   Z=j.sum(weight*value) if method=='ep' else 1.
   actual=jax.tree_util.tree_map(lambda a:j.einsum('s,sij->ij',weight,a)/Z,g)
   for a,b in zip(actual,reference):np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-8)
  # Affine-head MFVI has nonpositive variance sensitivity.
  _,g=heads.sampled_readout_adjoint(h[None],logits(m,v)[:1],label,'mfvi')
  self.assertTrue(np.all(np.asarray(g.variance)<=0))
 def test_ep_site_and_revisit(self):
  rng=np.random.default_rng(3);d=4;Hs=[rng.normal(size=(2,d)) for _ in range(3)];ys=[rng.normal(size=2) for _ in Hs];R=np.eye(2)*.7
  for diagonal in [False,True]:
   candidate=(j.eye(d),j.zeros(d));ref=(np.eye(d),np.zeros(d));sites=[(j.zeros((d,d)),j.zeros(d)) for _ in Hs];refs=[(np.zeros((d,d)),np.zeros(d)) for _ in Hs]
   for _ in range(2):
    for i,(H,y) in enumerate(zip(Hs,ys)):
     candidate,sites[i],tilt=I.ep_replace(candidate,sites[i],j.array(H),j.array(y),j.array(R),diagonal)
     Pc,hc=ref[0]-refs[i][0],ref[1]-refs[i][1];Cc=np.linalg.inv(Pc)
     P,h,_,_=ep_site(Cc@hc,Cc,H,y,R,diagonal);refs[i]=(P,h);ref=(Pc+P,hc+h)
     for a,b in zip(candidate,ref):np.testing.assert_allclose(a,b,atol=1e-9,rtol=1e-8)
   if not diagonal:np.testing.assert_allclose(candidate[0],np.eye(d)+sum(H.T@np.linalg.solve(R,H) for H in Hs),atol=1e-9)
 def test_objective_gradients(self):
  m=models.mlp(inputs=3,width=4,classes=2,dtype=j.float64);q=m.snapshot();p=m.priors();x=j.array([[.2,-.3,.6],[.5,.2,.1]]);y=j.array([0,1]);N=100;tau=.3;power=.7
  for method in ['mfvi','ep']:
   value,g=I.objective_and_grad(m.nodes,q,p,x,y,jax.random.key(3),N,tau,method=method,power=power)
   def objective(q):
    cavity,A=I.power_cavity(q,p,N,tau*power) if method=='ep' else (q,0.)
    z=messages.forward(m.nodes,cavity,Moments(x,j.zeros_like(x)))[0][m.nodes[-1].name]
    l=-heads.categorical_objective(heads.categorical_nodes(z),y,method,power)/len(y)
    return l-A/power if method=='ep' else l+tau/N*I.kl(q,p)
   v,g2=jax.value_and_grad(objective)(q)
   np.testing.assert_allclose(value,v,atol=1e-10)
   for a,b in zip(jax.tree_util.tree_leaves(g),jax.tree_util.tree_leaves(g2)):np.testing.assert_allclose(a,b,atol=1e-9,rtol=1e-8)
 def test_kl_certificate(self):
  rng=np.random.default_rng(11);X=rng.normal(size=(7,3));y=rng.normal(size=7);P=np.eye(3)+X.T@X;C=np.linalg.inv(P);mp=C@X.T@y
  m=rng.normal(size=3);v=np.exp(rng.normal(size=3));q=Moments(j.array(m),j.array(v));p=Moments(j.zeros(3),j.ones(3))
  logZ=-.5*(7*np.log(2*np.pi)+np.linalg.slogdet(np.eye(7)+X@X.T)[1]+y@np.linalg.solve(np.eye(7)+X@X.T,y))
  ell=-.5*(7*np.log(2*np.pi)+np.sum((y-X@m)**2)+np.sum(X**2@v))
  elbo=ell-float(I.kl({'w':q},{'w':p}));K=.5*(np.sum(np.diag(P)*v)+(m-mp)@P@(m-mp)-3+np.linalg.slogdet(C)[1]-np.log(v).sum())
  np.testing.assert_allclose(logZ-elbo,K,atol=1e-10)

class FrozenCheckpoint(unittest.TestCase):
 def test_checkpoint_offsets_and_weight_blocks(self):
  from bayesify_jax.gaussian_chain import frozen_mlp,candidate
  from bayesify_jax.gaussian_reference import chain_reference
  args=frozen_mlp();forward,a,w=candidate(*args[:6]);ref=chain_reference(*jax.tree_util.tree_map(np.asarray,args[:6]))
  for actual,m,C in zip(w,ref['weight_mean'],ref['weight_covariance']):
   np.testing.assert_allclose(actual.mean,m,atol=1e-9,rtol=1e-7);np.testing.assert_allclose(actual.covariance,C,atol=1e-9,rtol=1e-7)
  self.assertGreater(float(j.linalg.norm(args[4]-jax.nn.one_hot(3,10))),.01)

if __name__=='__main__':unittest.main()

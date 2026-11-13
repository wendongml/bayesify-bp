import unittest,numpy as np
import jax,jax.numpy as j
from bayesify_jax.gaussian import Moments
from bayesify_jax.residual import network_residual_backward
class Residual(unittest.TestCase):
 def test_linear_residual_is_zero_and_gaussian_head_matches(self):
  q={'w':Moments(j.array([.2,-.1]),j.array([.1,.2]))};A=j.array([[1.,.2],[-.4,.7]])
  f=lambda w:A@w['w'];r=network_residual_backward(f,q,0,jax.random.key(2),pairs=64,head_power=16)
  # Exact one-dimensional contrast quadrature, independent of residual code.
  n,w=np.polynomial.hermite.hermgauss(100);d=A[0]-A[1];m=d@q['w'].mean;v=(d*d)@q['w'].variance
  z=m+j.sqrt(2*v)*j.asarray(n);p=jax.nn.sigmoid(z);weight=j.asarray(w)/j.sqrt(j.pi)
  for method in ['mfvi','ep']:
   fac=p if method=='ep' else j.ones_like(p);Z=j.sum(weight*p) if method=='ep' else 1.
   gm=d*j.sum(weight*fac*(1-p))/Z
   H=(-p*(1-p)+(1-p)**2) if method=='ep' else -p*(1-p)
   gv=.5*d*d*j.sum(weight*fac*H)/Z
   np.testing.assert_allclose(r[method]['gradient'].mean,gm,atol=2e-5,rtol=2e-4)
   np.testing.assert_allclose(r[method]['gradient'].variance,gv,atol=2e-5,rtol=2e-4)
if __name__=='__main__':unittest.main()

class GradientMoments(unittest.TestCase):
 def test_shared_example_second_moment(self):
  from bayesify_jax.covariance import frozen_gradient_moments
  J=j.array([[1.,.3],[-.2,.8]]);v=j.array([.1,.2]);m=j.array([.3,-.4]);y=j.array([0.,1.])
  first,second,_=frozen_gradient_moments(m,J,v,y,40)
  e=jax.random.normal(jax.random.key(83),(200000,2),dtype=j.float64)
  r=jax.nn.sigmoid(m+(e*j.sqrt(v))@J.T)-y;g=r@J/2
  np.testing.assert_allclose(first,g.mean(0),atol=.001)
  np.testing.assert_allclose(second,(g*g).mean(0),atol=.0003)

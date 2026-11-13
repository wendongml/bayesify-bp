"""Readout derivative identity against independent Hessians and quadrature."""
import unittest
import numpy as np
import torch
from bayesify_torch.heads import sampled_readout_adjoint


class ReadoutPriceTest(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(17)

    def test_per_network_formulas_match_direct_weight_hessians(self):
        # Features are arbitrary fixed values, not Gaussian random variables.
        h=torch.tensor([-.4,2.]);w=torch.randn(3,2);b=torch.randn(3)
        for method in ('mfvi','ep'):
            def objective(w):
                lp=(w@h+b).log_softmax(-1)[1]
                return lp if method=='mfvi' else lp.exp()
            value,gm,gv=sampled_readout_adjoint(h[None],(w@h+b)[None],1,method)
            H=torch.func.hessian(objective)(w).reshape(6,6)
            torch.testing.assert_close(value[0],objective(w))
            torch.testing.assert_close(gm[0],torch.func.grad(objective)(w))
            torch.testing.assert_close(gv[0],.5*H.diag().reshape(3,2))

    def test_integrated_derivatives_match_gauss_hermite_objective(self):
        t,a=np.polynomial.hermite.hermgauss(40)
        e=torch.tensor(np.stack(np.meshgrid(t,t,indexing='ij'),-1).reshape(-1,2)*np.sqrt(2))
        weight=torch.tensor(np.outer(a,a).ravel()/np.pi)
        m,v=torch.tensor([.2,-.1]),torch.tensor([.3,.5]);h=1.7
        for method in ('mfvi','ep'):
            def objective(m,v):
                lp=((m+v.sqrt()*e)*h).log_softmax(-1)[:,0]
                return weight@lp if method=='mfvi' else (weight@lp.exp()).log()
            value,gm,gv=sampled_readout_adjoint(torch.full((len(e),1),h),(m+v.sqrt()*e)*h,0,method)
            Z=weight@value if method=='ep' else 1.
            direct=torch.func.grad(objective,argnums=(0,1))(m,v)
            torch.testing.assert_close((weight[:,None]*gm[:,:,0]).sum(0)/Z,direct[0],atol=1e-10,rtol=1e-8)
            torch.testing.assert_close((weight[:,None]*gv[:,:,0]).sum(0)/Z,direct[1],atol=1e-10,rtol=1e-8)

"""Independent integration checks for the R-equation PyTorch counterpart."""
import unittest
import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
from bayesify_torch.geometry import DiagonalMoments, tilted_update
from bayesify_torch.residual import residual_backward, network_residual_backward


class ResidualTest(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_no_residual_recovers_surrogate_and_distinct_updates(self):
        q = DiagonalMoments(torch.tensor([.4]), torch.tensor([.6]))
        sampled = dict(log_likelihood=torch.tensor([-.3, -.8]),
                       gm=torch.tensor([[1.], [-2.]]), gv=torch.tensor([[2.], [3.]]))
        for method, Z in (("mfvi", 1.), ("ep", .8)):
            gaussian = dict(gm=torch.tensor([.2]), gv=torch.tensor([-.1]), Z=torch.tensor(Z))
            result = residual_backward(q, gaussian, sampled, sampled, torch.ones(2), method)
            torch.testing.assert_close(result.gradient.mean, gaussian["gm"] / Z)
            torch.testing.assert_close(result.gradient.variance, gaussian["gv"] / Z)
            if method == "ep":
                actual = tilted_update(q, result.signal)
                expected_v = q.variance + result.signal.centered - result.signal.delta_mean.square()
                torch.testing.assert_close(actual.variance, expected_v)

    def test_ep_normalizer_is_corrected_before_dividing(self):
        q = DiagonalMoments(torch.tensor([.3]), torch.tensor([.7]))
        nonlinear = dict(log_likelihood=torch.log(torch.tensor([.2, .6])),
                         gm=torch.tensor([[2.], [-1.]]), gv=torch.tensor([[3.], [4.]]))
        surrogate = dict(log_likelihood=torch.log(torch.tensor([.4, .4])),
                         gm=torch.tensor([[.5], [.5]]), gv=torch.tensor([[.1], [.1]]))
        gaussian = dict(gm=torch.tensor([.8]), gv=torch.tensor([.4]), Z=torch.tensor(.5))
        w = torch.tensor([.5, 1.5])
        r = residual_backward(q, gaussian, nonlinear, surrogate, w, "ep")
        # Scalars independently expanded, including the non-unit density ratios.
        Z = .5 + (.5 * (.2 - .4) + 1.5 * (.6 - .4)) / 2
        gm = (.8 + (.5 * (.2 * 2 - .4 * .5) + 1.5 * (.6 * -1 - .4 * .5)) / 2) / Z
        gv = (.4 + (.5 * (.2 * 3 - .4 * .1) + 1.5 * (.6 * 4 - .4 * .1)) / 2) / Z
        torch.testing.assert_close(r.normalizer, torch.tensor(Z))
        torch.testing.assert_close(r.gradient.mean, torch.tensor([gm]))
        torch.testing.assert_close(r.gradient.variance, torch.tensor([gv]))
        self.assertGreater(abs(gm - .8 / .5), .1)  # Detect missing correction.

    def test_linear_network_matches_independent_gauss_hermite(self):
        q = DiagonalMoments(torch.tensor([.3]), torch.tensor([.4]))
        def predict(w):
            return torch.stack((1.2 * w[0] + .2, -.3 * w[0] - .1))
        result = network_residual_backward(predict, q, 0, pairs=16, head_power=16)
        # Independent 80-node scalar integration of sigmoid/log sigmoid;
        # no candidate forward map, Gaussian encoder or head is called here.
        e, weights = hermgauss(80); weights /= np.sqrt(np.pi)
        contrast = 1.5 * (.3 + np.sqrt(.8) * e) + .3
        p = 1 / (1 + np.exp(-contrast)); a = 1.5
        for method in ("mfvi", "ep"):
            Z = weights @ p if method == "ep" else 1.
            gm = weights @ (a * p * (1-p) if method == "ep" else a * (1-p)) / Z
            gv = weights @ (a*a*p*(1-p)*(1-2*p)/2 if method == "ep" else -a*a*p*(1-p)/2) / Z
            torch.testing.assert_close(result[method].gradient.mean, torch.tensor([gm]), atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(result[method].gradient.variance, torch.tensor([gv]), atol=2e-5, rtol=2e-5)

    def test_nonlinear_pair_correction_matches_quadrature_pathwise_identity(self):
        # Quadrature replaces random integration in this exact identity test.
        # h(w)=log sigmoid(w + .2 w²), h0(w)=log sigmoid(w). Values and
        # derivatives are computed independently with NumPy, not autograd.
        e, w = hermgauss(100); e = np.sqrt(2)*e; w /= np.sqrt(np.pi)
        m, v = .25, .7; x = m + np.sqrt(v)*e
        ell = -np.logaddexp(0, -(x+.2*x*x)); ell0 = -np.logaddexp(0, -x)
        g = (1-1/(1+np.exp(-(x+.2*x*x))))*(1+.4*x)
        g0 = 1-1/(1+np.exp(-x))
        q = DiagonalMoments(torch.tensor([m]), torch.tensor([v]))
        sampled = lambda l, a: dict(log_likelihood=torch.tensor(l), gm=torch.tensor(a[:, None]),
                                   gv=torch.tensor((a*e/(2*np.sqrt(v)))[:, None]))
        for method in ("mfvi", "ep"):
            a = np.exp(ell) if method == "ep" else np.ones_like(e)
            b = np.exp(ell0) if method == "ep" else np.ones_like(e)
            Z = w @ a if method == "ep" else 1.
            gaussian = dict(gm=torch.tensor([w@(b*g0)]),
                            gv=torch.tensor([w@(b*g0*e/(2*np.sqrt(v)))]), Z=torch.tensor(w@b))
            r = residual_backward(q, gaussian, sampled(ell,g), sampled(ell0,g0), torch.tensor(len(w)*w), method)
            torch.testing.assert_close(r.gradient.mean, torch.tensor([w@(a*g)/Z]), atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(r.gradient.variance, torch.tensor([w@(a*g*e/(2*np.sqrt(v)))/Z]), atol=1e-12, rtol=1e-12)

    def test_invalid_ep_normalizer_is_not_clipped(self):
        q = DiagonalMoments(torch.zeros(1), torch.ones(1))
        a = dict(log_likelihood=torch.tensor([-10.]), gm=torch.zeros(1,1), gv=torch.zeros(1,1))
        b = dict(log_likelihood=torch.zeros(1), gm=torch.zeros(1,1), gv=torch.zeros(1,1))
        with self.assertRaises(ValueError):
            residual_backward(q, dict(gm=torch.zeros(1), gv=torch.zeros(1), Z=torch.tensor(.1)), a,b,torch.ones(1),"ep")


if __name__ == "__main__":
    unittest.main()

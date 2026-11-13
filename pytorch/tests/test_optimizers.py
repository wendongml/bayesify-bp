import unittest

import torch

from bayesify_torch.layers import GaussianParameter
from bayesify_torch.optimizers import EPAdam, Vadam, gaussian_kl, power_cavity


class OptimizerTest(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self.parameter = GaussianParameter((2,), prior_variance=2.0, initial_scale=0.25)
        with torch.no_grad():
            self.parameter.mean.copy_(torch.tensor([0.2, -0.1]))
            self.parameter.prior_mean.copy_(torch.tensor([0.0, 0.0]))

    def test_power_cavity_is_information_interpolation(self):
        cavity = power_cavity({"w": self.parameter}, observations=10, alpha=2).beliefs["w"]
        q, p, k = self.parameter.moments(), self.parameter.prior(), 0.2
        torch.testing.assert_close(1 / cavity.variance, (1 - k) / q.variance + k / p.variance)
        torch.testing.assert_close(
            cavity.mean / cavity.variance,
            (1 - k) * q.mean / q.variance + k * p.mean / p.variance,
        )

    def test_kl_is_zero_at_prior(self):
        with torch.no_grad():
            self.parameter.mean.copy_(self.parameter.prior_mean)
            self.parameter.log_variance.copy_(self.parameter.prior_variance.log())
        self.assertAlmostEqual(gaussian_kl([self.parameter]).item(), 0.0)

    def test_vadam_updates_mean_and_installs_positive_variance(self):
        old = self.parameter.mean.detach().clone()
        optimizer = Vadam([self.parameter], 100, 0.1, 1e-2)
        optimizer.step([torch.tensor([0.3, -0.4])], [torch.tensor([0.2, 0.5])])
        self.assertFalse(torch.equal(old, self.parameter.mean))
        self.assertTrue(torch.all(self.parameter.variance > 0))

    def test_ep_adam_uses_variance_chain_rule(self):
        old = self.parameter.log_variance.detach().clone()
        optimizer = EPAdam([self.parameter], 1e-2, 1e-2)
        optimizer.step([torch.zeros(2)], [torch.ones(2)])
        self.assertTrue(torch.all(self.parameter.log_variance < old))


if __name__ == "__main__":
    unittest.main()

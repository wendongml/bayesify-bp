import unittest

import torch
from torch.func import functional_call

from bayesify_torch.heads import categorical_quadrature_head
from bayesify_torch.inference import named_gaussian_parameters
from bayesify_torch.models import TinyResidualMLP
from bayesify_torch.optimizers import power_cavity
from bayesify_torch.training import (
    _MomentPass,
    _cavity_replacements,
    moment_ep_gradients,
    moment_likelihood_gradients,
    moment_predict,
    moment_vadam_second,
    sampled_ep_gradients,
    sampled_likelihood_gradients,
    sampled_predict,
)


class TrainingTest(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(14)
        self.model = TinyResidualMLP(inputs=4, hidden=5, outputs=3)
        self.x = torch.randn(6, 4)
        self.y = torch.arange(6) % 3

    def test_moment_mfvi_and_ep_use_same_graph_but_different_objectives(self):
        vi = moment_likelihood_gradients(self.model.network, self.x, self.y, "mfvi")
        ep = moment_likelihood_gradients(self.model.network, self.x, self.y, "ep", power=0.5)
        self.assertEqual(vi.mean.keys(), ep.mean.keys())
        self.assertNotAlmostEqual(vi.objective.item(), ep.objective.item())
        self.assertTrue(all(torch.isfinite(g).all() for g in vi.variance.values()))

    def test_sampled_gradient_moments(self):
        result = sampled_likelihood_gradients(self.model.network, self.x, self.y, samples=3)
        self.assertTrue(torch.isfinite(result.loss))
        self.assertTrue(all(torch.all(g >= 0) for g in result.second.values()))

    def test_prediction_protocols_average_probabilities(self):
        a = sampled_predict(self.model.network, self.x, samples=4)
        b = moment_predict(self.model.network, self.x)
        torch.testing.assert_close(a.sum(1), torch.ones(len(a)))
        torch.testing.assert_close(b.sum(1), torch.ones(len(b)))

    def test_power_ep_chain_rule_matches_direct_autodiff(self):
        graph, x, y = self.model.network, self.x, self.y
        alpha, observations = 0.2, 31
        candidate = moment_ep_gradients(graph, x, y, observations, alpha)

        parameters = dict(named_gaussian_parameters(graph))
        cavity = power_cavity(parameters, observations, alpha)
        passed = functional_call(
            _MomentPass(graph), _cavity_replacements(cavity), (x,), strict=False
        )
        head = categorical_quadrature_head(passed.output, y, "ep")
        direct_loss = -head.value - cavity.log_field
        coordinates = tuple(
            value for p in parameters.values() for value in (p.mean, p.log_variance)
        )
        direct = torch.autograd.grad(direct_loss, coordinates)
        for (name, parameter), gm, ga in zip(parameters.items(), direct[0::2], direct[1::2]):
            torch.testing.assert_close(candidate.mean[name], gm)
            torch.testing.assert_close(candidate.variance[name], ga / parameter.variance)

    def test_sampled_power_ep_and_moment_vadam_statistics_are_finite(self):
        graph, x, y = self.model.network, self.x, self.y
        sampled = sampled_ep_gradients(graph, x, y, 29, 0.1, samples=3)
        moment = moment_likelihood_gradients(graph, x, y, "mfvi")
        second = moment_vadam_second(graph, x, y, moment.mean)
        for collection in (sampled.mean, sampled.variance, second):
            self.assertTrue(all(torch.isfinite(value).all() for value in collection.values()))
        self.assertTrue(all((value >= 0).all() for value in second.values()))


if __name__ == "__main__":
    unittest.main()

import unittest

import torch
import torch.nn.functional as F

from bayesify_torch.correlation import (
    SharedMoments,
    StructuredGaussian,
    covariance,
    gaussian_readout,
    linear_shared,
    relu_shared,
)


class CorrelationTest(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(8)

    def test_covariance_is_factor_gram_matrix(self):
        state = SharedMoments(torch.randn(3, 4), torch.randn(7, 3, 4))
        expected = torch.stack([u.mT @ u / state.rank for u in state.factors.permute(1, 0, 2)])
        torch.testing.assert_close(covariance(state), expected)

    def test_linear_directions_equal_explicit_jvp(self):
        rank, batch, inputs, outputs = 5, 3, 4, 2
        x = SharedMoments(torch.randn(batch, inputs), torch.randn(rank, batch, inputs))
        weight, dweight = torch.randn(outputs, inputs), torch.randn(rank, outputs, inputs)
        bias, dbias = torch.randn(outputs), torch.randn(rank, outputs)
        actual = linear_shared(x, weight, dweight, bias, dbias)
        expected = []
        for r in range(rank):
            _, tangent = torch.func.jvp(
                lambda a, w, b: F.linear(a, w, b),
                (x.mean, weight, bias),
                (x.factors[r], dweight[r], dbias[r]),
            )
            expected.append(tangent)
        torch.testing.assert_close(actual.factors, torch.stack(expected))

    def test_relu_uses_exact_marginal_mean_and_price_gate(self):
        mean = torch.tensor([[-0.4, 0.7]])
        factors = torch.tensor([[[0.3, 0.1]], [[-0.2, 0.4]]])
        state = SharedMoments(mean, factors)
        out = relu_shared(state)
        scale = state.variance.sqrt()
        t = mean / scale
        probability = 0.5 * torch.erfc(-t / torch.sqrt(torch.tensor(2.0)))
        density = torch.exp(-0.5 * t.square()) / torch.sqrt(torch.tensor(2 * torch.pi))
        torch.testing.assert_close(out.mean, mean * probability + scale * density)
        torch.testing.assert_close(out.factors, factors * probability)

    def test_gaussian_readout_matches_displayed_formula(self):
        x = SharedMoments(torch.randn(2, 3), torch.randn(4, 2, 3))
        weight, weight_variance = torch.randn(5, 3), torch.rand(5, 3) * 0.1
        bias, bias_variance = torch.randn(5), torch.rand(5) * 0.1
        mean, cov = gaussian_readout(x, weight, weight_variance, bias, bias_variance)
        projected = torch.einsum("rbi,oi->rbo", x.factors, weight)
        expected = torch.einsum("rbi,rbj->bij", projected, projected) / x.rank
        expected += torch.diag_embed(F.linear(x.mean.square() + x.variance, weight_variance, bias_variance))
        torch.testing.assert_close(mean, F.linear(x.mean, weight, bias))
        torch.testing.assert_close(cov, expected)

    def test_structured_gaussian_covariance(self):
        q = StructuredGaussian(torch.zeros(3), torch.tensor([1.0, 2.0, 3.0]), torch.randn(2, 3))
        torch.testing.assert_close(q.covariance(), torch.diag(q.diagonal) + q.factors.mT @ q.factors)


if __name__ == "__main__":
    unittest.main()

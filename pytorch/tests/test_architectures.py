import unittest

import torch

from bayesify_torch.architectures import densenet40, resnet20
from bayesify_torch.geometry import DiagonalMoments, EuclideanAdjoint, encode_diagonal
from bayesify_torch.models import MomentMLP


class ArchitectureTest(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(5)

    def _check(self, model, expected_mean_parameters):
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        point = model(x)
        moment = model.forward_moments(DiagonalMoments(x, torch.zeros_like(x)))
        self.assertEqual(model.mean_parameter_count, expected_mean_parameters)
        self.assertEqual(point.shape, (1, 10))
        self.assertEqual(moment.output.mean.shape, (1, 10))
        self.assertTrue(torch.isfinite(moment.output.mean).all())
        self.assertTrue(torch.isfinite(moment.output.variance).all())
        # A nonlinear Bayesian mean need not equal the mean-weight network;
        # requiring equality here would erase the very ReLU correction tested.
        self.assertTrue(torch.all(moment.output.variance >= 0))
        self.assertGreater(moment.output.variance.sum().item(), 0)

        # Exercise the complete skip/dense graph in reverse, including every
        # architecture-specific projected layer and all shared-parent sums.
        child = encode_diagonal(
            moment.output,
            EuclideanAdjoint(torch.randn_like(moment.output.mean), torch.randn_like(moment.output.variance)),
        )
        messages = moment.backward(child)
        self.assertIsNotNone(messages.input)
        self.assertGreater(len(messages.parameters), 0)
        for signal in messages.parameters.values():
            self.assertTrue(torch.isfinite(signal.delta_mean).all())
            self.assertTrue(torch.isfinite(signal.centered).all())

    def test_resnet20(self):
        self._check(resnet20(), 269_722)

    def test_densenet40(self):
        self._check(densenet40(), 1_019_722)

    def test_mnist_mlp(self):
        model = MomentMLP().double()
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        passed = model.forward_moments(DiagonalMoments(x, torch.zeros_like(x)))
        self.assertEqual(output.shape, (2, 10))
        self.assertEqual(passed.output.mean.shape, (2, 10))
        self.assertEqual(
            sum(child.mean.numel() for child in model.modules() if hasattr(child, "mean")),
            478_410,
        )


if __name__ == "__main__":
    unittest.main()

import unittest
import warnings

import torch

from bayesify_torch import DiagonalMoments, MomentLeNet, TinyResidualMLP, mfvi_categorical_head


class GraphTest(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(11)

    def test_literal_and_projected_graph_messages(self):
        model = TinyResidualMLP()
        x = DiagonalMoments(torch.randn(8, 4), torch.rand(8, 4) + 0.3)
        projected = model.forward_moments(x)
        literal = model.forward_moments(x, literal=True)
        labels = torch.arange(8) % 3
        child = mfvi_categorical_head(projected.output, labels).signal
        a, b = projected.backward(child), literal.backward(child)
        for left, right in zip(a.input, b.input):
            torch.testing.assert_close(left, right, rtol=3e-9, atol=3e-9)
        self.assertEqual(a.parameters.keys(), b.parameters.keys())
        for name in a.parameters:
            for left, right in zip(a.parameters[name], b.parameters[name]):
                torch.testing.assert_close(left, right, rtol=3e-9, atol=3e-9)

    def test_deterministic_model_is_fx_traceable(self):
        model = TinyResidualMLP()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*has_cuda.*")
            warnings.filterwarnings("ignore", message=".*has_cudnn.*")
            warnings.filterwarnings("ignore", message=".*has_mps.*")
            warnings.filterwarnings("ignore", message=".*has_mkldnn.*")
            traced = torch.fx.symbolic_trace(model)
        value = torch.randn(2, 4)
        torch.testing.assert_close(traced(value), model(value))

    def test_lenet_uses_the_same_literal_and_projected_layers(self):
        model = MomentLeNet()
        x = DiagonalMoments(torch.randn(1, 1, 28, 28), torch.zeros(1, 1, 28, 28))
        projected = model.forward_moments(x)
        literal = model.forward_moments(x, literal=True)
        self.assertEqual(projected.output.mean.shape, (1, 10))
        child = mfvi_categorical_head(projected.output, torch.tensor([3])).signal
        a, b = projected.backward(child), literal.backward(child)
        self.assertEqual(a.parameters.keys(), b.parameters.keys())
        for name in a.parameters:
            for left, right in zip(a.parameters[name], b.parameters[name]):
                torch.testing.assert_close(left, right, rtol=3e-8, atol=3e-9)


if __name__ == "__main__":
    unittest.main()

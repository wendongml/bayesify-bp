import unittest

import torch

from bayesify_torch.geometry import DiagonalMoments, EuclideanAdjoint, encode_diagonal
from bayesify_torch.layers import (
    MomentAdd,
    MomentAvgPool2d,
    MomentConcat,
    MomentConv2d,
    MomentFlatten,
    MomentLinear,
    MomentReLU,
)


def state(shape):
    return DiagonalMoments(torch.randn(*shape), torch.rand(*shape) + 0.2)


def assert_signal_close(case, left, right, rtol=2e-9, atol=2e-9):
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


class LayerMessageTest(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(5)

    def compare(self, layer, *parents):
        projected = layer.project(*parents)
        literal = layer.literal(*parents)
        torch.testing.assert_close(projected.output.mean, literal.output.mean)
        torch.testing.assert_close(projected.output.variance, literal.output.variance)
        child = encode_diagonal(
            projected.output,
            EuclideanAdjoint(torch.randn_like(projected.output.mean), torch.randn_like(projected.output.variance)),
        )
        a, b = projected.pullback(child), literal.pullback(child)
        self.assertEqual(a.parameters.keys(), b.parameters.keys())
        for left, right in zip(a.parents, b.parents):
            assert_signal_close(self, left, right)
        for name in a.parameters:
            assert_signal_close(self, a.parameters[name], b.parameters[name])

    def test_linear(self):
        self.compare(MomentLinear(4, 3), state((6, 4)))

    def test_conv2d(self):
        self.compare(MomentConv2d(2, 3, 3, stride=2, padding=1), state((2, 2, 7, 5)))

    def test_relu(self):
        self.compare(MomentReLU(), state((5, 7)))

    def test_pool_and_flatten(self):
        image = state((2, 3, 6, 8))
        self.compare(MomentAvgPool2d(2), image)
        self.compare(MomentFlatten(), image)

    def test_add_distinct_and_same_parent(self):
        self.compare(MomentAdd(), state((3, 5)), state((3, 5)))
        self.compare(MomentAdd(same_parent=True), state((3, 5)))

    def test_concat(self):
        self.compare(MomentConcat(1), state((4, 2)), state((4, 3)))


if __name__ == "__main__":
    unittest.main()

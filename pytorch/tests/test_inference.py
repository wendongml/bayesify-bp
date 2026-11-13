import copy
import unittest

import torch

from bayesify_torch import (
    DiagonalMoments,
    TinyResidualMLP,
    apply_global_ep_tilt,
    apply_mfvi,
    ep_gaussian_head,
    mfvi_categorical_head,
    named_gaussian_parameters,
)


class InferenceUpdateTest(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(17)
        self.x = DiagonalMoments(torch.randn(5, 4), torch.zeros(5, 4))

    def messages(self, model, method, literal):
        passed = model.forward_moments(self.x, literal=literal)
        if method == "mfvi":
            head = mfvi_categorical_head(passed.output, torch.arange(5) % 3)
        else:
            head = ep_gaussian_head(passed.output, torch.zeros_like(passed.output.mean), 2.0)
        return passed.backward(head.signal).parameters

    def assert_same_belief(self, left, right):
        for (a_name, a), (b_name, b) in zip(
            named_gaussian_parameters(left.network), named_gaussian_parameters(right.network)
        ):
            self.assertEqual(a_name, b_name)
            torch.testing.assert_close(a.mean, b.mean, rtol=2e-9, atol=2e-10)
            torch.testing.assert_close(a.variance, b.variance, rtol=2e-9, atol=2e-10)

    def test_literal_and_projected_mfvi_updates_match(self):
        projected = TinyResidualMLP()
        literal = copy.deepcopy(projected)
        apply_mfvi(projected.network, self.messages(projected, "mfvi", False), rho=0.01)
        apply_mfvi(literal.network, self.messages(literal, "mfvi", True), rho=0.01)
        self.assert_same_belief(projected, literal)

    def test_literal_and_projected_global_ep_updates_match(self):
        projected = TinyResidualMLP()
        literal = copy.deepcopy(projected)
        apply_global_ep_tilt(projected.network, self.messages(projected, "ep", False), rho=0.001)
        apply_global_ep_tilt(literal.network, self.messages(literal, "ep", True), rho=0.001)
        self.assert_same_belief(projected, literal)


if __name__ == "__main__":
    unittest.main()

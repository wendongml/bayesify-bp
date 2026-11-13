import unittest

import torch
import math

from bayesify_torch.tutorial_evidence import (
    approximation_benchmark,
    exact_gaussian_equivalence,
    nonlinear_message_equivalence,
    _node_loss,
)


class TutorialEvidenceTest(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_exact_gaussian_all_four_message_types_agree(self):
        for reference, computed in exact_gaussian_equivalence().values():
            torch.testing.assert_close(reference, computed, atol=1e-10, rtol=1e-9)

    def test_ep_and_mfvi_literal_messages_match_projected_vjp(self):
        for method in ("ep", "mfvi"):
            for reference, computed in nonlinear_message_equivalence(method).values():
                torch.testing.assert_close(reference, computed, atol=1e-10, rtol=1e-8)

    def test_native_approximation_benchmark_is_finite(self):
        rows, estimates = approximation_benchmark(
            reference_samples=128, ranks=(1, 4), mc_budgets=(4, 16)
        )
        self.assertEqual(len(rows), 5)
        self.assertIn("reference", estimates)
        for row in rows:
            for field in (
                "seconds",
                "mean_error",
                "variance_error",
                "mean_message_error",
                "second_message_error",
            ):
                self.assertTrue(torch.isfinite(torch.tensor(row[field])))

    def test_ep_uses_factorwise_normalizer_not_expected_log(self):
        logits = torch.tensor([[[0., 2.], [2., 0.]], [[3., 0.], [0., 1.]]], requires_grad=True)
        labels = torch.tensor([1, 0])
        p = logits.softmax(-1)
        expected_ep = -torch.log((p[:, 0, 1].mean(),)[0]) - torch.log(p[:, 1, 0].mean())
        torch.testing.assert_close(_node_loss(logits, labels, "ep"), expected_ep)
        self.assertGreater(_node_loss(logits, labels, "mfvi").item(), expected_ep.item())
        # Scaling by node count is essential: repeating identical nodes must
        # not change the objective or its aggregate derivative.
        repeated = logits.repeat(2, 1, 1)
        torch.testing.assert_close(_node_loss(repeated, labels, "ep"), expected_ep)

    def test_ep_fidelity_and_independent_reference_repeat(self):
        rows, values = approximation_benchmark(reference_samples=256, ranks=(4,),
                                               mc_budgets=(32,), method="ep")
        self.assertIn("reference_repeat_errors", values)
        self.assertEqual(len(rows), 3)
        self.assertFalse(torch.equal(values["reference"][0], values["reference_repeat"][0]))
        for row in rows:
            self.assertTrue(math.isfinite(row["centered_message_error"]))


if __name__ == "__main__":
    unittest.main()

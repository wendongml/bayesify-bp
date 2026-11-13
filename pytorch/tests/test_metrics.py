import unittest

import numpy as np

from bayesify_torch.metrics import ood_metrics, ranking_curve, uncertainty_metrics


class MetricTest(unittest.TestCase):
    def test_perfect_classifier(self):
        p = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
        m = uncertainty_metrics(p, np.array([0, 1, 0, 1]), bins=2)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertAlmostEqual(m["brier"], 0.05)
        self.assertAlmostEqual(m["ece"], 0.15)

    def test_ties_do_not_depend_on_input_order(self):
        score = np.array([0.5, 0.5, 0.2, 0.2, 0.1])
        positive = np.array([1, 0, 1, 0, 0], dtype=bool)
        reference = ranking_curve(score, positive)
        permutation = np.array([1, 0, 3, 2, 4])
        other = ranking_curve(score[permutation], positive[permutation])
        self.assertAlmostEqual(reference.auroc, other.auroc)
        self.assertAlmostEqual(reference.auprc, other.auprc)

    def test_ood_direction(self):
        p_id = np.array([[0.99, 0.01], [0.98, 0.02]])
        p_ood = np.array([[0.50, 0.50], [0.55, 0.45]])
        m = ood_metrics(p_id, p_ood)
        self.assertEqual(m["entropy_auroc"], 1.0)
        self.assertEqual(m["msp_auroc"], 1.0)


if __name__ == "__main__":
    unittest.main()

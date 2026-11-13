import unittest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bayesify_torch.metrics import uncertainty_metrics, ranking_curve
from bayesify_torch.plotting import risk_coverage_plot, ood_roc_plot


class ReaderResultsTest(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_plotted_risk_has_same_mean_as_table_aurc(self):
        # Entropy and maximum-confidence order differ on this fixture.
        p = np.array([[.6, .4, 0], [.61, .195, .195], [.9, .05, .05]])
        y = np.array([0, 1, 0])
        curve = risk_coverage_plot(p, y).lines[0].get_ydata()
        self.assertAlmostEqual(curve.mean(), uncertainty_metrics(p, y)["aurc"])

    def test_plot_and_table_use_same_tie_groups(self):
        id_score, ood_score = np.array([.1, .5, .5]), np.array([.5, .5, .9])
        line = ood_roc_plot(id_score, ood_score).lines[0]
        ref = ranking_curve(np.r_[id_score, ood_score], [False]*3 + [True]*3)
        np.testing.assert_array_equal(line.get_xdata(), ref.fpr)
        np.testing.assert_array_equal(line.get_ydata(), ref.tpr)

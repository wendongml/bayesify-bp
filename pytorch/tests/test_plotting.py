import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bayesify_torch.plotting import agreement_plot, uncertainty_panel


class PlottingTest(unittest.TestCase):
    def test_agreement_plot_has_identity_line_and_square_axes(self):
        ax = agreement_plot([0, 1, 2], [0, 1.1, 1.9])
        self.assertEqual(len(ax.lines), 1)
        self.assertEqual(ax.get_aspect(), 1.0)
        plt.close(ax.figure)

    def test_uncertainty_panel(self):
        probability = np.array([[.8, .2], [.4, .6], [.7, .3], [.1, .9]])
        labels = np.array([0, 1, 1, 1])
        figure, axes = uncertainty_panel(
            probability, labels, id_score=[.1, .2], ood_score=[.7, .8], method="test"
        )
        self.assertEqual(len(axes), 3)
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()

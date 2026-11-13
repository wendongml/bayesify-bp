"""Closed-form score/ranking fixtures, independent of the metric algorithms."""
import unittest
import numpy as np
from bayesify_jax.uncertainty import uncertainty_metrics, ranking_curve


class Uncertainty(unittest.TestCase):
    def test_scores_and_perfect_error_ranking(self):
        p=np.array([[.9,.1],[.6,.4],[.2,.8],[.4,.6]])
        actual=uncertainty_metrics(p,np.array([0,1,1,0]),bins=10)
        expected={'accuracy':.5,'nll':-np.log([.9,.4,.8,.4]).mean(),
                  'brier':.385,'ece':.375,'aurc':5/24,
                  'excess_aurc':0.,'error_auroc':1.,'error_auprc':1.}
        for name,value in expected.items():
            self.assertAlmostEqual(actual[name],value,places=12)

    def test_tied_ranking_is_order_independent(self):
        # Each of the two score levels contains one positive and one negative.
        score=np.array([.2,.2,.8,.8]);positive=np.array([0,1,0,1])
        for order in [[0,1,2,3],[1,0,3,2],[3,1,2,0]]:
            curve=ranking_curve(score[order],positive[order])
            self.assertEqual(curve.auroc,.5)
            self.assertEqual(curve.auprc,.5)
            self.assertEqual(curve.fpr95,1.)


if __name__=='__main__':unittest.main()

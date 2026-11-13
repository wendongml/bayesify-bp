import unittest
import torch
import math
from bayesify_torch.geometry import DiagonalMoments, EuclideanAdjoint, encode_diagonal, decode_diagonal
from bayesify_torch.layers import MomentMaxPool2d, gaussian_max_pair


class MaxPoolTest(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(91)

    def test_iid_standard_normal_pair_has_known_moments(self):
        q=DiagonalMoments(torch.zeros(2),torch.ones(2))
        out=gaussian_max_pair(q,q)
        torch.testing.assert_close(out.mean,torch.full((2,),1/math.sqrt(math.pi)))
        torch.testing.assert_close(out.variance,torch.full((2,),1-1/math.pi))

    def test_handwritten_reverse_agrees_with_autodiff(self):
        for variance in [torch.rand(2,3,4,4)+.2,torch.zeros(2,3,4,4)]:
            q=DiagonalMoments(torch.randn(2,3,4,4),variance)
            layer=MomentMaxPool2d()
            left,right=layer.literal(q),layer.project(q)
            signal=encode_diagonal(left.output,EuclideanAdjoint(torch.randn_like(left.output.mean),
                                                                torch.randn_like(left.output.mean)))
            for a,b in zip(left.pullback(signal).parents[0],right.pullback(signal).parents[0]):
                torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-8)

    def test_zero_uncertainty_matches_real_maxpool(self):
        x=torch.randn(2,3,4,4)
        layer=MomentMaxPool2d()
        q=layer.forward_moments(DiagonalMoments(x,torch.zeros_like(x)))
        torch.testing.assert_close(q.mean,layer(x))
        self.assertEqual(q.variance.abs().max(),0)

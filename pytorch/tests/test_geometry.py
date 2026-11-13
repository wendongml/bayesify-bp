import unittest

import torch

from bayesify_torch.geometry import (
    DiagonalMoments,
    EuclideanAdjoint,
    FullMoments,
    canonical_mfvi_update,
    decode_diagonal,
    decode_full,
    encode_diagonal,
    encode_full,
    tilted_update,
)
from bayesify_torch.heads import ep_gaussian_head, mfvi_gaussian_head


class GeometryTest(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(3)

    def test_diagonal_round_trip_including_deterministic_coordinate(self):
        q = DiagonalMoments(
            torch.randn(8), torch.tensor([0.0, 1e-197, 0.2, 0.4, 0.7, 1.0, 1.4, 2.0])
        )
        g = EuclideanAdjoint(torch.randn(8), torch.randn(8))
        recovered = decode_diagonal(q, encode_diagonal(q, g))
        torch.testing.assert_close(recovered.mean, g.mean)
        torch.testing.assert_close(recovered.variance, g.variance)

    def test_full_round_trip(self):
        matrix = torch.randn(5, 5)
        covariance = matrix @ matrix.mT + 0.5 * torch.eye(5)
        q = FullMoments(torch.randn(5), covariance)
        gm = torch.randn(5)
        gc = torch.randn(5, 5)
        gc = (gc + gc.mT) / 2
        recovered = decode_full(q, encode_full(q, (gm, gc)))
        torch.testing.assert_close(recovered[0], gm, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(recovered[1], gc, rtol=1e-10, atol=1e-10)

    def test_scalar_ep_and_mfvi_counterexample(self):
        q = DiagonalMoments(torch.zeros(1), torch.ones(1))
        y = torch.zeros(1)
        ep = ep_gaussian_head(q, y, noise_variance=1.0)
        ep_updated = tilted_update(q, ep.signal)
        self.assertAlmostEqual(ep_updated.variance.item(), 0.5, places=12)

        vi = mfvi_gaussian_head(q, y, noise_variance=1.0)
        vi_updated = canonical_mfvi_update(q, vi.signal, q)
        self.assertAlmostEqual(vi_updated.variance.item(), 0.5, places=12)
        wrong_variance = q.variance + vi.signal.delta_raw_second
        self.assertAlmostEqual(wrong_variance.item(), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()

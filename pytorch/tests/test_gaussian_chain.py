import unittest

import torch

from bayesify_torch.gaussian_chain import (
    backprop_gaussian_chain,
    condition_gaussian_chain,
    gaussian_chain_fixture,
    install_full,
    posterior_signal,
    smooth_gaussian_chain,
)


class GaussianChainTest(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self.fixture = gaussian_chain_fixture()

    def test_dense_conditioning_matches_local_smoothing(self):
        layers, weights, input_belief, observation, observation_covariance = self.fixture
        forward, smoother = smooth_gaussian_chain(
            layers, weights, input_belief, observation, observation_covariance
        )
        dense = condition_gaussian_chain(
            layers, weights, input_belief, observation, observation_covariance
        )
        for left, right in zip(smoother.activations[1:], dense.activations[1:]):
            torch.testing.assert_close(left.mean, right.mean, atol=1e-10, rtol=1e-9)
            torch.testing.assert_close(left.covariance, right.covariance, atol=1e-10, rtol=1e-9)
        for left, right in zip(smoother.weights, dense.weights):
            torch.testing.assert_close(left.mean, right.mean, atol=1e-10, rtol=1e-9)
            torch.testing.assert_close(left.covariance, right.covariance, atol=1e-10, rtol=1e-9)
        self.assertEqual(len(forward.activations), len(layers) + 1)

    def test_fisher_backprop_installs_exact_posterior_moments(self):
        layers, weights, input_belief, observation, observation_covariance = self.fixture
        forward, messages = backprop_gaussian_chain(
            layers, weights, input_belief, observation, observation_covariance
        )
        dense = condition_gaussian_chain(
            layers, weights, input_belief, observation, observation_covariance
        )
        for prior, message, exact in zip(forward.activations[1:], messages.activations, dense.activations[1:]):
            installed = install_full(prior, message)
            torch.testing.assert_close(installed.mean, exact.mean, atol=1e-10, rtol=1e-9)
            torch.testing.assert_close(installed.covariance, exact.covariance, atol=1e-10, rtol=1e-9)
        for prior, message, exact in zip(weights, messages.weights, dense.weights):
            installed = install_full(prior, message)
            torch.testing.assert_close(installed.mean, exact.mean, atol=1e-10, rtol=1e-9)
            torch.testing.assert_close(installed.covariance, exact.covariance, atol=1e-10, rtol=1e-9)

    def test_backprop_messages_equal_independent_raw_moment_corrections(self):
        layers, weights, input_belief, observation, observation_covariance = self.fixture
        forward, messages = backprop_gaussian_chain(
            layers, weights, input_belief, observation, observation_covariance
        )
        dense = condition_gaussian_chain(
            layers, weights, input_belief, observation, observation_covariance
        )
        for prior, candidate, exact in zip(forward.activations[1:], messages.activations, dense.activations[1:]):
            reference = posterior_signal(prior, exact)
            torch.testing.assert_close(candidate.delta_mean, reference.delta_mean, atol=1e-10, rtol=1e-9)
            torch.testing.assert_close(candidate.delta_raw_second, reference.delta_raw_second, atol=1e-10, rtol=1e-9)
        for prior, candidate, exact in zip(weights, messages.weights, dense.weights):
            reference = posterior_signal(prior, exact)
            torch.testing.assert_close(candidate.delta_mean, reference.delta_mean, atol=1e-10, rtol=1e-9)
            torch.testing.assert_close(candidate.delta_raw_second, reference.delta_raw_second, atol=1e-10, rtol=1e-9)


if __name__ == "__main__":
    unittest.main()

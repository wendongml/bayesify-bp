import unittest
import copy

import torch

from bayesify_torch.benchmark import GaussianTrainer, TrainingConfig, evaluate_batches
from bayesify_torch.models import TinyResidualMLP
from bayesify_torch.graph import MomentDAG
from bayesify_torch.layers import MomentBatchNorm2d, MomentGlobalAvgPool2d, MomentLinear


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(21)
        self.x = torch.randn(8, 4)
        self.y = torch.arange(8) % 3

    def test_all_five_training_paths_are_finite(self):
        configurations = [
            ("point", "point", 1),
            ("mfvi", "sampled", 2),
            ("mfvi", "moment", 1),
            ("ep", "sampled", 3),
            ("ep", "moment", 1),
        ]
        for method, approximation, samples in configurations:
            with self.subTest(method=method, approximation=approximation):
                graph = TinyResidualMLP(4, 5, 3).network.double()
                trainer = GaussianTrainer(
                    graph,
                    TrainingConfig(
                        method,
                        approximation,
                        observations=53,
                        learning_rate=2e-4,
                        variance_learning_rate=1e-4,
                        samples=samples,
                    ),
                )
                loss = trainer.step(self.x, self.y)
                self.assertTrue(torch.isfinite(torch.tensor(loss)))
                result = evaluate_batches(trainer, [(self.x, self.y)], draws=3)
                self.assertTrue(0 <= result["accuracy"] <= 1)
                self.assertTrue(torch.isfinite(torch.tensor(result["nll"])))

    def test_real_mnist_image_batches_use_the_same_graph_for_all_methods(self):
        # The dataset yields NCHW images, not pre-flattened teaching tensors.
        # This exercises the public trainer/model boundary and full readout.
        from bayesify_torch.models import MomentMLP, MomentLeNet
        from bayesify_torch.inference import named_gaussian_parameters
        for constructor in (MomentMLP, MomentLeNet):
            for method, mode in (("point","point"),("mfvi","sampled"),
                                 ("mfvi","moment"),("ep","sampled"),("ep","moment")):
                with self.subTest(model=constructor.__name__,method=method,mode=mode):
                    graph=constructor().network.double()
                    with torch.no_grad():
                        for _,p in named_gaussian_parameters(graph):
                            p.prior_variance.fill_(.001);p.log_variance.fill_(-9.210340371976184)
                    trainer=GaussianTrainer(graph,TrainingConfig(method,mode,55000,1e-4,full_readout=True))
                    x=torch.rand(2,1,28,28);y=torch.tensor([3,8])
                    self.assertTrue(torch.isfinite(torch.tensor(trainer.step(x,y))))
                    result=evaluate_batches(trainer,[(x,y)],draws=2)
                    self.assertTrue(torch.isfinite(torch.tensor(result['nll'])))

    def test_repeating_minibatch_does_not_change_average_objective_or_updates(self):
        # N is the dataset size, not batch size. Duplicating every minibatch
        # example leaves mean likelihood and the one global field unchanged.
        # Identical RNG states keep complete-network draws/HVP probes paired.
        for method, mode, rank in (("mfvi", "sampled",0), ("mfvi", "moment",0),
                                  ("ep", "sampled",0), ("ep", "moment",0),
                                  ("mfvi","moment",3), ("ep","moment",3)):
            with self.subTest(method=method, mode=mode, rank=rank):
                g = TinyResidualMLP(4, 5, 3).network.double()
                config = TrainingConfig(method, mode, 53, 2e-4, samples=3, rank=rank)
                a, b = GaussianTrainer(g, config), GaussianTrainer(copy.deepcopy(g), config)
                for step in range(3):
                    la = a.step(self.x, self.y, torch.Generator().manual_seed(100+step))
                    lb = b.step(self.x.repeat(2, 1), self.y.repeat(2),
                                torch.Generator().manual_seed(100+step))
                    self.assertAlmostEqual(la, lb, places=10)
                    for name, p in a.named.items():
                        torch.testing.assert_close(p.mean, b.named[name].mean, atol=1e-11, rtol=1e-9)
                        torch.testing.assert_close(p.variance, b.named[name].variance, atol=1e-13, rtol=1e-9)

    def test_prediction_uses_same_moments_without_building_backward_records(self):
        from unittest.mock import patch
        from bayesify_torch.models import MomentMLP, MomentLeNet
        from bayesify_torch.geometry import DiagonalMoments
        from bayesify_torch.training import moment_predict
        from bayesify_torch.correlation import root_nodes
        for constructor in (MomentMLP, MomentLeNet):
            graph=constructor().network.double().eval()
            x=torch.rand(2,1,28,28);q=DiagonalMoments(x,torch.zeros_like(x))
            expected=root_nodes(graph.forward_moments(q,full_readout=True).output).softmax(-1).mean(0)
            with patch('torch.func.vjp',side_effect=AssertionError('Prediction must not build a pullback')):
                actual=moment_predict(graph,x,full_readout=True)
            torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-10)

    def test_normalization_is_committed_once_per_batch_for_all_training_paths(self):
        x = torch.randn(4, 3, 4, 4) + 2
        y = torch.arange(4) % 3
        expected_mean, expected_var = x.mean((0,2,3)), x.var((0,2,3), unbiased=True)
        for method, mode in (("point", "point"), ("mfvi", "sampled"), ("mfvi", "moment"),
                             ("ep", "sampled"), ("ep", "moment")):
            with self.subTest(method=method, mode=mode):
                graph = MomentDAG([
                    ("norm", MomentBatchNorm2d(3), ("input",)),
                    ("pool", MomentGlobalAvgPool2d(), ("norm",)),
                    ("head", MomentLinear(3,3), ("pool",))]).double()
                trainer = GaussianTrainer(graph, TrainingConfig(method,mode,53,2e-4,samples=3))
                norm = graph.layers['norm']
                # Merely inspecting a point pass must not mutate state.
                graph(x); graph(x)
                torch.testing.assert_close(norm.running_mean, torch.zeros(3))
                torch.testing.assert_close(norm.running_var, torch.ones(3))
                for step in range(1, 4):
                    trainer.step(x,y,torch.Generator().manual_seed(700+step))
                    torch.testing.assert_close(norm.running_mean,(1-.9**step)*expected_mean)
                    torch.testing.assert_close(norm.running_var,.9**step+(1-.9**step)*expected_var)
                saved = {n:a.clone() for n,a in graph.named_buffers()}
                graph.eval()
                evaluate_batches(trainer, [(x,y)], draws=4)
                for n,a in graph.named_buffers():torch.testing.assert_close(a,saved[n])


if __name__ == "__main__":
    unittest.main()

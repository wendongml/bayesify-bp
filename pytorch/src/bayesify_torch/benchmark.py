"""Small, explicit training loop shared by MNIST and CIFAR examples.

Dataset download, checkpoint selection and plotting stay outside the optimizer
so a reader can see the complete update in one screen.  Each Bayesian mode
uses one Gaussian belief and averages class *probabilities* at prediction time.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor

from .graph import MomentDAG
from .inference import named_gaussian_parameters
from .optimizers import EPAdam, Vadam
from .training import (
    moment_ep_gradients,
    moment_likelihood_gradients,
    moment_predict,
    moment_vadam_second,
    sampled_ep_gradients,
    sampled_likelihood_gradients,
    sampled_predict,
    normalization_observations,
)


@dataclass(frozen=True)
class TrainingConfig:
    method: str                    # point, mfvi, or ep
    approximation: str            # point, sampled, or moment
    observations: int
    learning_rate: float
    variance_learning_rate: float | None = None
    temperature: float = 0.01
    power: float = 1.0
    samples: int = 1
    weight_decay: float = 0.0
    rank: int = 0                  # 0: diagonal activations; >0: shared covariance directions
    direction_seed: int = 41
    full_readout: bool = False
    curvature_refresh: int = 1
    adam_beta2: float = 0.999

    def __post_init__(self):
        allowed = {("point", "point"), ("mfvi", "sampled"), ("mfvi", "moment"),
                   ("ep", "sampled"), ("ep", "moment")}
        if (self.method, self.approximation) not in allowed:
            raise ValueError(f"unsupported method/approximation: {self.method}/{self.approximation}")
        if self.rank < 0:
            raise ValueError("rank cannot be negative")
        if self.curvature_refresh < 1:
            raise ValueError("curvature_refresh must be positive")


class GaussianTrainer:
    """One minibatch update with no hidden inference substitution."""

    def __init__(self, graph: MomentDAG, config: TrainingConfig):
        self.graph, self.config = graph, config
        self.named = dict(named_gaussian_parameters(graph))
        self.updates = 0
        self.curvature = {name:torch.zeros_like(p.mean) for name,p in self.named.items()}
        parameters = list(self.named.values())
        variance_rate = config.variance_learning_rate or config.learning_rate
        if config.method == "point":
            self.optimizer = torch.optim.AdamW(
                [parameter.mean for parameter in parameters],
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=(0.9, config.adam_beta2),
            )
        elif config.method == "mfvi":
            self.optimizer = Vadam(
                parameters,
                config.observations,
                config.temperature,
                config.learning_rate,
            )
        else:
            self.optimizer = EPAdam(parameters, config.learning_rate, variance_rate)

    @torch.no_grad()
    def commit_statistics(self, observations):
        if self.graph.training:
            for name, (mean, variance) in observations.items():
                layer = self.graph.layers[name]
                layer.running_mean.lerp_(mean, layer.momentum)
                layer.running_var.lerp_(variance, layer.momentum)

    def step(
        self, inputs: Tensor, labels: Tensor, generator: torch.Generator | None = None
    ) -> float:
        c = self.config
        names = tuple(self.named)
        if c.method == "point":
            self.optimizer.zero_grad(set_to_none=True)
            states = self.graph(inputs, return_states=True)
            loss = F.cross_entropy(states[self.graph.nodes[-1].name], labels)
            loss.backward()
            self.optimizer.step()
            self.commit_statistics(normalization_observations(self.graph, states))
            return float(loss.detach())

        if c.method == "mfvi" and c.approximation == "sampled":
            result = sampled_likelihood_gradients(
                self.graph, inputs, labels, c.samples, generator
            )
            self.optimizer.step(
                [result.mean[name] for name in names],
                [result.second[name] for name in names],
            )
            self.commit_statistics(result.statistics)
            return float(result.loss)

        if c.method == "mfvi":
            # The head is a SUM by default for message calculations. Vadam
            # consumes the gradient of the MEAN minibatch likelihood, just
            # like sampled_likelihood_gradients and the point-loss HVP.
            result = moment_likelihood_gradients(
                self.graph, inputs, labels, "mfvi", scale=1 / len(labels),
                rank=c.rank, seed=c.direction_seed,
                full_readout=c.full_readout,
            )
            if self.updates % c.curvature_refresh == 0:
                probe = torch.Generator(device=inputs.device).manual_seed(c.direction_seed)
                second = moment_vadam_second(self.graph, inputs, labels, result.mean, probe)
                self.curvature = {n:(second[n]-g.square()).detach() for n,g in result.mean.items()}
            second = {n:g.square()+self.curvature[n] for n,g in result.mean.items()}
            self.optimizer.step(
                [result.mean[name] for name in names],
                [second[name] for name in names],
            )
            self.commit_statistics(result.statistics)
            self.updates += 1
            return float(-result.objective.detach())

        alpha = c.temperature * c.power
        if c.approximation == "sampled":
            result = sampled_ep_gradients(
                self.graph,
                inputs,
                labels,
                c.observations,
                alpha,
                c.samples,
                c.power,
                generator,
                scale=1 / len(labels),
            )
        else:
            result = moment_ep_gradients(
                self.graph,
                inputs,
                labels,
                c.observations,
                alpha,
                c.power,
                scale=1 / len(labels),
                rank=c.rank, seed=c.direction_seed,
                full_readout=c.full_readout,
            )
        self.optimizer.step(
            [result.mean[name] for name in names],
            [result.variance[name] for name in names],
        )
        self.commit_statistics(result.statistics)
        # Only the likelihood is averaged. The external field was counted
        # once globally; dividing it by B would weaken the prior as B grows.
        return float(result.loss.detach())

    @torch.no_grad()
    def probabilities(
        self, inputs: Tensor, draws: int = 32, generator: torch.Generator | None = None
    ) -> Tensor:
        c = self.config
        if c.method == "point":
            return self.graph(inputs).softmax(-1)
        if c.approximation == "moment":
            return moment_predict(self.graph, inputs, c.rank, c.direction_seed, c.full_readout)
        return sampled_predict(self.graph, inputs, draws, generator)


@torch.no_grad()
def evaluate_batches(
    trainer: GaussianTrainer,
    batches: Iterable[tuple[Tensor, Tensor]],
    draws: int = 32,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor | float]:
    """Predictive NLL uses ``-log E_q[p(y|x,W)]``, never mean sampled CE."""
    probabilities, labels = [], []
    for inputs, target in batches:
        probabilities.append(trainer.probabilities(inputs, draws, generator).cpu())
        labels.append(target.cpu())
    probability, label = torch.cat(probabilities), torch.cat(labels)
    selected = probability.gather(1, label[:, None]).clamp_min(1e-30)
    return {
        "probability": probability,
        "label": label,
        "accuracy": float((probability.argmax(1) == label).double().mean()),
        "nll": float(-selected.log().mean()),
    }


__all__ = ["TrainingConfig", "GaussianTrainer", "evaluate_batches"]

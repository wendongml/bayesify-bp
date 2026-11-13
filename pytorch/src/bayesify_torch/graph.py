"""An explicit DAG forward pass and reverse message traversal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from torch import Tensor, nn

from .geometry import DiagonalMoments, MomentSignal, add_signals
from .layers import LayerMessages, LocalRecord
from .correlation import RootMoments


@dataclass(frozen=True)
class Node:
    name: str
    parents: tuple[str, ...]


@dataclass(frozen=True)
class RecordedNode:
    node: Node
    record: LocalRecord


@dataclass(frozen=True)
class GraphMessages:
    input: MomentSignal | None
    activations: dict[str, MomentSignal | tuple[Tensor, Tensor]]
    parameters: dict[str, MomentSignal]


@dataclass(frozen=True)
class GraphPass:
    output: DiagonalMoments | RootMoments
    states: dict[str, DiagonalMoments | RootMoments]
    records: tuple[RecordedNode, ...]

    def backward(self, child: MomentSignal | tuple[Tensor, Tensor]) -> GraphMessages:
        # Only a full readout receives an ordinary (mean, root) adjoint.
        # Its local pullback encodes Gaussian signals for all upstream nodes.
        pending = {self.records[-1].node.name: child}
        activation_messages = {}
        parameter_messages: dict[str, MomentSignal] = {}
        for item in reversed(self.records):
            signal = pending.get(item.node.name)
            if signal is None:
                continue
            messages: LayerMessages = item.record.pullback(signal)
            activation_messages[item.node.name] = signal
            for parent, parent_message in zip(item.node.parents, messages.parents):
                pending[parent] = add_signals(pending.get(parent), parent_message)
            for parameter, parameter_message in messages.parameters.items():
                key = f"{item.node.name}.{parameter}"
                parameter_messages[key] = add_signals(parameter_messages.get(key), parameter_message)
        return GraphMessages(pending.get("input"), activation_messages, parameter_messages)


class MomentDAG(nn.Module):
    """A small, inspectable graph whose nodes are ordinary ``nn.Module`` objects."""

    def __init__(self, nodes: Iterable[tuple[str, nn.Module, tuple[str, ...]]]):
        super().__init__()
        triples = list(nodes)
        self.layers = nn.ModuleDict({name: module for name, module, _ in triples})
        self.nodes = tuple(Node(name, tuple(parents)) for name, _, parents in triples)
        known = {"input"}
        for node in self.nodes:
            if not set(node.parents) <= known:
                raise ValueError(f"unknown or forward parent in node {node.name}")
            known.add(node.name)

    def forward(self, x: Tensor, return_states: bool = False):
        states: dict[str, Tensor] = {"input": x}
        for node in self.nodes:
            states[node.name] = self.layers[node.name](*(states[parent] for parent in node.parents))
        return states if return_states else states[self.nodes[-1].name]

    def forward_moments(self, x: DiagonalMoments, *, literal: bool = False, full_readout: bool = False) -> GraphPass:
        states: dict[str, DiagonalMoments] = {"input": x}
        records: list[RecordedNode] = []
        method = "literal" if literal else "project"
        for node in self.nodes:
            parents = tuple(states[parent] for parent in node.parents)
            operation = 'project_full' if full_readout and node is self.nodes[-1] else method
            record: LocalRecord = getattr(self.layers[node.name], operation)(*parents)
            states[node.name] = record.output
            records.append(RecordedNode(node, record))
        return GraphPass(states[self.nodes[-1].name], states, tuple(records))

    def forward_moment_values(self, x: DiagonalMoments, *, full_readout=False):
        """The same layer mathematics, without constructing unused pullbacks."""
        states = {"input": x}
        for node in self.nodes:
            operation = "forward_full_moments" if full_readout and node is self.nodes[-1] else "forward_moments"
            states[node.name] = getattr(self.layers[node.name], operation)(
                *(states[parent] for parent in node.parents))
        return states[self.nodes[-1].name]

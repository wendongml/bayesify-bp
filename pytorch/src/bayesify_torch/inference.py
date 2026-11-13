"""Finite Gaussian belief updates, kept separate from message transport.

The reverse pass computes likelihood messages at one frozen belief snapshot.
Only after every proposed belief is validated do these functions mutate model
state.  This separation prevents an early layer update from changing the
snapshot used by a later layer's message.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from .geometry import MomentSignal, canonical_mfvi_update, tilted_update
from .graph import MomentDAG
from .layers import GaussianParameter


def named_gaussian_parameters(graph: MomentDAG) -> Iterator[tuple[str, GaussianParameter]]:
    """Yield names in the same ``node.parameter`` coordinates as graph messages."""
    for node in graph.nodes:
        layer = graph.layers[node.name]
        for local_name, child in layer.named_children():
            if isinstance(child, GaussianParameter):
                yield f"{node.name}.{local_name}", child


def apply_mfvi(
    graph: MomentDAG, messages: Mapping[str, MomentSignal], rho: float
) -> dict[str, tuple]:
    """Apply the canonical MFVI update to every persistent weight belief.

    For ``tau=1/v`` and ``h=m/v``, likelihood adjoints ``(g_m,g_v)`` give

    ``tau+ = (1-rho) tau + rho (tau0 - 2 g_v)``,
    ``h+   = (1-rho) h   + rho (h0 + g_m - 2 m g_v)``.

    This is not an Adam step and is not a unit increment of the variance.
    """
    parameters = dict(named_gaussian_parameters(graph))
    missing, extra = parameters.keys() - messages.keys(), messages.keys() - parameters.keys()
    if missing or extra:
        raise KeyError(f"parameter/message mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    proposed = {
        name: canonical_mfvi_update(parameter.moments(), messages[name], parameter.prior(), rho)
        for name, parameter in parameters.items()
    }
    for name, parameter in parameters.items():
        parameter.install(proposed[name])
    return proposed


def apply_global_ep_tilt(
    graph: MomentDAG, messages: Mapping[str, MomentSignal], rho: float = 1.0
) -> dict[str, tuple]:
    """Install one global tilted-moment correction.

    If ``s=m^2+v``, an EP signal ``(dm,ds)`` gives
    ``m+=m+rho*dm`` and ``v+=v+rho*ds-(m+)^2+m^2``.  Equivalently,
    ``v+=v+rho*c-rho^2*dm^2`` for ``c=ds-2m*dm``.  The quadratic term is
    essential.  This routine is the single-global-factor teaching update; a
    multi-factor EP implementation must additionally remove and replace sites.
    """
    parameters = dict(named_gaussian_parameters(graph))
    missing, extra = parameters.keys() - messages.keys(), messages.keys() - parameters.keys()
    if missing or extra:
        raise KeyError(f"parameter/message mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    proposed = {
        name: tilted_update(parameter.moments(), messages[name], rho)
        for name, parameter in parameters.items()
    }
    for name, parameter in parameters.items():
        parameter.install(proposed[name])
    return proposed

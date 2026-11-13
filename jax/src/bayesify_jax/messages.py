"""Forward snapshots and explicit reverse traversal of a shared computation DAG.

A Node stores layer type, parent IDs and *parameter IDs*. Reusing the same
parameter ID adds all its adjoints; it never creates an independent weight copy.
A diagonal Add discards cross-branch covariance. Shared-factor propagation in
covariance.py retains it. Duplicate(x) explicitly represents x+x with variance4v.
"""
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from .gaussian import Moments,add_signal,decode
from . import layers as L

@dataclass(frozen=True)
class Node:
    name:str
    kind:str
    parents:tuple
    parameters:tuple=()
    options:tuple=()


def local_map(node,statistics=None):
    kind=node.kind
    if kind=='dense':return L.dense
    if kind=='readout':return L.readout
    if kind=='correlated_readout':return lambda *args:L.correlated_readout(*args,statistics[node.name])
    if kind=='conv':return lambda x,w,b:L.convolution(x,w,b,*node.options)
    if kind=='conv0':return lambda x,w:L.convolution(x,w,Moments(jnp.zeros(w.mean.shape[-1],w.mean.dtype),jnp.zeros(w.mean.shape[-1],w.mean.dtype)),*node.options)
    if kind=='norm':
        m,v=statistics[node.name]
        return lambda x,g,b:L.normalization(x,g,b,m,v)
    return {'relu':L.relu,'maxpool':L.max_pool,'avgpool':L.avg_pool,'flatten':L.flatten,
            'add':L.add,'duplicate':L.duplicate,'concat':L.concatenate,'globalpool':L.global_pool,
            'shortcut':L.shortcut,'identity':lambda x:x}[kind]


def forward(nodes,beliefs,x,statistics=None):
    states={'input':x};records=[]
    for node in nodes:
        parents=tuple(states[n] for n in node.parents)+tuple(beliefs[n] for n in node.parameters)
        y=local_map(node,statistics)(*parents);states[node.name]=y
        records.append((node,parents,y))
    return states,records


def backward(records,signal,statistics=None,literal=False):
    pending={records[-1][0].name:signal};parameters={};activations={}
    for node,parents,y in reversed(records):
        child=pending[node.name];activations[node.name]=child;f=local_map(node,statistics)
        if not literal: messages=L.projected(f,parents,child)
        elif node.kind=='dense':messages=L.dense_backward(*parents,child)
        elif node.kind=='readout':messages=L.readout_backward(*parents,child)
        elif node.kind=='correlated_readout':raise ValueError('correlated readout uses projected local VJP, not a claimed independent literal derivative')
        elif node.kind=='conv':messages=L.convolution_backward(*parents,child,*node.options)
        elif node.kind=='conv0':
            w=parents[1];z=Moments(jnp.zeros(w.mean.shape[-1],w.mean.dtype),jnp.zeros(w.mean.shape[-1],w.mean.dtype))
            messages=L.convolution_backward(*parents,z,child,*node.options)[:2]
        elif node.kind=='relu':messages=L.relu_backward(*parents,child)
        elif node.kind=='maxpool':messages=L.max_pool_backward(*parents,child)
        elif node.kind=='avgpool':messages=L.avg_pool_backward(*parents,child)
        elif node.kind=='norm':messages=L.normalization_backward(*parents,child,*statistics[node.name])
        else:messages=L.linear_literal(f,parents,child)
        k=len(node.parents)
        for name,d in zip(node.parents,messages[:k]):pending[name]=add_signal(pending.get(name),d)
        for name,d in zip(node.parameters,messages[k:]):parameters[name]=add_signal(parameters.get(name),d)
    return pending['input'],activations,parameters


def point_forward(nodes,weights,x,statistics=None,training=False):
    states={'input':x};updated={}
    for node in nodes:
        xs=[states[n] for n in node.parents];ws=[weights[n] for n in node.parameters];kind=node.kind
        if kind in ('dense','readout','correlated_readout'):y=xs[0]@ws[0]+ws[1]
        elif kind=='conv':y=L.conv(xs[0],ws[0],*node.options)+ws[1]
        elif kind=='conv0':y=L.conv(xs[0],ws[0],*node.options)
        elif kind=='norm':
            if training:
                axes=tuple(range(xs[0].ndim-1));m=xs[0].mean(axes);v=xs[0].var(axes);updated[node.name]=(m,v*(xs[0].size/xs[0].shape[-1])/(xs[0].size/xs[0].shape[-1]-1))
            else:m,v=statistics[node.name]
            y=(xs[0]-m)*jax.lax.rsqrt(v+1e-5)*ws[0]+ws[1]
        elif kind=='relu':y=jax.nn.relu(xs[0])
        elif kind=='maxpool':
            from flax import nnx
            y=nnx.max_pool(xs[0],window_shape=(2,2),strides=(2,2),padding='VALID')
        else:
            qs=[Moments(z,jnp.zeros_like(z)) for z in xs];y=local_map(node,statistics)(*qs).mean
        states[node.name]=y
    return states[nodes[-1].name],updated


def frozen_statistics(nodes,beliefs,x,statistics):
    # Compute once per forward snapshot; do not mutate BN state during reverse.
    _,batch=point_forward(nodes,{n:q.mean for n,q in beliefs.items()},x,statistics,True)
    return jax.tree_util.tree_map(jax.lax.stop_gradient,batch)

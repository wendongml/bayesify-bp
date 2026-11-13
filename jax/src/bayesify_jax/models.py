"""Flax NNX models built from the same mathematical layer kernels.

No optimizer is hidden in a layer. The model owns Gaussian weight blocks,
fixed priors, and running normalization statistics. snapshot() returns the
explicit immutable belief tree used throughout a complete reverse pass.
"""
import math
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from .gaussian import Moments
from .messages import Node,forward,point_forward,local_map

def module_dict(values=None):
    """NNX containers: recent Flax requires explicit data fields.

    Flax 0.10 tabulation adds private attributes to nnx.Dict, so its old
    Mapping implementation is unsuitable during inspection. Plain dictionaries
    are native graph containers in that pinned release.
    """
    return nnx.Dict(values or {}) if hasattr(nnx, "data") else dict(values or {})

class GaussianParameter(nnx.Module):
    def __init__(self,mean,variance,prior_mean=None,prior_variance=1.):
        # Honor the declared model dtype even when the notebook enables x64
        # for its independent mathematical checks. Convolution requires both
        # operands to have the same dtype in every moment term.
        variance=jnp.asarray(variance,dtype=mean.dtype)
        prior_variance=jnp.asarray(prior_variance,dtype=mean.dtype)
        self.mean=nnx.Param(mean);self.log_variance=nnx.Param(jnp.log(jnp.broadcast_to(variance,mean.shape)))
        self.prior_mean=nnx.Variable(jnp.zeros_like(mean) if prior_mean is None else prior_mean)
        self.prior_variance=nnx.Variable(jnp.broadcast_to(prior_variance,mean.shape))
    def belief(self):return Moments(self.mean.value,jnp.exp(self.log_variance.value))
    def prior(self):return Moments(self.prior_mean.value,self.prior_variance.value)
    def install(self,q):
        self.mean.value=q.mean;self.log_variance.value=jnp.log(q.variance)

class MomentLayer(nnx.Module):
    def __init__(self,node,parameters):
        self.node=node;self.parameters=module_dict(parameters)
    def __call__(self,**parents):
        x=tuple(parents.values())
        # Real module call used by nnx.tabulate, with the declared mean network.
        ws={n:p.mean.value for n,p in self.parameters.items()}
        one=Node(self.node.name,self.node.kind,tuple(f'x{i}' for i in range(len(x))),self.node.parameters,self.node.options)
        qs=tuple(Moments(z,jnp.zeros_like(z)) for z in x)
        ps=tuple(Moments(ws[n],jnp.zeros_like(ws[n])) for n in one.parameters)
        if one.kind=='norm':
            return (x[0]-self.running_mean.value)*jax.lax.rsqrt(self.running_variance.value+1e-5)*ps[0].mean+ps[1].mean
        if one.kind=='maxpool':
            return nnx.max_pool(x[0],window_shape=(2,2),strides=(2,2),padding='VALID')
        return local_map(one)(*qs,*ps).mean
    def moments(self,*x):
        stat={self.node.name:(self.running_mean.value,self.running_variance.value)} if self.node.kind=='norm' else None
        return local_map(self.node,stat)(*x,*(self.parameters[n].belief() for n in self.node.parameters))

class MomentModel(nnx.Module):
    def __init__(self,nodes,parameters,statistics,name):
        self.nodes=tuple(nodes);self.parameters=module_dict(parameters);self.name=name;self.layers=module_dict()
        for node in nodes:
            layer=MomentLayer(node,{k:parameters[k] for k in node.parameters})
            if node.kind=='norm':layer.running_mean=nnx.Variable(statistics[node.name][0]);layer.running_variance=nnx.Variable(statistics[node.name][1])
            self.layers[node.name]=layer
    def __call__(self,x):
        states={'input':x}
        for node in self.nodes:states[node.name]=self.layers[node.name](**{f'parent{i}':states[k] for i,k in enumerate(node.parents)})
        return states[self.nodes[-1].name]
    def snapshot(self):return {n:p.belief() for n,p in self.parameters.items()}
    def priors(self):return {n:p.prior() for n,p in self.parameters.items()}
    def statistics(self):
        return {n:(l.running_mean.value,l.running_variance.value) for n,l in self.layers.items() if l.node.kind=='norm'}
    def install(self,beliefs,statistics=None):
        for n,q in beliefs.items():self.parameters[n].install(q)
        if statistics is not None:
            for n,(m,v) in statistics.items():self.layers[n].running_mean.value=m;self.layers[n].running_variance.value=v
    def forward_moments(self,x,literal=False):
        return forward(self.nodes,self.snapshot(),x,self.statistics())

class Builder:
    """Only architecture wiring/initialization; inference lives in explicit functions."""
    def __init__(self,seed=0,prior_variance=1e-3,initial_ratio=.1,dtype=jnp.float32,he=False,center_prior=False):
        self.rng=np.random.default_rng(seed);self.pv=prior_variance;self.ratio=initial_ratio;self.dtype=dtype;self.he=he;self.center_prior=center_prior
        self.nodes=[];self.parameters={};self.statistics={}
    def add(self,name,kind,*parents,shape=None,options=()):
        ids=()
        if shape is not None:
            if kind=='norm':m=jnp.ones(shape,self.dtype);bias=jnp.zeros(shape,self.dtype)
            else:
                fan=np.prod(shape[:-1]);bound=1/math.sqrt(fan)
                m=jnp.asarray(self.rng.uniform(-bound,bound,shape),dtype=self.dtype)
                bias=jnp.zeros((shape[-1],),dtype=self.dtype)
            if self.he and kind!='norm':m=jnp.asarray(self.rng.normal(0,math.sqrt(2/np.prod(shape[:-1])),shape),dtype=self.dtype)
            if kind=='conv0':m=jnp.asarray(self.rng.normal(0,math.sqrt(2/(np.prod(shape[:2])*shape[-1])),shape),dtype=self.dtype)
            ids=(name+'.weight',) if kind=='conv0' else (name+'.weight',name+'.bias')
            pv=self.pv
            for n,a in zip(ids,(m,bias)):
                self.parameters[n]=GaussianParameter(a,1e-5 if kind=='norm' else pv*self.ratio,prior_mean=a if self.center_prior else None,prior_variance=pv)
            if kind=='norm':self.statistics[name]=(jnp.zeros(shape,self.dtype),jnp.ones(shape,self.dtype))
        self.nodes.append(Node(name,kind,parents,ids,tuple(options)))
        return name
    def finish(self,name):return MomentModel(self.nodes,self.parameters,self.statistics,name)


def mlp(seed=0,inputs=784,width=400,classes=10,**kwargs):
    b=Builder(seed,**kwargs);x=b.add('flatten','flatten','input')
    for i,dim in enumerate((inputs,width),1):
        x=b.add(f'dense{i}','dense',x,shape=(dim,width));x=b.add(f'relu{i}','relu',x)
    b.add('readout','readout',x,shape=(width,classes));return b.finish('MLP')


def lenet(seed=0,classes=10,**kwargs):
    b=Builder(seed,he=True,**kwargs);x='input'
    for i,ci,co,pad in ((1,1,6,2),(2,6,16,0)):
        x=b.add(f'conv{i}','conv',x,shape=(5,5,ci,co),options=(1,pad));x=b.add(f'pool{i}','maxpool',x);x=b.add(f'relu{i}','relu',x)
    x=b.add('flatten','flatten',x)
    for i,ci,co in ((1,400,120),(2,120,84)):
        x=b.add(f'dense{i}','dense',x,shape=(ci,co));x=b.add(f'dense_relu{i}','relu',x)
    b.add('readout','readout',x,shape=(84,classes));return b.finish('LeNet')


def resnet20(seed=0,classes=10,**kwargs):
    b=Builder(seed,initial_ratio=.01,center_prior=False,he=True,**kwargs);x=b.add('stem_conv','conv0','input',shape=(3,3,3,16),options=(1,1))
    x=b.add('stem_bn','norm',x,shape=(16,));x=b.add('stem_relu','relu',x);ci=16
    for stage,co in enumerate((16,32,64),1):
        for block in range(1,4):
            prefix=f'stage{stage}_block{block}';stride=2 if stage>1 and block==1 else 1
            skip=x if stride==1 else b.add(prefix+'_shortcut','shortcut',x)
            h=b.add(prefix+'_conv1','conv0',x,shape=(3,3,ci,co),options=(stride,1));h=b.add(prefix+'_bn1','norm',h,shape=(co,));h=b.add(prefix+'_relu1','relu',h)
            h=b.add(prefix+'_conv2','conv0',h,shape=(3,3,co,co),options=(1,1));h=b.add(prefix+'_bn2','norm',h,shape=(co,))
            x=b.add(prefix+'_sum','add',skip,h);x=b.add(prefix+'_relu2','relu',x);ci=co
    x=b.add('global_pool','globalpool',x);b.add('classifier','readout',x,shape=(ci,classes));return b.finish('ResNet-20')


def densenet40(seed=0,classes=10,growth=12,**kwargs):
    b=Builder(seed,initial_ratio=.01,center_prior=False,he=True,**kwargs);ci=16;x=b.add('stem_conv','conv0','input',shape=(3,3,3,ci),options=(1,1))
    for stage in range(1,4):
        for block in range(1,13):
            p=f'dense{stage}_{block}';h=b.add(p+'_bn','norm',x,shape=(ci,));h=b.add(p+'_relu','relu',h)
            h=b.add(p+'_conv','conv0',h,shape=(3,3,ci,growth),options=(1,1));x=b.add(p+'_concat','concat',x,h);ci+=growth
        if stage<3:
            p=f'transition{stage}';x=b.add(p+'_bn','norm',x,shape=(ci,));x=b.add(p+'_relu','relu',x)
            x=b.add(p+'_conv','conv0',x,shape=(1,1,ci,ci),options=(1,0));x=b.add(p+'_pool','avgpool',x)
    x=b.add('final_bn','norm',x,shape=(ci,));x=b.add('final_relu','relu',x);x=b.add('global_pool','globalpool',x)
    b.add('classifier','readout',x,shape=(ci,classes));return b.finish('DenseNet-40')


def parameter_count(model):return sum(p.mean.value.size for p in model.parameters.values())


def correlated_mlp(model):
    """Rank-four hidden covariance, retaining all original parent adjoints."""
    from dataclasses import replace
    node=model.nodes[-1]
    node=replace(node,kind='correlated_readout',parents=('relu2','relu1','dense2'),parameters=node.parameters+('dense2.weight',))
    model.nodes=model.nodes[:-1]+(node,)
    model.layers[node.name]=MomentLayer(node,{n:model.parameters[n] for n in node.parameters})
    return model


def readout_basis(beliefs,rank=4):
    # Input-space right singular vectors of the R [out,in] matrix = left of JAX [in,out].
    U,_,_=jnp.linalg.svd(beliefs['dense2.weight'].mean,full_matrices=False)
    return jax.lax.stop_gradient(U[:,:rank])

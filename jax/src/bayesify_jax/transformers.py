"""GPT-2/RoBERTa with Gaussian Q/V LoRA, pure JAX and Flax NNX.

Frozen pretrained weights are separated from adapter beliefs. We support the
original independent Gaussian A+B family and conditional Gaussian A/point B.
Shared weights are reused across tokens/examples. No per-token weight resampling.
LayerNorm/attention moments use local directional calculus, not independent
attention-logit variances. Paired nonlinear carriers are an explicit alternative.
"""
from dataclasses import dataclass
import math
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from .gaussian import Moments
from .covariance import Shared,spherical_carrier
from .models import GaussianParameter,module_dict

@dataclass(frozen=True)
class Config:
    architecture:str='gpt2'
    width:int=768
    heads:int=12
    layers:int=12
    intermediate:int=3072
    vocabulary:int=50257
    positions:int=1024
    rank:int=4
    alpha:float=32.
    classes:int=2
    eps:float=1e-5
    pad_id:int=1
    family:str='ab'


def layer_norm(x,scale,bias,eps):
    a=x-x.mean(-1,keepdims=True)
    return a*jax.lax.rsqrt((a*a).mean(-1,keepdims=True)+eps)*scale+bias


def attention(q,k,v,mask,heads,causal):
    B,T,D=q.shape;d=D//heads
    reshape=lambda z:z.reshape(B,T,heads,d).transpose(0,2,1,3)
    q,k,v=map(reshape,(q,k,v));scores=jnp.einsum('bhtd,bhsd->bhts',q,k)/math.sqrt(d)
    valid=mask[:,None,None,:].astype(bool)
    if causal:valid=valid & jnp.tril(jnp.ones((T,T),dtype=bool))[None,None]
    P=jax.nn.softmax(jnp.where(valid,scores,jnp.finfo(scores.dtype).min),-1)
    return jnp.einsum('bhts,bhsd->bhtd',P,v).transpose(0,2,1,3).reshape(B,T,D)


def affine(x,base,name,adapters=None,scale=1.):
    weights=adapters if adapters is not None and name+'.weight' in adapters else base
    y=x@weights[name+'.weight']+weights[name+'.bias']
    if adapters is not None and name+'.A' in adapters:
        y=y+scale*(x@adapters[name+'.A'])@adapters[name+'.B']
    return y


def block(x,mask,base,adapters,cfg,index,linear_fn=None):
    p=f'block{index}.';scale=cfg.alpha/cfg.rank
    normal=adapters if cfg.family=='full' else base
    ln=lambda z,n:layer_norm(z,normal[p+n+'.weight'],normal[p+n+'.bias'],cfg.eps)
    linear=(lambda z,n:affine(z,base,p+n,adapters,scale)) if linear_fn is None else (lambda z,n:linear_fn(z,p+n))
    h=ln(x,'ln1') if cfg.architecture=='gpt2' else x
    q,k,v=[linear(h,n) for n in ('query','key','value')]
    h=linear(attention(q,k,v,mask,cfg.heads,cfg.architecture=='gpt2'),'attention_output')
    x=x+h
    if cfg.architecture=='roberta':x=ln(x,'ln1')
    h=ln(x,'ln2') if cfg.architecture=='gpt2' else x
    h=linear(jax.nn.gelu(linear(h,'ff1'),approximate=cfg.architecture=='gpt2'),'ff2')
    y=x+h
    return y if cfg.architecture=='gpt2' else ln(y,'ln2')


def embeddings(base,ids,mask,cfg):
    if cfg.architecture=='gpt2':positions=jnp.arange(ids.shape[1])[None,:]
    else:positions=jnp.cumsum(ids!=cfg.pad_id,axis=1)*(ids!=cfg.pad_id)+cfg.pad_id
    h=base['token_embedding'][ids]+base['position_embedding'][positions]
    if cfg.architecture=='roberta':
        h=h+base['type_embedding'][0]
        h=layer_norm(h,base['embedding_ln.weight'],base['embedding_ln.bias'],cfg.eps)
    return h


def readout(h,base,adapters,mask,cfg,target_ids):
    if cfg.architecture=='gpt2':
        weights=adapters if cfg.family=='full' else base
        h=layer_norm(h,weights['final_ln.weight'],weights['final_ln.bias'],cfg.eps)
        last=h[jnp.arange(h.shape[0]),jnp.maximum(mask.sum(-1)-1,0)]
        return last@weights['token_embedding'][target_ids].T
    h=jnp.tanh(affine(h[:,0],base,'classifier.dense',adapters,cfg.alpha/cfg.rank))
    return affine(h,base,'classifier.out',adapters,cfg.alpha/cfg.rank)


def point(base,adapters,ids,mask,cfg,target_ids):
    h=embeddings(adapters if cfg.family=='full' else base,ids,mask,cfg)
    for i in range(cfg.layers):h=block(h,mask,base,adapters,cfg,i)
    return readout(h,base,adapters,mask,cfg,target_ids)


def tangent(base,beliefs,ids,mask,cfg,target_ids,key,rank=4,return_states=False):
    """One shared direction identity throughout all blocks; local JVPs only."""
    keys=jax.random.split(key,len(beliefs));means={n:q.mean for n,q in beliefs.items()}
    U={n:jnp.sqrt(q.variance)*jax.random.rademacher(k,(rank,)+q.mean.shape,dtype=q.mean.dtype) for (n,q),k in zip(sorted(beliefs.items()),keys)}
    embed=lambda w:embeddings(w if cfg.family=='full' else base,ids,mask,cfg)
    h=embed(means);s=Shared(h,jax.vmap(lambda u:jax.jvp(embed,(means,),(u,))[1])(U));states=[s]
    for i in range(cfg.layers):
        f=lambda x,w:block(x,mask,base,w,cfg,i)
        mean=f(s.mean,means)
        factors=jax.vmap(lambda dh,dw:jax.jvp(f,(s.mean,means),(dh,dw))[1])(s.factors,U)
        s=Shared(mean,factors);states.append(s)
    f=lambda x,w:readout(x,base,w,mask,cfg,target_ids)
    out=Shared(f(s.mean,means),jax.vmap(lambda dh,dw:jax.jvp(f,(s.mean,means),(dh,dw))[1])(s.factors,U))
    return (out,states) if return_states else out


def scalar_moments(base,beliefs,ids,mask,cfg,target_ids):
    """Exact delta-method binary variance; no random directional sketch.

    J_n is refreshed at the current mean. v_n=sum_j J_nj² v_j. A per-example
    VJP avoids constructing a B×B activation Jacobian; transformer examples
    have no cross-example operators. Differentiating this map includes dJ/dm.
    """
    means={n:q.mean for n,q in beliefs.items()}
    one=lambda w,ids,mask:jnp.diff(point(base,w,ids[None],mask[None],cfg,target_ids),axis=-1)[0,0]
    m,J=jax.vmap(jax.value_and_grad(one),in_axes=(None,0,0))(means,ids,mask)
    v=sum((a*a*beliefs[n].variance).reshape(len(ids),-1).sum(-1) for n,a in J.items())
    return Moments(m,v),J


def scalar_likelihood(base,beliefs,ids,mask,cfg,target_ids,labels,method='ep'):
    from .heads import binary_gaussian
    # Forward scalar moments -> head adjoint -> pullback through the same
    # cached delta-method map. This is distinct from freezing J in Vadam.
    f=lambda q:scalar_moments(base,q,ids,mask,cfg,target_ids)[0]
    output,pull=jax.vjp(f,beliefs)
    value,adj=jax.value_and_grad(lambda q:binary_gaussian(q,labels,method))(output)
    return value,pull(adj)[0]


def nonlinear_carrier(base,beliefs,ids,mask,cfg,target_ids,key,pairs=2,stochastic=True):
    """Paired local nonlinear quadrature and conditional spherical recompression.

    Training carries a random conditional moment state. Setting stochastic=False
    keeps the local node mean only and is an ablation, not equivalent prediction.
    For independent A+B, node draws include the bilinear deltaA deltaB term.
    """
    mean={n:q.mean for n,q in beliefs.items()}
    if cfg.family=='full':raise ValueError('Use full-weight tangent or shared-network sampling; this carrier is LoRA-specific.')
    h=embeddings(base,ids,mask,cfg)
    for i in range(cfg.layers):
        key,k1,k2=jax.random.split(key,3);keys=jax.random.split(k1,len(beliefs))
        perturb={n:jnp.sqrt(q.variance)*jax.random.normal(k,(pairs,)+q.mean.shape,dtype=q.mean.dtype) for (n,q),k in zip(sorted(beliefs.items()),keys)}
        plus=jax.vmap(lambda d:block(h,mask,base,jax.tree_util.tree_map(lambda m,e:m+e,mean,d),cfg,i))(perturb)
        minus=jax.vmap(lambda d:block(h,mask,base,jax.tree_util.tree_map(lambda m,e:m-e,mean,d),cfg,i))(perturb)
        h=spherical_carrier(plus,minus,k2) if stochastic else ((plus+minus)/2).mean(0)
    return readout(h,base,mean,mask,cfg,target_ids)


def sampled(base,beliefs,ids,mask,cfg,target_ids,key,samples=32):
    keys=jax.random.split(key,len(beliefs))
    weights={n:q.mean+jnp.sqrt(q.variance)*jax.random.normal(k,(samples,)+q.mean.shape,dtype=q.mean.dtype) for (n,q),k in zip(sorted(beliefs.items()),keys)}
    logits=jax.vmap(lambda w:point(base,w,ids,mask,cfg,target_ids))(weights)
    return logits


def gaussian_prediction(state,order=32):
    if state.mean.shape[-1]==2:
        z,w=np.polynomial.hermite.hermgauss(order);m=state.mean[:,1]-state.mean[:,0]
        v=((state.factors[:,:,1]-state.factors[:,:,0])**2).mean(0)
        p=(jax.nn.sigmoid(m[:,None]+jnp.sqrt(2*v[:,None])*jnp.asarray(z))*jnp.asarray(w)/jnp.sqrt(jnp.pi)).sum(-1)
        return jnp.stack((1-p,p),-1)
    from .heads import full_covariance_probabilities
    C=jnp.einsum('rbi,rbj->bij',state.factors,state.factors)/state.rank
    return full_covariance_probabilities(state.mean,C)


def random_base(cfg,seed=0,dtype=jnp.float32):
    rng=np.random.default_rng(seed);D=cfg.width;F=cfg.intermediate;base={}
    arr=lambda shape:jnp.asarray(rng.normal(0,.02,shape),dtype=dtype)
    base['token_embedding']=arr((cfg.vocabulary,D));base['position_embedding']=arr((cfg.positions,D))
    for i in range(cfg.layers):
        for name,ci,co in [('query',D,D),('key',D,D),('value',D,D),('attention_output',D,D),('ff1',D,F),('ff2',F,D)]:
            base[f'block{i}.{name}.weight']=arr((ci,co));base[f'block{i}.{name}.bias']=jnp.zeros(co,dtype)
        for name in ['ln1','ln2']:base[f'block{i}.{name}.weight']=jnp.ones(D,dtype);base[f'block{i}.{name}.bias']=jnp.zeros(D,dtype)
    if cfg.architecture=='gpt2':base['final_ln.weight']=jnp.ones(D,dtype);base['final_ln.bias']=jnp.zeros(D,dtype)
    else:
        base['type_embedding']=arr((1,D));base['embedding_ln.weight']=jnp.ones(D,dtype);base['embedding_ln.bias']=jnp.zeros(D,dtype)
        for n,co in [('classifier.dense',D),('classifier.out',cfg.classes)]:base[n+'.weight']=arr((D,co));base[n+'.bias']=jnp.zeros(co,dtype)
    return base


def initial_beliefs(cfg,seed=41,variance=1e-6,dtype=jnp.float32,classifier_adapters=False):
    rng=np.random.default_rng(seed);result={}
    names=[(f'block{i}.{n}',cfg.width,cfg.width) for i in range(cfg.layers) for n in ['query','value']]
    if classifier_adapters:names += [('classifier.dense',cfg.width,cfg.width),('classifier.out',cfg.width,cfg.classes)]
    for name,ci,co in names:
        for factor,shape in [('A',(ci,cfg.rank)),('B',(cfg.rank,co))]:
            m=jnp.asarray(rng.normal(0,.02,shape) if factor=='A' else np.zeros(shape),dtype=dtype)
            v=variance if cfg.family=='ab' or factor=='A' else 0.
            result[name+'.'+factor]=Moments(m,jnp.full(shape,v,dtype))
    return result

class Transformer(nnx.Module):
    def __init__(self,cfg,base,beliefs,target_ids=(0,1)):
        self.config=cfg;self.base=module_dict({n:nnx.Variable(a) for n,a in base.items()})
        self.adapters=module_dict({n:GaussianParameter(q.mean,jnp.maximum(q.variance,1e-30)) for n,q in beliefs.items()})
        self.deterministic=tuple(n for n,q in beliefs.items() if np.all(np.asarray(q.variance)==0))
        self.target_ids=nnx.Variable(jnp.array(target_ids))
        self.blocks=module_dict({str(i):TransformerBlock(cfg,i,{n:v for n,v in self.base.items() if n.startswith(f'block{i}.')},{n:v for n,v in self.adapters.items() if n.startswith(f'block{i}.')}) for i in range(cfg.layers)})
    def base_arrays(self):return {n:v.value for n,v in self.base.items()}
    def snapshot(self):return {n:Moments(p.mean.value,jnp.zeros_like(p.mean.value) if n in self.deterministic else p.belief().variance) for n,p in self.adapters.items()}
    def __call__(self,ids,mask):
        base=self.base_arrays();weights={n:q.mean for n,q in self.snapshot().items()}
        h=embeddings(weights if self.config.family=='full' else base,ids,mask,self.config)
        for i in range(self.config.layers):h=self.blocks[str(i)](x=h,mask=mask)
        return readout(h,base,{n:q.mean for n,q in self.snapshot().items()},mask,self.config,self.target_ids.value)
    def forward_moments(self,ids,mask,key,rank=4):return tangent(self.base_arrays(),self.snapshot(),ids,mask,self.config,self.target_ids.value,key,rank)


def load_pretrained(architecture,cache_dir=None):
    """Read public safetensors directly; no PyTorch/TF model or execution."""
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file
    import json
    repo='openai-community/gpt2' if architecture=='gpt2' else 'FacebookAI/roberta-base'
    revision='607a30d783dfa663caf39e06633721c8d4cfcd7e' if architecture=='gpt2' else 'e2da8e2f811d1448a5b465c236feacd80ffbac7b'
    raw=load_file(hf_hub_download(repo,'model.safetensors',revision=revision,cache_dir=cache_dir))
    config=json.load(open(hf_hub_download(repo,'config.json',revision=revision,cache_dir=cache_dir)))
    cfg=Config(architecture=architecture,rank=4 if architecture=='gpt2' else 8,alpha=32 if architecture=='gpt2' else 8,
               vocabulary=config.get('vocab_size'),positions=config.get('n_positions',config.get('max_position_embeddings')),
               eps=config.get('layer_norm_epsilon',config.get('layer_norm_eps',1e-5)),classes=2 if architecture=='gpt2' else 150,family='a')
    base=random_base(cfg)
    def get(*names):
        for n in names:
            if n in raw:return jnp.asarray(raw[n])
        raise KeyError(names)
    if architecture=='gpt2':
        base['token_embedding']=get('wte.weight','transformer.wte.weight');base['position_embedding']=get('wpe.weight','transformer.wpe.weight')
        for i in range(cfg.layers):
            prefix=f'h.{i}.' if f'h.{i}.ln_1.weight' in raw else f'transformer.h.{i}.'
            W=get(prefix+'attn.c_attn.weight');b=get(prefix+'attn.c_attn.bias')
            for k,n in enumerate(['query','key','value']):base[f'block{i}.{n}.weight']=W[:,k*cfg.width:(k+1)*cfg.width];base[f'block{i}.{n}.bias']=b[k*cfg.width:(k+1)*cfg.width]
            for a,n in [('attn.c_proj','attention_output'),('mlp.c_fc','ff1'),('mlp.c_proj','ff2'),('ln_1','ln1'),('ln_2','ln2')]:
                for suffix in ['weight','bias']:base[f'block{i}.{n}.{suffix}']=get(prefix+a+'.'+suffix)
        base['final_ln.weight']=get('ln_f.weight','transformer.ln_f.weight');base['final_ln.bias']=get('ln_f.bias','transformer.ln_f.bias')
    else:
        prefix='roberta.' if 'roberta.embeddings.word_embeddings.weight' in raw else ''
        for src,dst in [('word_embeddings.weight','token_embedding'),('position_embeddings.weight','position_embedding'),('token_type_embeddings.weight','type_embedding'),('LayerNorm.weight','embedding_ln.weight'),('LayerNorm.bias','embedding_ln.bias')]:base[dst]=get(prefix+'embeddings.'+src)
        for i in range(cfg.layers):
            for a,n in [('attention.self.query','query'),('attention.self.key','key'),('attention.self.value','value'),('attention.output.dense','attention_output'),('attention.output.LayerNorm','ln1'),('intermediate.dense','ff1'),('output.dense','ff2'),('output.LayerNorm','ln2')]:
                for suffix in ['weight','bias']:
                    v=get(prefix+f'encoder.layer.{i}.'+a+'.'+suffix);base[f'block{i}.{n}.{suffix}']=v.T if v.ndim==2 else v
    return cfg,base


class TransformerBlock(nnx.Module):
    """An inspectable Flax block; weights are references to the parent model."""
    def __init__(self,cfg,index,base,adapters):
        self.config=cfg;self.index=index;self.base=module_dict(base);self.adapters=module_dict(adapters)
    def __call__(self,x,mask):
        return block(x,mask,{n:p.value for n,p in self.base.items()},
                     {n:p.mean.value for n,p in self.adapters.items()},self.config,self.index)


def moment_likelihood(base,beliefs,ids,mask,cfg,target_ids,key,labels,rank=4,method='mfvi'):
    """Explicit local reverse of the transformer moment graph.

    Each block receives one Shared activation boundary and its own Gaussian
    adapter blocks. Forward snapshots and noise IDs stay fixed during reverse.
    No full-network posterior oracle or numerical conditioning enters updates.
    """
    from .heads import categorical_objective
    from .gaussian import encode
    keys=jax.random.split(key,len(beliefs));noise={n:jax.random.rademacher(k,(rank,)+q.mean.shape,dtype=q.mean.dtype) for (n,q),k in zip(sorted(beliefs.items()),keys)}
    records=[]
    names=tuple(n for n in sorted(beliefs) if not n.startswith(('block','classifier.','final_ln.')))
    qs={n:beliefs[n] for n in names} if cfg.family=='full' else {}
    def embed_local(unused,q):
        means={n:p.mean for n,p in q.items()};U={n:jnp.sqrt(p.variance)*noise[n] for n,p in q.items()}
        f=lambda w:embeddings(w if cfg.family=='full' else base,ids,mask,cfg)
        h=f(means)
        factors=jax.vmap(lambda u:jax.jvp(f,(means,),(u,))[1])(U) if U else jnp.zeros((rank,)+h.shape,dtype=h.dtype)
        return Shared(h,factors)
    s=embed_local((),qs);records.append(((),qs,embed_local))
    for i in range(cfg.layers):
        names=tuple(n for n in sorted(beliefs) if n.startswith(f'block{i}.'));qs={n:beliefs[n] for n in names}
        def local(s,q,i=i,names=names):
            means={n:p.mean for n,p in q.items()};U={n:jnp.sqrt(p.variance)*noise[n] for n,p in q.items()}
            f=lambda h,w:block(h,mask,base,w,cfg,i)
            return Shared(f(s.mean,means),jax.vmap(lambda dh,dw:jax.jvp(f,(s.mean,means),(dh,dw))[1])(s.factors,U))
        out=local(s,qs);records.append((s,qs,local));s=out
    names=tuple(n for n in sorted(beliefs) if n.startswith('classifier.') or (cfg.family=='full' and n.startswith(('final_ln.','token_embedding'))));qs={n:beliefs[n] for n in names}
    def head_local(s,q):
        means={n:p.mean for n,p in q.items()};U={n:jnp.sqrt(p.variance)*noise[n] for n,p in q.items()}
        f=lambda h,w:readout(h,base,w,mask,cfg,target_ids)
        return Shared(f(s.mean,means),jax.vmap(lambda dh,dw:jax.jvp(f,(s.mean,means),(dh,dw))[1])(s.factors,U))
    out=head_local(s,qs);records.append((s,qs,head_local))
    objective=lambda z:categorical_objective(jnp.concatenate((z.mean[None]+z.factors,z.mean[None]-z.factors),0),labels,method)
    value,adj=jax.value_and_grad(objective)(out);grad={};boundary=[]
    for s,qs,f in reversed(records):
        _,pull=jax.vjp(f,s,qs);adj,g=pull(adj)
        for n,a in g.items():grad[n]=jax.tree_util.tree_map(lambda x,y:x+y,grad[n],a) if n in grad else a
        boundary.append(adj)
    return value,grad,list(reversed(boundary)),{n:encode(beliefs[n],g) for n,g in grad.items()}


def full_weight_beliefs(base,variance=1e-8):
    """Independent Gaussian coordinates for every embedding, norm, affine and head.

    The data IDs and masks remain deterministic. Repeated token IDs gather the
    SAME uncertain embedding rows; shared directions retain that covariance.
    The R full-RoBERTa path uses variance 1e-8 and prior variance 1e-6.
    """
    return {n:Moments(a,jnp.full_like(a,variance)) for n,a in base.items()}

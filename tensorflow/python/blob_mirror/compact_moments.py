"""Small-rank Gaussian propagation with merged affine means and covariance.

State(m,U) represents C=U^T U/K. This is a moment approximation, not K sampled
networks. A fixed sign sketch adds independent adapter uncertainty to U.
Covariance directions share their index through residual/attention branches.

The merged affine operation changes only execution. The optional SDPA cubature
changes the attention approximation: apply attention to jointly paired Q/K/V
nodes and project their output first two moments. Report it as an ablation.
"""
import math
import torch
import torch.nn.functional as F
from nonlinear_moments import State, root
from transfer_moments import TransferMoments


def activation(z,kind):
    if kind=='tanh':return z.tanh()
    if kind=='gpt_gelu':return .5*z*(1+torch.tanh(math.sqrt(2/math.pi)*(z+.044715*z.pow(3))))
    return z*torch.special.ndtr(z)

def activation_derivative(z,kind):
    if kind=='tanh':return 1-z.tanh().square()
    if kind=='gpt_gelu':
        c=math.sqrt(2/math.pi);t=torch.tanh(c*(z+.044715*z.pow(3)))
        return .5*(1+t)+.5*z*(1-t.square())*c*(1+3*.044715*z.square())
    return torch.special.ndtr(z)+z*torch.exp(-z.square()/2)/math.sqrt(2*math.pi)

def elementwise_moments(m,U,nodes,weights,signs,kind):
    v=U.square().mean(0)
    shape=(len(nodes),)+(1,)*m.ndim
    z=m[None]+root(2*v)[None]*nodes.reshape(shape)
    y=activation(z,kind);w=weights.reshape(shape)
    mean=(w*y).sum(0);variance=(w*(y-mean[None]).square()).sum(0)
    plus,minus=activation(m[None]+U,kind),activation(m[None]-U,kind)
    center=(plus+minus).mean(0)/2
    cols=(plus-minus)/2+((plus+minus)/2-center[None])*signs
    raw=cols.square().mean(0)
    threshold=(16*torch.finfo(m.dtype).eps*(1+m.abs())).square()
    active=(v>threshold)&(raw>torch.finfo(raw.dtype).tiny)
    ratio=torch.where(active,variance,torch.ones_like(variance))/torch.where(active,raw,torch.ones_like(raw))
    cols=torch.where(active[None],cols*root(ratio)[None],activation_derivative(m,kind)[None]*U)
    return torch.where(active,mean,activation(m,kind)),cols

fused_elementwise=torch.compile(elementwise_moments,dynamic=True,fullgraph=True)

def tangent_elementwise(m,U,kind):
    return activation(m,kind),activation_derivative(m,kind)[None]*U

def tangent_norm(m,U,weight,bias,eps):
    c=m-m.mean(-1,keepdim=True)
    s2=c.square().mean(-1,keepdim=True)+eps
    du=U-U.mean(-1,keepdim=True)
    projected=(du-c[None]*(du*c[None]).mean(-1,keepdim=True)/s2[None])*weight/s2.sqrt()[None]
    return F.layer_norm(m,weight.shape,weight,bias,eps),projected

def tangent_softmax(m,U,mask):
    p=(m+mask).softmax(-1)
    return p,p[None]*(U-(U*p[None]).sum(-1,keepdim=True))

fused_tangent_elementwise=torch.compile(tangent_elementwise,dynamic=True,fullgraph=True)
fused_tangent_norm=torch.compile(tangent_norm,dynamic=True,fullgraph=True)
fused_tangent_softmax=torch.compile(tangent_softmax,dynamic=True,fullgraph=True)


class CompactMoments(TransferMoments):
    def __init__(self, rank=2, seed=197, nonlinear=True, attention_rule='original', order=5, match_diagonal=False):
        super().__init__(rank=rank,seed=seed,nonlinear=nonlinear,order=order)
        self.attention_rule=attention_rule
        self.match_diagonal=match_diagonal
        self.affines={}
        self.probes={}
        self.quadrature={}
        self.task=None

    def forward_task(self,model,batch,task,target_ids=None,**kwargs):
        self.task=task
        return super().forward_task(model,batch,task,target_ids,**kwargs)

    def elementwise(self,x,fn,derivative):
        kind='tanh' if fn is torch.tanh else 'gpt_gelu' if self.task=='sst2' else 'gelu'
        if not self.nonlinear:
            return State(*fused_tangent_elementwise(x.m,x.U,kind))
        key=(x.m.device,x.m.dtype)
        if key not in self.quadrature:
            self.quadrature[key]=(torch.as_tensor(self.nodes,device=x.m.device,dtype=x.m.dtype),
                                  torch.as_tensor(self.weights,device=x.m.device,dtype=x.m.dtype))
        nodes,weights=self.quadrature[key]
        signs=self.noise(x.m.new_zeros(())).reshape((self.rank,)+(1,)*x.m.ndim)
        mean,U=fused_elementwise(x.m,x.U,nodes,weights,signs,kind)
        return State(mean,U)

    def norm(self,layer,x):
        if self.nonlinear:return super().norm(layer,x)
        return State(*fused_tangent_norm(x.m,x.U,layer.weight,layer.bias,layer.eps))

    def softmax(self,x,mask):
        if self.nonlinear:return super().softmax(x,mask)
        return State(*fused_tangent_softmax(x.m,x.U,mask))

    def noise(self,x):
        key=(self.counter,tuple(x.shape),x.device,x.dtype)
        if key not in self.probes:
            self.probes[key]=super().noise(x)
        else:
            self.counter+=1
        return self.probes[key]

    def linear(self,layer,x):
        # With frozen weights, W_eff=W+s B A is identical for every call.
        # Precompute it once. Account for this setup separately from inference.
        if layer not in self.affines:
            base=layer.base_layer if hasattr(layer,'base_layer') else layer
            W=base.weight.detach()
            if hasattr(layer,'lora_A'):
                W=W.clone()
                for name in layer.active_adapters:
                    W=W+(layer.lora_B[name].weight@layer.lora_A[name].weight)*layer.scaling[name]
            self.affines[layer]=(W,base.bias)
        W,bias=self.affines[layer]
        y=F.linear(torch.cat((x.m[None],x.U),0),W)
        m=y[0] if bias is None else y[0]+bias
        U=y[1:]
        diagonal=U.square().mean(0) if self.match_diagonal else None
        if hasattr(layer,'lora_A_rho'):
            for name in layer.active_adapters:
                rho=layer.lora_A_rho[name]
                sigma2=rho.pow(4) if layer.bayes_eps>=0 else F.softplus(rho).square()
                # E Var(BAx|x)=B diag(V_A (m^2+diag C)) B^T.
                v=F.linear(x.m.square()+(x.v if self.nonlinear else 0),sigma2)
                U=U+F.linear(self.noise(v)*root(v)[None],layer.lora_B[name].weight)*layer.scaling[name]
                if self.match_diagonal:
                    diagonal=diagonal+F.linear(v,layer.lora_B[name].weight.square())*layer.scaling[name]**2
        if self.match_diagonal:
            # Independent injected noise has no cross term with the incoming
            # uncertainty. Enforce its analytic marginal variance instead of
            # retaining a noisy finite-sketch cross term. Cross-covariance is
            # still approximated; matching the diagonal does not make it exact.
            raw=U.square().mean(0)
            safe=torch.where(raw>0,raw,torch.ones_like(raw))
            U=torch.where((raw>0)[None],U*root(diagonal/safe)[None],root(diagonal)[None].expand_as(U))
        return State(m,U)

    def attention(self,layer,x,mask):
        if self.attention_rule=='original':
            return super().attention(layer,x,mask)
        assert self.attention_rule=='cubature'
        q,k,v=[self.linear(getattr(layer,n),x) for n in ['query','key','value']]
        B,T,D=q.m.shape
        H,d=layer.num_attention_heads,layer.attention_head_size
        def paired(z):
            nodes=torch.cat((z.m[None]+z.U,z.m[None]-z.U),0)
            return nodes.reshape(2*self.rank*B,T,H,d).transpose(1,2)
        # All Q/K/V use the same node, retaining represented cross-covariance.
        masks=mask.expand(B,*mask.shape[1:])[None].expand(2*self.rank,B,*mask.shape[1:])
        masks=masks.reshape(2*self.rank*B,*mask.shape[1:])
        y=F.scaled_dot_product_attention(paired(q),paired(k),paired(v),
                                         attn_mask=masks,dropout_p=0.)
        y=y.transpose(1,2).reshape(2*self.rank,B,T,D)
        return self.pair_columns(y[:self.rank],y[self.rank:])


def ordinary_forward(model,batch,task,target_ids,affines=None,compile_mlp=False):
    """Point execution with fused SDPA, giving ordinary the same kernel benefit."""
    def linear(layer,x):
        if affines is None or not hasattr(layer,'lora_A'):return layer(x)
        if layer not in affines:
            base=layer.base_layer;W=base.weight.detach().clone()
            for name in layer.active_adapters:
                W=W+(layer.lora_B[name].weight@layer.lora_A[name].weight)*layer.scaling[name]
            affines[layer]=(W,base.bias)
        return F.linear(x,*affines[layer])
    net=model.base_model.get_base_model()
    ids=batch['input_ids'];B,T=ids.shape
    if task=='sst2':
        h=net.transformer.wte(ids)+net.transformer.wpe(torch.arange(T,device=ids.device))[None]
        layers=net.transformer.h
        valid=torch.ones(T,T,device=ids.device,dtype=torch.bool).tril()[None,None]&batch['attention_mask'][:,None,None,:].bool()
    else:
        h=net.roberta.embeddings(input_ids=ids)
        layers=net.roberta.encoder.layer
        valid=batch['attention_mask'][:,None,None,:].bool()
    for layer in layers:
        if task=='sst2':
            x=layer.ln_1(h);att=layer.attn;maps=att.c_attn
            H,d=att.num_heads,att.head_dim
        else:
            x=h;att=layer.attention.self;maps=att
            H,d=att.num_attention_heads,att.attention_head_size
        q,k,v=[linear(getattr(maps,n),x).reshape(B,T,H,d).transpose(1,2) for n in ['query','key','value']]
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=valid,dropout_p=0.)
        y=y.transpose(1,2).reshape(B,T,-1)
        if task=='sst2':
            h=h+att.c_proj(y)
            if compile_mlp:
                from fused_carrier import apply_mlp
                h=apply_mlp(layer,h[None],h.new_ones(1))
            else:
                h=h+layer.mlp.c_proj(layer.mlp.act(layer.mlp.c_fc(layer.ln_2(h))))
        else:
            h=layer.attention.output.LayerNorm(h+layer.attention.output.dense(y))
            h=layer.output.LayerNorm(h+layer.output.dense(layer.intermediate.intermediate_act_fn(layer.intermediate.dense(h))))
    if task=='sst2':
        h=net.transformer.ln_f(h)
        h=h[torch.arange(B,device=h.device),batch['attention_mask'].sum(-1)-1]
        if affines is not None:
            # Python-list advanced indexing creates a host-to-device index
            # copy. Cache the two unchanged readout rows before graph capture.
            if 'readout' not in affines:affines['readout']=net.lm_head.weight[target_ids].detach()
            return F.linear(h,affines['readout'])
        return F.linear(h,net.lm_head.weight[target_ids])
    return net.classifier(h)


def gaussian_probabilities(mean,variance):
    """The existing narrow/broad integration, without unused loss derivatives.

Evaluate both branches and select on GPU: no CPU synchronization or data-
dependent indexing. The broad branch gets variance one at unused narrow
coordinates, avoiding undefined 0/0. This preserves the integration rule.
"""
    from gaussian_logistic import rules
    nodes,weights,z,zw=rules(str(mean.device))
    m=mean.double();v=variance.double();broad=v>=4
    d=m[:,None]+v.sqrt()[:,None]*nodes
    n1=(d.sigmoid()*weights).sum(-1);n0=((-d).sigmoid()*weights).sum(-1)
    safe=torch.where(broad,v,torch.ones_like(v))[:,None]
    mu=m[:,None];sd=safe.sqrt()
    plus=torch.exp(-(z-mu).square()/(2*safe))/torch.sqrt(2*torch.pi*safe)
    minus=torch.exp(-(-z-mu).square()/(2*safe))/torch.sqrt(2*torch.pi*safe)
    tailplus=torch.exp(-mu+safe/2+torch.special.log_ndtr((mu-safe-40)/sd)).squeeze(-1)
    tailminus=torch.exp(mu+safe/2+torch.special.log_ndtr((-mu-safe-40)/sd)).squeeze(-1)
    correction=((minus-plus)*(-z).sigmoid()*zw).sum(-1)+tailminus-tailplus
    b1=torch.special.ndtr((mu/sd).squeeze(-1))+correction
    b0=torch.special.ndtr((-mu/sd).squeeze(-1))-correction
    return torch.stack((torch.where(broad,b0,n0),torch.where(broad,b1,n1)),-1).to(mean.dtype)


class InferenceGraphs:
    """Capture each observed input shape; replay the identical cached math."""
    def __init__(self,forward):
        self.forward=forward;self.graphs={};self.pool=torch.cuda.graph_pool_handle()

    def __call__(self,batch):
        key=tuple(batch['input_ids'].shape)
        if key not in self.graphs:
            static={k:v.clone() for k,v in batch.items()}
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):self.forward(static)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=self.pool):result=self.forward(static)
            self.graphs[key]=(graph,static,result)
        graph,static,result=self.graphs[key]
        for k,v in batch.items():static[k].copy_(v)
        graph.replay()
        return result

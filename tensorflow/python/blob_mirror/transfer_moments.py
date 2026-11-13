"""The same nonlinear moment operators applied to both transfer architectures."""
import math
from types import SimpleNamespace
import torch
from nonlinear_moments import Moments,State

class TransferMoments(Moments):
    def gpt_block(self,layer,x,mask):
        z=self.norm(layer.ln_1,x);attn=layer.attn
        qkv=attn.c_attn
        view=SimpleNamespace(query=qkv.query,key=qkv.key,value=qkv.value,
            num_attention_heads=attn.num_heads,attention_head_size=attn.head_dim)
        h=self.attention(view,z,mask)
        x=self.add(x,self.linear(attn.c_proj,h))
        h=self.linear(layer.mlp.c_fc,self.norm(layer.ln_2,x))
        c=math.sqrt(2/math.pi)
        def gelu(z):return .5*z*(1+torch.tanh(c*(z+.044715*z.pow(3))))
        def dg(z):
            t=torch.tanh(c*(z+.044715*z.pow(3)))
            return .5*(1+t)+.5*z*(1-t.square())*c*(1+3*.044715*z.square())
        h=self.elementwise(h,gelu,dg)
        return self.add(x,self.linear(layer.mlp.c_proj,h))

    def forward_task(self,model,batch,task,target_ids=None,checkpoint_blocks=False):
        self.counter=0;net=model.base_model.get_base_model();ids=batch['input_ids']
        if task=='sst2':
            g=net.transformer;positions=torch.arange(ids.shape[1],device=ids.device)
            m=g.wte(ids)+g.wpe(positions)[None];layers=g.h
            causal=torch.ones(ids.shape[1],ids.shape[1],device=ids.device,dtype=torch.bool).tril()
            valid=causal[None,None]&batch['attention_mask'][:,None,None,:].bool()
            mask=m.new_zeros(valid.shape).masked_fill(~valid,torch.finfo(m.dtype).min)
        else:
            m=net.roberta.embeddings(input_ids=ids);layers=net.roberta.encoder.layer
            mask=(1-batch['attention_mask'][:,None,None,:].to(m.dtype))*torch.finfo(m.dtype).min
        x=State(m,torch.zeros((self.rank,)+tuple(m.shape),device=m.device,dtype=m.dtype))
        for layer in layers:
            fn=self.gpt_block if task=='sst2' else self.block
            if checkpoint_blocks:
                from torch.utils.checkpoint import checkpoint
                start=self.counter
                def apply(mean,U,layer=layer,fn=fn,start=start):
                    self.counter=start;z=fn(layer,State(mean,U),mask);return z.m,z.U
                mean,U=checkpoint(apply,x.m,x.U,use_reentrant=False,preserve_rng_state=False);x=State(mean,U)
            else:x=fn(layer,x,mask)
        if task=='sst2':
            x=self.norm(net.transformer.ln_f,x);last=batch['attention_mask'].sum(-1)-1
            row=torch.arange(len(last),device=last.device)
            x=State(x.m[row,last],x.U[:,row,last])
            w=net.lm_head.weight[target_ids[1]]-net.lm_head.weight[target_ids[0]]
            return x.m@w,(x.U@w).square().mean(0)
        x=State(x.m[:,0],x.U[:,:,0]);x=self.tanh(self.linear(net.classifier.dense,x))
        return self.linear(net.classifier.out_proj,x)

def categorical_expectation(state,labels=None):
    # Positive symmetric cubature weights. Average probabilities for prediction;
    # average log probabilities for the VI expected likelihood. Never swap them.
    z=torch.cat((state.m[None]+state.U,state.m[None]-state.U),0)
    logp=z.log_softmax(-1)
    if labels is None:return logp.exp().mean(0)
    return -logp[:,torch.arange(len(labels),device=labels.device),labels].mean()

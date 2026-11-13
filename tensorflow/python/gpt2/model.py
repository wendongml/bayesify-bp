"""GPT-2 with Gaussian Q/V LoRA blocks. No backbone weights are trained.

Each draw is one complete adapter network, shared across all examples/tokens.
The frozen LM head evaluates only the two declared sentiment token rows.
"""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Model, AutoTokenizer


class GaussianWeight(nn.Module):
    def __init__(self, shape, std, variance):
        super().__init__()
        self.mean = nn.Parameter(torch.randn(shape) * std)
        self.log_variance = nn.Parameter(torch.full(shape, variance).log())


class QVLoRA(nn.Module):
    def __init__(self, base, rank=4, alpha=32, variance=1e-6):
        super().__init__(); self.base=base; self.scale=alpha/rank
        d=base.weight.shape[0]
        self.blocks=nn.ModuleList([GaussianWeight((d,rank),.02,variance),
            GaussianWeight((rank,d),0.,variance), GaussianWeight((d,rank),.02,variance),
            GaussianWeight((rank,d),0.,variance)])
        self.current=None

    def forward(self,x):
        a,b,c,d=self.current if self.current is not None else [p.mean for p in self.blocks]
        q=(x@a)@b; v=(x@c)@d
        return self.base(x)+self.scale*torch.cat((q,torch.zeros_like(q),v),dim=-1)


class GPT2LoRA(nn.Module):
    def __init__(self, path, seed=41, variance=1e-6, device='cuda'):
        super().__init__(); torch.manual_seed(seed)
        self.backbone=GPT2Model.from_pretrained(path,attn_implementation='eager')
        for p in self.backbone.parameters(): p.requires_grad_(False)
        self.backbone.eval()  # Dropout is disabled for every method, not weight uncertainty.
        for block in self.backbone.h:
            block.attn.c_attn=QVLoRA(block.attn.c_attn,variance=variance)
        tok=AutoTokenizer.from_pretrained(path)
        neg,pos=tok.encode(' negative'),tok.encode(' positive')
        assert len(neg)==len(pos)==1
        self.register_buffer('label_ids',torch.tensor([neg[0],pos[0]]))
        self.to(device)

    @property
    def layers(self):
        return [p for b in self.backbone.h for p in b.attn.c_attn.blocks]

    def belief(self):
        return [p.mean for p in self.layers],[p.log_variance.exp() for p in self.layers]

    def forward(self,ids,mask,weights=None):
        if weights is None: weights=[p.mean for p in self.layers]
        for i,block in enumerate(self.backbone.h): block.attn.c_attn.current=weights[4*i:4*i+4]
        h=self.backbone(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
        last=h[torch.arange(len(ids),device=ids.device),mask.sum(-1)-1]
        return F.linear(last,self.backbone.wte.weight[self.label_ids])

    def belief_state(self):
        return [{k:v.detach().cpu() for k,v in p.state_dict().items()} for p in self.layers]

    def restore_belief(self,state):
        for p,s in zip(self.layers,state): p.load_state_dict(s)

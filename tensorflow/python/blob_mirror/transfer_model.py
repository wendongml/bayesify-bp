"""External BLoB on the tutorial's architectures, with explicit adaptation scope.

Only GPT-2's fused QKV storage is split: its point function is unchanged. This
lets the original BLoB Linear code apply to query/value exactly as in RoBERTa.
The Bayesian family is the source random A / deterministic B, NOT our former
random A+B or CLINC full-weight family. No algorithm is called EP here.
"""
import math
import torch
from torch import nn
from transformers import GPT2LMHeadModel,RobertaForSequenceClassification
from transformers.pytorch_utils import Conv1D
from peft import LoraConfig,get_peft_model
from replicate import arguments,Local,BLoB,MLE

class SplitQKV(nn.Module):
    def __init__(self,layer):
        super().__init__();d=layer.in_features
        for i,name in enumerate(['query','key','value']):
            m=nn.Linear(d,d,device=layer.weight.device,dtype=layer.weight.dtype)
            with torch.no_grad():m.weight.copy_(layer.weight[i*d:(i+1)*d]);m.bias.copy_(layer.bias[i*d:(i+1)*d])
            setattr(self,name,m)
    def forward(self,x):return torch.cat([self.query(x),self.key(x),self.value(x)],-1)

def convert_conv(module):
    for name,child in list(module.named_children()):
        if isinstance(child,Conv1D):
            linear=nn.Linear(child.weight.shape[0],child.weight.shape[1],device=child.weight.device,dtype=child.weight.dtype)
            with torch.no_grad():linear.weight.copy_(child.weight.T);linear.bias.copy_(child.bias)
            setattr(module,name,linear)
        else:convert_conv(child)

def make_model(task,method,seed,n,epochs,full_head=False,mean_lr=None,tutorial_init=False,point_decay=.01):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    if task=='sst2':
        base=GPT2LMHeadModel.from_pretrained('gpt2',revision='607a30d783dfa663caf39e06633721c8d4cfcd7e',attn_implementation='eager')
        base.config.use_cache=False;convert_conv(base)
        for block in base.transformer.h:block.attn.c_attn=SplitQKV(block.attn.c_attn)
        pc=LoraConfig(r=4,lora_alpha=32,lora_dropout=0.,target_modules=['query','value'])
    else:
        base=RobertaForSequenceClassification.from_pretrained('roberta-base',revision='e2da8e2f811d1448a5b465c236feacd80ffbac7b',num_labels=150,attn_implementation='eager')
        targets=['query','value'] if full_head else ['query','value','classifier.dense','classifier.out_proj']
        pc=LoraConfig(r=8,lora_alpha=8,lora_dropout=0.,target_modules=targets)
    peft=get_peft_model(base,pc)
    if full_head:
        assert task=='clinc'
        # A new 150-class task needs a trainable classifier. Its parameters are
        # point estimates in BOTH matched methods; Bayesian uncertainty remains
        # in the backbone adapters. This is not full-weight Bayesian fine tuning.
        for p in peft.get_base_model().classifier.parameters():p.requires_grad_(True)
    # The established SST-2 adapter initializer. Pair it across all methods.
    if task=='sst2':
        with torch.no_grad():
            for name,p in peft.named_parameters():
                if 'lora_A.' in name:p.normal_(0.,.02)
                if 'lora_B.' in name:p.zero_()
    requested_means={n:p.detach().clone() for n,p in peft.named_parameters() if 'lora_A.' in n or 'lora_B.' in n} if tutorial_init else None
    args=arguments('map' if method=='ordinary' else 'blob',seed)
    args.num_samples=n;args.n_epochs=epochs;args.outdim=2 if task=='sst2' else 150
    args.max_train_steps=epochs*math.ceil(n/32);args.lr=mean_lr if mean_lr is not None else (3e-4 if task=='sst2' else 5e-5)
    args.opt_wd=point_decay if method=='ordinary' else 0.;args.warmup_ratio=.05 if task=='sst2' else .06
    model=(MLE if method=='ordinary' else BLoB)(peft,pc,args,Local()).cuda()
    if requested_means is not None:
        # The source's second PEFT construction reinitializes adapters. Restore
        # the requested tutorial initializer only in the explicitly named run.
        restored=set()
        with torch.no_grad():
            for name,p in peft.named_parameters():
                if name in requested_means:
                    p.copy_(requested_means[name]);restored.add(name)
        assert restored==set(requested_means),'PEFT changed adapter names: requested initializer was not restored'
    if full_head:
        # WrapperBase constructs another PEFT wrapper, which freezes non-adapter
        # parameters again. Restore the declared point-head scope AFTER it.
        for p in model.base_model.get_base_model().classifier.parameters():p.requires_grad_(True)
        # Keep the source likelihood AdamW / KL-SGD ordering. Give the point
        # classifier the SAME decoupled decay in both methods; its bias has none.
        named=dict(model.named_parameters());ids={id(p):name for name,p in named.items()}
        groups=[]
        for group in model.opt.param_groups:
            head=[p for p in group['params'] if 'classifier' in ids[id(p)]]
            rest=[p for p in group['params'] if 'classifier' not in ids[id(p)]]
            if rest:groups.append({**group,'params':rest})
            if head:groups.append({**group,'params':head,'weight_decay':0. if all('bias' in ids[id(p)] for p in head) else point_decay})
        model.opt.param_groups=groups
    # Dropout disabled in the historical configuration. eval() leaves gradients
    # enabled and the source's explicit blobsample switch remains true.
    model.eval();model.M=int(100*n**(math.pi/8)/32);model.i=1
    return model

def logits(model,batch,task,target_ids=None):
    net=model.base_model.get_base_model()
    if task=='clinc':return net(input_ids=batch['input_ids'],attention_mask=batch['attention_mask']).logits
    h=net.transformer(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False).last_hidden_state
    last=batch['attention_mask'].sum(-1)-1
    h=h[torch.arange(len(h),device=h.device),last]
    return torch.nn.functional.linear(h,net.lm_head.weight[target_ids])

def set_sample(model,active):
    for layer in model.modules():
        if hasattr(layer,'blobsample'):layer.blobsample=active

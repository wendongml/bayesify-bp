"""Fixed labelled splits and prompt construction; input tokens are deterministic."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

PREFIX='Review: great acting and direction\nSentiment: positive.\n\nReview: boring and predictable\nSentiment: negative.\n\n'

def load(root,limit=4096,seed=41,length=128):
    root=Path(root);cache=root/f'tokens_{length}_{limit}_{seed}.pt'
    receipt=root/f'data_receipt_{length}_{limit}_{seed}.json'
    if cache.exists():
        result=torch.load(cache,weights_only=False)
        if not receipt.exists():receipt.write_text(json.dumps({k:v for k,v in result.items() if k not in ('train','validation')},indent=2))
        return result
    tok=AutoTokenizer.from_pretrained(root/'gpt2')
    prefix=tok.encode(PREFIX);suffix=tok.encode('\nSentiment:')
    result={};rng=np.random.default_rng(seed)
    for split in ('train','validation'):
        ds=load_from_disk(str(root/f'sst2_{split}'))
        selected=np.arange(len(ds))
        if split=='train' and limit:
            y=np.asarray(ds['label']);selected=np.concatenate([rng.permutation(np.flatnonzero(y==k))[:limit//2] for k in (0,1)])
            rng.shuffle(selected)
        ids=np.full((len(selected),length),tok.eos_token_id,np.int64);mask=np.zeros_like(ids);labels=[];truncated=0
        for row,index in enumerate(selected):
            example=ds[int(index)];body=tok.encode('Review: '+example['sentence'])
            budget=length-len(prefix)-len(suffix);assert budget>0
            truncated+=len(body)>budget
            tokens=prefix+body[:budget]+suffix
            ids[row,:len(tokens)]=tokens;mask[row,:len(tokens)]=1;labels.append(example['label'])
        result[split]=(torch.tensor(ids),torch.tensor(mask),torch.tensor(labels))
        result[split+'_indices']=selected.tolist();result[split+'_truncated']=int(truncated)
    result['prompt']=dict(prefix=PREFIX,suffix='\nSentiment:',max_length=length,
                          truncation='review body right truncation; classification suffix always preserved')
    torch.save(result,cache)
    (root/'data_receipt.json').write_text(json.dumps({k:v for k,v in result.items() if k not in ('train','validation')},indent=2))
    receipt.write_text((root/'data_receipt.json').read_text())
    return result

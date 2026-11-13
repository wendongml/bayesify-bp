"""The existing tutorial's text inputs and split, independent of predictions."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parent/'results'
REVISIONS={'sst2':'8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb',
           'clinc':'155b9c710419136e17307b80d0a13e68cd46b4ec'}

def prepare(task):
    folder=ROOT/task;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'data.pt').exists():return torch.load(folder/'data.pt',weights_only=False)
    tokenizer=AutoTokenizer.from_pretrained('gpt2' if task=='sst2' else 'roberta-base',
        revision='607a30d783dfa663caf39e06633721c8d4cfcd7e' if task=='sst2' else 'e2da8e2f811d1448a5b465c236feacd80ffbac7b')
    if task=='sst2':
        tokenizer.pad_token=tokenizer.eos_token
        ds=load_dataset('stanfordnlp/sst2',revision=REVISIONS[task])
        prefix=tokenizer.encode('Review: great acting and direction\nSentiment: positive.\n\nReview: boring and predictable\nSentiment: negative.\n\n',add_special_tokens=False)
        suffix=tokenizer.encode('\nSentiment:',add_special_tokens=False)
        def encode(text):
            ids=torch.full((len(text),128),tokenizer.eos_token_id,dtype=torch.long);mask=torch.zeros_like(ids)
            for i,t in enumerate(text):
                body=tokenizer.encode('Review: '+t,add_special_tokens=False)[:128-len(prefix)-len(suffix)]
                row=prefix+body+suffix;ids[i,:len(row)]=torch.tensor(row);mask[i,:len(row)]=1
            return {'input_ids':ids,'attention_mask':mask}
        data={}
        for name,split in [('train','train'),('report','validation')]:
            data[name]=encode(ds[split]['sentence']);data[name]['labels']=torch.tensor(ds[split]['label'])
        assert len(data['train']['labels'])==67349 and len(data['report']['labels'])==872
        data['target_ids']=[tokenizer.encode(t,add_special_tokens=False)[0] for t in [' negative',' positive']]
        assert all(len(tokenizer.encode(t,add_special_tokens=False))==1 for t in [' negative',' positive'])
    else:
        ds=load_dataset('clinc/clinc_oos','plus',revision=REVISIONS[task])
        names=ds['train'].features['intent'].names;oos=names.index('oos');classes=[i for i in range(len(names)) if i!=oos]
        def encode(split,inside=True):
            rows=[r for r in ds[split] if (r['intent']!=oos)==inside]
            tokens=tokenizer([r['text'] for r in rows],padding='max_length',truncation=True,max_length=128,return_tensors='pt')
            return {**dict(tokens),'labels':torch.tensor([classes.index(r['intent']) if inside else -1 for r in rows])}
        alltrain=encode('train');groups={}
        for i,row in enumerate(alltrain['input_ids']):
            tokens=tuple(row[:alltrain['attention_mask'][i].sum()].tolist());groups.setdefault(tokens,[]).append(i)
        order=sorted(groups,key=lambda k:hashlib.sha256(('20261023:'+','.join(map(str,k))).encode()).digest())
        counts={i:0 for i in range(150)};stop=[];labels=alltrain['labels'].tolist()
        for key in order:
            group=groups[key]
            if any(counts[labels[i]]<10 for i in group):
                stop.extend(group)
                for i in group:counts[labels[i]]+=1
        stop=sorted(stop);fit=sorted(set(range(len(labels)))-set(stop))
        assert len(fit)==13500 and len(stop)==1500
        data={'train':{k:v[fit] for k,v in alltrain.items()},'stop':{k:v[stop] for k,v in alltrain.items()},
              'development':encode('validation'),'report':encode('test'),'ood':encode('test',False)}
        (folder/'split.json').write_text(json.dumps({'fit':fit,'stop':stop,'indexing':'zero-based original ID-only training rows'}))
    torch.save(data,folder/'data.pt')
    (folder/'data_receipt.json').write_text(json.dumps({'task':task,'dataset_revision':REVISIONS[task],
        'shapes':{n:list(a['input_ids'].shape) for n,a in data.items() if isinstance(a,dict)},
        'tokenizer_revision':tokenizer.init_kwargs.get('_commit_hash'),
        'sha256':hashlib.sha256((folder/'data.pt').read_bytes()).hexdigest()},indent=2))
    return data

def batches(data,batch_size=32,order=None):
    if order is None:order=torch.arange(len(data['labels']))
    for i in range(0,len(order),batch_size):
        b={k:v[order[i:i+batch_size]].cuda() for k,v in data.items()}
        width=int(b['attention_mask'].sum(-1).max())
        yield {k:v[:,:width] if v.ndim==2 else v for k,v in b.items()}

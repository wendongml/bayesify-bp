"""Native prediction from the original trained belief; same released kernel.

SST-2 uses the fixed third-epoch fit; CLINC uses the training-only stopping fit.
No historical scores, checkpoints or test labels enter training or selection.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from transfer_data import ROOT,prepare,batches
from transfer_model import make_model
from transfer_moments import categorical_expectation
from gaussian_logistic import expectations
from compact_moments import CompactMoments,ordinary_forward
p=argparse.ArgumentParser();p.add_argument('--task',required=True);a=p.parse_args()
task=a.task;data=prepare(task);epoch='epoch3' if task=='sst2' else 'best'
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
engine=CompactMoments(rank=1,attention_rule='cubature') if task=='sst2' else CompactMoments(rank=2,nonlinear=False)
for method in ['ordinary','hybrid']:
    folder=ROOT/task/f'v3_{method}_seed41'
    model=make_model(task,method,41,len(data['train']['labels']),3 if task=='sst2' else 6,
        full_head=task=='clinc',mean_lr=3e-4 if task=='sst2' else 1e-3,tutorial_init=task=='sst2',point_decay=.01)
    state=torch.load(folder/f'{epoch}.pt',map_location='cpu',weights_only=False)
    named=dict(model.named_parameters())
    with torch.no_grad():
        for name,w in state['weights'].items():named[name].copy_(w)
    affines={}
    def forward(b):
        if method=='ordinary':return ordinary_forward(model,b,task,data.get('target_ids'),affines).softmax(-1)
        z=engine.forward_task(model,b,task,data.get('target_ids'))
        if task=='clinc':return categorical_expectation(z)
        _,_,_,p0,p1=expectations(*z,b['labels']);return torch.stack((p0,p1),-1)
    with torch.inference_mode():
        start=time.perf_counter()
        for b in batches(data['report'],32):forward(b)
        torch.cuda.synchronize();setup=time.perf_counter()-start;times=[]
        for repeat in range(5):
            torch.cuda.synchronize();start=time.perf_counter();ps=[]
            for b in batches(data['report'],32):ps.append(forward(b).cpu())
            torch.cuda.synchronize();times.append(time.perf_counter()-start)
        probability=torch.cat(ps).double().numpy()
    np.savez_compressed(folder/'native_predictions.npz',probabilities=probability,labels=data['report']['labels'].numpy())
    result={'task':task,'method':method,'epoch':state['epoch'],'training_seconds':state['training_seconds'],
            'inference_seconds':float(np.median(times)),'inference_repeats':times,'setup_seconds':setup,
            'hardware':torch.cuda.get_device_name(),'checkpoint':epoch,'mode':'point' if method=='ordinary' else 'conditional moments'}
    (folder/'native_prediction.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
    del model
    torch.cuda.empty_cache()

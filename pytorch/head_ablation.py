from pathlib import Path
import numpy as np,torch,json,time
from bayesify_torch.architectures import resnet20
from bayesify_torch.inference import named_gaussian_parameters
from bayesify_torch.correlation import shared_forward,root_objective,RootMoments
import argparse
p=argparse.ArgumentParser(description='Fixed-belief readout diagnostic; no training.')
p.add_argument('--work',type=Path,required=True,help='Folder with checkpoints, cifar_first.npz and results/learned_fidelity')
w=p.parse_args().work;torch.set_num_threads(2);device='cuda';dtype=torch.float64
# Only the K-1 dimensional Gaussian likelihood integral changes. The learned
# weight belief, directions, hidden closure, and local reverse are frozen.
rows=[]
for method in ['mfvi','ep']:
 g=resnet20().network.to(device=device,dtype=dtype).eval();g.load_state_dict(torch.load(w/f'checkpoints/resnet20_{method}.pt',map_location='cpu',weights_only=False)['state'])
 for p in g.parameters():p.requires_grad_(False)
 data=np.load(w/'cifar_first.npz');x=torch.tensor(data['input'].transpose(0,3,1,2),device=device,dtype=dtype);y=torch.tensor(data['labels'],device=device)
 rng=np.random.default_rng(6200);noise={}
 for n,p in named_gaussian_parameters(g):
  sh=tuple(p.mean.shape);cs=(sh[2],sh[3],sh[1],sh[0]) if len(sh)==4 else sh[::-1] if len(sh)==2 else sh
  e=2*rng.integers(0,2,(64,)+cs,dtype=np.int8)-1;e=e.transpose(0,4,3,1,2) if len(sh)==4 else e.transpose(0,2,1) if len(sh)==2 else e
  noise[n]=torch.tensor(e,device=device,dtype=dtype)
 ref=np.load(w/f'results/learned_fidelity/{method}.npz');arrays={}
 for rank in [4,16,64]:
  passed=shared_forward(g,x,rank,noise={n:e[:rank] for n,e in noise.items()});z=passed.output
  for power in [0,10,14,18]:
   if power:
    e=torch.special.ndtri(torch.quasirandom.SobolEngine(9,scramble=True,seed=7301).draw_base2(power-1,dtype=dtype)).to(device);e=torch.cat((e,-e))
   def head(m,root):
    if power==0:return root_objective(RootMoments(m,root),y,method)
    A=root[:,:-1]-root[:,-1:];L=torch.linalg.qr(A.mT,mode='reduced').R.mT
    sign=torch.where(L.diagonal(dim1=-2,dim2=-1)>=0,1.,-1.);L=L*sign[:,None,:]
    delta=torch.einsum('bij,sj->sbi',L,e);a=m[:,:-1]-m[:,-1:]+delta
    logits=torch.cat((a,torch.zeros_like(a[...,:1])),-1);ll=logits.log_softmax(-1)[...,3]
    return ll.mean() if method=='mfvi' else torch.logsumexp(ll.flatten(),0)-np.log(len(ll))
   torch.cuda.synchronize();t=time.perf_counter();child,val=torch.func.grad_and_value(head,argnums=(0,1))(*z)
   weights,acts,signals=passed.backward(child);torch.cuda.synchronize();dt=time.perf_counter()-t
   d=signals['classifier.weight'];a=d.delta_mean.cpu().numpy();b=d.centered.cpu().numpy();key=f'r{rank}_head{power}';arrays[key+'_mean']=a;arrays[key+'_centered']=b
   rel=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
   row=dict(method=method,rank=rank,head_nodes=18 if power==0 else 2**power,mean_error=rel(a,ref['mc_mean']),variance_error=rel(b,ref['mc_centered']),reverse_seconds=dt,objective=float(val));rows.append(row);print(json.dumps(row),flush=True)
 np.savez_compressed(w/f'results/learned_fidelity/{method}_head_ablation.npz',**arrays)
(w/'results/learned_fidelity/head_ablation.json').write_text(json.dumps(rows,indent=2))

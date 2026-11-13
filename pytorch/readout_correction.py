"""Readout Price identity plus paired nonlinear residual; one Gaussian belief.

For h=log p_y (MFVI) or p_y (EP), d_m E[h]=E[dh/dw] and
 d_v E[h]=E[d²h/dw²]/2. Add/subtract the frozen linear-logit surrogate.
The normalizer and both derivatives are corrected BEFORE EP division.
Sampling uses r=(q+shifted q)/2 only as an integration proposal; q is unchanged.
"""
from pathlib import Path
import torch,numpy as np,json,time
from bayesify_torch.architectures import resnet20
from bayesify_torch.inference import named_gaussian_parameters
from learned_fidelity import readout_mc
import argparse
p=argparse.ArgumentParser(description='Fixed-belief readout diagnostic; no training.')
p.add_argument('--work',type=Path,required=True,help='Folder with checkpoints, cifar_first.npz and results/learned_fidelity')
w=p.parse_args().work;torch.set_num_threads(2);dtype=torch.float64;device='cuda';rows=[]
for method in ['mfvi','ep']:
 graph=resnet20().network.to(device=device,dtype=dtype).eval();graph.load_state_dict(torch.load(w/f'checkpoints/resnet20_{method}.pt',map_location='cpu',weights_only=False)['state'])
 for p in graph.parameters():p.requires_grad_(False)
 data=np.load(w/'cifar_first.npz');x=torch.tensor(data['input'].transpose(0,3,1,2),device=device,dtype=dtype)
 ps=dict(named_gaussian_parameters(graph));names=list(ps);sizes=[p.mean.numel() for p in ps.values()]
 m=torch.cat([p.mean.flatten() for p in ps.values()]);v=torch.cat([p.variance.flatten() for p in ps.values()]);sd=v.sqrt()
 def forward(w):
  values={f'layers.{n}.mean':a.reshape(ps[n].mean.shape) for n,a in zip(names,w.split(sizes))}
  states=torch.func.functional_call(graph,values,(x,),{'return_states':True})
  return states['global_pool'][0],states['classifier'][0]
 torch.cuda.synchronize();start=time.perf_counter()
 h0,z0=forward(m);J=torch.func.jacrev(lambda w:forward(w)[1])(m).detach();C=(J*v)@J.T
 other=[i for i in range(10) if i!=3];D=torch.eye(10,device=device,dtype=dtype)[other]-torch.eye(10,device=device,dtype=dtype)[3]
 a=D@z0;L=torch.linalg.cholesky(D@C@D.T);competitor=int(a.argmax());delta=L[competitor]
 e=torch.special.ndtri(torch.quasirandom.SobolEngine(9,scramble=True,seed=7301).draw_base2(18,dtype=dtype)).to(device)
 e=torch.cat((e,e+delta));ratio=2*torch.sigmoid(delta.square().sum()/2-e@delta)
 z=(a+e@L.T)@torch.eye(10,device=device,dtype=dtype)[other]
 val,gm,gv=readout_mc(h0.expand(len(z),-1),z,3,method)
 baseZ=(ratio*val).mean() if method=='ep' else z.new_tensor(1.)
 basegm=(ratio[:,None,None]*gm).mean(0);basegv=(ratio[:,None,None]*gv).mean(0)
 shift=sd*(J[other[competitor]]-J[3]);generator=torch.Generator(device=device).manual_seed(173)
 sums=[torch.zeros_like(baseZ),torch.zeros_like(basegm),torch.zeros_like(basegv)];count=0;out={};reference=np.load(w/f'results/learned_fidelity/{method}.npz')
 torch.cuda.synchronize();preparation=time.perf_counter()-start;t=time.perf_counter()
 for group in range(1024):
  # 16 independent normals; each produces +/- e and +/- e + shift.
  e=torch.randn(16,m.numel(),device=device,dtype=dtype,generator=generator);e=torch.stack((e,-e,e+shift,-e+shift),1).flatten(0,1)
  weight=2*torch.sigmoid(shift.square().sum()/2-e@shift);h,z=torch.func.vmap(forward)(m+sd*e)
  val,gm,gv=readout_mc(h,z,3,method);zs=z0+(e*sd)@J.T;vs,ms,ss=readout_mc(h0.expand(len(zs),-1),zs,3,method)
  if method=='ep':sums[0]+=(weight*(val-vs)).sum()
  sums[1]+=(weight[:,None,None]*(gm-ms)).sum(0);sums[2]+=(weight[:,None,None]*(gv-ss)).sum(0);count+=len(e)
  if group+1 in [4,16,64,256,1024]:
   Z=baseZ+sums[0]/count;dm=ps['classifier.weight'].variance*(basegm+sums[1]/count)/Z;dv=2*ps['classifier.weight'].variance**2*(basegv+sums[2]/count)/Z
   arrays=[dm.cpu().numpy(),dv.cpu().numpy()];rel=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b));torch.cuda.synchronize()
   row=dict(method=method,networks=count,mean_error=rel(arrays[0],reference['mc_mean']),variance_error=rel(arrays[1],reference['mc_centered']),seconds=time.perf_counter()-t,preparation_seconds=preparation,total_seconds=preparation+time.perf_counter()-t)
   rows.append(row);print(json.dumps(row),flush=True);out[f's{count}_mean'],out[f's{count}_centered']=arrays
   np.savez_compressed(w/f'results/learned_fidelity/{method}_price_residual.npz',**out)
   (w/'results/learned_fidelity/price_residual.json').write_text(json.dumps(rows,indent=2))

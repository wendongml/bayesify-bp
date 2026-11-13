"""Learned ResNet belief: native MC ground truth, approximation quality and cost.

All methods use the SAME weights, input and EP/MFVI objective. No optimization
occurs. Independent batches of antithetic complete-network draws give MCSEs;
Price's identity estimates the readout mean/variance adjoints without score
function noise. The comparison ranks approximations, not programming languages.
"""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
import torch
from torch.func import functional_call,vmap
from bayesify_torch.architectures import resnet20
from bayesify_torch.inference import named_gaussian_parameters
from bayesify_torch.correlation import shared_likelihood,shared_forward
from bayesify_torch.training import moment_likelihood_gradients
from bayesify_torch.geometry import DiagonalMoments,EuclideanAdjoint,encode_diagonal
from bayesify_torch import data


def readout_mc(features,logits,label,method):
    # d_m E[f]=E[df/dW], d_v E[f]=E[d²f/dW²]/2 (Gaussian Price identity).
    p=logits.softmax(-1);r=-p.clone();r[:,label]+=1;H=-p*(1-p)
    if method=='mfvi':value=logits.log_softmax(-1)[:,label];scale=torch.ones_like(value)
    else:value=p[:,label];scale=value;H=H+r.square()
    a=scale[:,None,None];h=features[:,None,:]
    return value,a*r[:,:,None]*h,.5*a*H[:,:,None]*h.square()


def run(checkpoint,method,output='results/learned_fidelity',batches=1024,draws=64,data_root='data',input_file=None):
    torch.set_num_threads(2);device='cuda' if torch.cuda.is_available() else 'cpu'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    dtype=torch.float32 if device=='cuda' else torch.float64
    graph=resnet20().network.to(device=device,dtype=dtype).eval();graph.load_state_dict(saved.get('state',saved.get('model')))
    for p in graph.parameters():p.requires_grad_(False)
    params=dict(named_gaussian_parameters(graph));q={n:p.moments() for n,p in params.items()}
    if input_file:
        with np.load(input_file) as a:x,y=a['input'],a['labels']
    else:x,y=data.cifar10(data_root)['test']
    x=torch.from_numpy(x[:1].transpose(0,3,1,2)).to(device=device,dtype=dtype);label=int(y[0])
    last=graph.nodes[-1];head=last.name+'.weight';hidden=last.parents[0]
    def one(weights):
        states=functional_call(graph,{f'layers.{n}.mean':w for n,w in weights.items()},(x,),{'return_states':True})
        return states[hidden][0],states[last.name][0]
    def sync():
        if device=='cuda':torch.cuda.synchronize()
    rng=torch.Generator(device=device).manual_seed(9000)
    values=[];adjoints=[];logits=[];start=time.perf_counter()
    with torch.no_grad():
        for batch in range(batches):
            weights={}
            for n,p in params.items():
                e=torch.randn((draws//2,)+p.mean.shape,dtype=p.mean.dtype,device=device,generator=rng)*p.variance.sqrt()
                weights[n]=torch.cat((p.mean+e,p.mean-e))
            h,z=vmap(one)(weights);value,gm,gv=readout_mc(h,z,label,method)
            values.append(float(value.mean()));adjoints.append((gm.mean(0).cpu().numpy(),gv.mean(0).cpu().numpy()))
            logits.append(z.cpu().numpy())
            if batch%128==0:print(json.dumps({'method':method,'MC_batches':batch,'total':batches}),flush=True)
    sync();mc_seconds=time.perf_counter()-start
    A=np.asarray(adjoints,dtype=np.float64);Z=np.mean(values) if method=='ep' else 1.;g=A.mean(0)/Z
    p=q[head];mean,variance=p.mean.detach().cpu().numpy(),p.variance.detach().cpu().numpy()
    signal=np.stack((variance*g[0],2*variance**2*g[1]))
    influence=A/Z-(np.asarray(values)/Z-1)[:,None,None,None]*g[None] if method=='ep' else A
    se=np.std(influence,axis=0,ddof=1)/np.sqrt(batches)
    se=np.stack((variance*se[0],2*variance**2*se[1]))
    z=np.asarray(logits,dtype=np.float64);flat=z.reshape(-1,z.shape[-1]);mu=flat.mean(0);C=np.cov(flat,rowvar=False,bias=True)
    output_mean_se=z.mean(1).std(0,ddof=1)/np.sqrt(batches)
    output_var_se=((z*z).mean(1)-2*mu*z.mean(1)).std(0,ddof=1)/np.sqrt(batches)
    arrays=dict(mc_output_mean=mu,mc_covariance=C,mc_mean=signal[0],mc_centered=signal[1],
                mcse_mean=se[0],mcse_centered=se[1],mcse_output_mean=output_mean_se,mcse_output_variance=output_var_se)
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/(method+'_mc.npz'),**arrays)
    norm=lambda a:float(np.linalg.norm(np.asarray(a,dtype=np.float64)))
    rel=lambda a,b:norm(a-b)/max(norm(b),1e-300)
    # Nested directions make each increase in rank add information rather than
    # changing all preceding probes. Reuse exactly these arrays for cross-framework checks.
    rng=np.random.default_rng(6200);noise={}
    for n,p in params.items():
        shape=tuple(p.mean.shape)
        common_shape=(shape[2],shape[3],shape[1],shape[0]) if len(shape)==4 else shape[::-1] if len(shape)==2 else shape
        e=(2*rng.integers(0,2,(64,)+common_shape,dtype=np.int8)-1)
        e=e.transpose(0,4,3,1,2) if len(shape)==4 else e.transpose(0,2,1) if len(shape)==2 else e
        noise[n]=torch.as_tensor(e,device=device,dtype=dtype)
    rows=[]
    for rank in (0,1,4,16,64):
        def calculate():
            if rank:
                value,weights,_,signals,passed=shared_likelihood(graph,x,torch.tensor([label],device=device),rank,
                    noise={n:e[:rank] for n,e in noise.items()},method=method)
                return passed.output,signals[head]
            result=moment_likelihood_gradients(graph,x,torch.tensor([label],device=device),method,full_readout=True)
            z=graph.forward_moments(DiagonalMoments(x,torch.zeros_like(x)),full_readout=True).output
            return z,result.messages.parameters[head]
        calculate();times=[]
        for _ in range(3):
            sync();start=time.perf_counter();out,d=calculate();sync();times.append(time.perf_counter()-start)
        key='diagonal' if rank==0 else f'rank{rank}'
        arrays[key+'_output_mean']=out.mean[0].detach().cpu().numpy();arrays[key+'_covariance']=out.covariance[0].detach().cpu().numpy()
        arrays[key+'_mean']=d.delta_mean.detach().cpu().numpy();arrays[key+'_centered']=d.centered.detach().cpu().numpy()
        row={'approximation':key,'rank':rank,'forward_backward_seconds':float(np.median(times)),
             'mean_relative_error':rel(arrays[key+'_output_mean'],mu),'covariance_relative_error':rel(arrays[key+'_covariance'],C),
             'mean_signal_relative_error':rel(arrays[key+'_mean'],signal[0]),'centered_signal_relative_error':rel(arrays[key+'_centered'],signal[1])}
        rows.append(row);print(json.dumps({'method':method,**row}),flush=True)
    np.savez_compressed(folder/(method+'.npz'),**arrays)
    receipt=dict(method=method,checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        networks=batches*draws,independent_batches=batches,MC_seconds=mc_seconds,test_index=0,
        hardware=torch.cuda.get_device_name() if device=='cuda' else 'CPU',arithmetic_dtype=str(dtype),
        reference='complete shared-weight networks; antithetic pairs; Price readout adjoints',
        MC_relative_standard_error={'output_mean':norm(output_mean_se)/max(norm(mu),1e-300),
          'output_variance':norm(output_var_se)/max(norm(np.diag(C)),1e-300),
          'mean_signal':norm(se[0])/max(norm(signal[0]),1e-300),'centered_signal':norm(se[1])/max(norm(signal[1]),1e-300)},rows=rows)
    (folder/(method+'.json')).write_text(json.dumps(receipt,indent=2))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--method',choices=['mfvi','ep'],required=True)
    p.add_argument('--batches',type=int,default=1024);p.add_argument('--output',default='results/learned_fidelity');p.add_argument('--data',default='data');p.add_argument('--input-file')
    a=p.parse_args();run(a.checkpoint,a.method,a.output,a.batches,data_root=a.data,input_file=a.input_file)

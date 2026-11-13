"""Fixed ResNet beliefs: shared-weight MC reference versus retained moment ranks.

No training, sample fitting or oracle gradients. All readout coordinates are
shown. Reference uncertainty comes from independent batches of full networks;
EP averages unnormalized likelihood derivatives BEFORE dividing by the average
likelihood. Centered variance sensitivity is 2 v^2 dPhi/dv, not a finite tilt.
"""
import os
os.environ.setdefault('XLA_FLAGS','--xla_disable_hlo_passes=constant_folding')
from pathlib import Path
import json,time,gc
import numpy as np
import jax,jax.numpy as j
from flax import serialization
from bayesify_jax import data,messages,covariance,inference,heads
from bayesify_jax.gaussian import encode
from bayesify_jax.training import Protocol,initialize
from bayesify_jax.plots import agreement_plot
from bayesify_jax.experiments import save
import matplotlib.pyplot as plt


def run(method,batches=8192,draws=64,reference_only=False):
    assert draws%2==0
    folder=Path('results')/f'resnet20_moment_{method}_seed41';cfg=Protocol(**json.loads((folder/'config.json').read_text()));model,_,state,_=initialize(cfg,45000)
    state=serialization.from_bytes({'state':state,'epoch':0},(folder/'best.msgpack').read_bytes())['state'];q=state['belief'];stats=state['statistics'];x,y=data.cifar10()['test'];x=j.asarray(x[:1]);label=int(y[0]);out=Path('results/learned_fidelity');out.mkdir(exist_ok=True)
    # Accumulate tiny variance-signal statistics in float64, outside training.
    norm=lambda a:float(np.linalg.norm(np.asarray(a,dtype=np.float64)))
    readout='classifier.weight';last=model.nodes[-1]
    assert last.kind=='readout' and last.parameters[0]==readout
    def batch(q,key):
        plus=inference.sample_weights(q,key,draws//2)
        # Each paired marginal is the same Gaussian belief. Independent
        # batches, rather than paired nodes, determine MC standard errors.
        weights={n:j.concatenate((a,2*q[n].mean-a),0) for n,a in plus.items()}
        def one(w):
            h=messages.point_forward(model.nodes[:-1],w,x,stats)[0][0]
            return h,h@w[last.parameters[0]]+w[last.parameters[1]]
        h,z=jax.vmap(one)(weights)
        value,g=heads.sampled_readout_adjoint(h,z,label,method)
        return value.mean(),jax.tree_util.tree_map(lambda a:a.mean(0),g),z
    fn=jax.jit(batch);start=time.perf_counter();jax.block_until_ready(fn(q,jax.random.key(71)));compile_s=time.perf_counter()-start
    gs=[];vs=[];zs=[];start=time.perf_counter()
    for i in range(batches):
        v,g,z=fn(q,jax.random.key(9000+i));gs.append(jax.device_get(g));vs.append(float(v));zs.append(np.asarray(z))
    mc_seconds=time.perf_counter()-start;Z=np.mean(vs) if method=='ep' else 1.
    print(method,'MC reference complete',{'networks':batches*draws,'seconds':mc_seconds},flush=True)
    avg=jax.tree_util.tree_map(lambda *a:j.asarray(np.mean(a,axis=0)/Z),*gs);signal=encode(q[readout],avg)
    logits=np.concatenate(zs).astype(np.float64);mu=logits.mean(0);C=np.cov(logits,rowvar=False,bias=True);readout='classifier.weight'
    arrays=dict(np.load(out/(method+'.npz'))) if reference_only else {}
    previous=json.loads((out/(method+'.json')).read_text()) if reference_only else None
    # Output means and weight-mean signals are different objects. The latter
    # already uses mc_mean below; do not overwrite the former with that signal.
    arrays.update(mc_output_mean=mu,mc_covariance=C)
    # Delta-method uncertainty for E[z] and E[z²]-E[z]². Each batch is
    # independent; antithetic nodes within a batch are not counted as such.
    batch_mean=np.asarray([np.asarray(z,dtype=np.float64).mean(0) for z in zs])
    batch_second=np.asarray([(np.asarray(z,dtype=np.float64)**2).mean(0) for z in zs])
    arrays['mcse_output_mean']=batch_mean.std(0,ddof=1)/np.sqrt(batches)
    arrays['mcse_output_variance']=(batch_second-2*mu*batch_mean).std(0,ddof=1)/np.sqrt(batches)
    # Delta-method MCSE for the ratio E[g]/E[f], with independent MC batches.
    errors={}
    for stat in ['mean','centered']:
        ref=np.asarray(getattr(signal,stat));arrays['mc_'+stat]=ref
        influence=[]
        for g,v in zip(gs,vs):
            a=jax.tree_util.tree_map(lambda u,mean:u/Z-mean*(v/Z-1) if method=='ep' else u,g,avg)
            influence.append(np.asarray(getattr(encode(q[readout],a),stat)))
        se=np.std(np.asarray(influence,dtype=np.float64),axis=0,ddof=1)/np.sqrt(batches);arrays['mcse_'+stat]=se
        errors[stat]=norm(se)/max(norm(ref),1e-300)
    rows=[]
    for rank in [1,4,16,64]:
        if reference_only:
            # Reuse the exact saved approximation and its measured timing;
            # only improve the independently sampled reference distribution.
            row=dict(next(r for r in previous['rows'] if r['rank']==rank))
            row['mean_relative_error']=norm(arrays[f'rank{rank}_output_mean'].astype(np.float64)-mu)/max(norm(mu),1e-300)
            row['covariance_relative_error']=norm(arrays[f'rank{rank}_covariance'].astype(np.float64)-C)/max(norm(C),1e-300)
            for stat in ['mean','centered']:
                row[stat+'_signal_relative_error']=norm(arrays[f'rank{rank}_{stat}'].astype(np.float64)-arrays['mc_'+stat])/max(norm(arrays['mc_'+stat]),1e-300)
            rows.append(row)
            continue
        def calc(q):
            _,g,_,_=covariance.shared_likelihood(model.nodes,q,x,j.array([label]),jax.random.key(7960),rank,stats,method)
            states,_=covariance.shared_records(model.nodes,q,x,jax.random.key(7960),rank,stats)
            return states[model.nodes[-1].name],g
        f=jax.jit(calc);begin=time.perf_counter();z,g=f(q);jax.block_until_ready((z,g));cs=time.perf_counter()-begin
        times=[]
        for _ in range(3):
            begin=time.perf_counter();z,g=f(q);jax.block_until_ready((z,g));times.append(time.perf_counter()-begin)
        d=encode(q[readout],g[readout]);arrays[f'rank{rank}_output_mean']=np.asarray(z.mean[0]);arrays[f'rank{rank}_covariance']=np.asarray(z.covariance[0])
        row={'rank':rank,'compile_seconds':cs,'forward_backward_seconds':float(np.median(times)),'mean_relative_error':norm(np.asarray(z.mean[0],dtype=np.float64)-mu)/max(norm(mu),1e-300),'covariance_relative_error':norm(np.asarray(z.covariance[0],dtype=np.float64)-C)/max(norm(C),1e-300)}
        for stat in ['mean','centered']:
            a=np.asarray(getattr(d,stat));ref=arrays['mc_'+stat];arrays[f'rank{rank}_{stat}']=a;row[stat+'_signal_relative_error']=norm(a.astype(np.float64)-ref)/max(norm(ref),1e-300)
        rows.append(row);print(method,row,flush=True);del f;jax.clear_caches();gc.collect()
    assert arrays['mc_output_mean'].shape==np.diag(arrays['mc_covariance']).shape
    assert arrays['mc_mean'].shape==arrays['mc_centered'].shape==q[readout].mean.shape
    np.savez_compressed(out/(method+'.npz'),**arrays)
    (out/(method+'.json')).write_text(json.dumps({'method':method,'networks':batches*draws,'independent_antithetic_pairs':batches*draws//2,'batches':batches,'MC_compile_seconds':compile_s,'MC_execution_seconds':mc_seconds,'readout_relative_MCSE':errors,'output_relative_MCSE':{'mean':norm(arrays['mcse_output_mean'])/max(norm(mu),1e-300),'variance':norm(arrays['mcse_output_variance'])/max(norm(np.diag(C)),1e-300)},'gradient_estimator':'Gaussian readout Price identity; complete-network draws','timing_scope':'MC forward plus readout adjoint; moment full forward/backward','test_index':0,'seed':cfg.seed,'checkpoint':'best validation NLL','rows':rows},indent=2))
    fig,ax=plt.subplots(2,2,figsize=(8,7))
    for i,rank in enumerate([1,64]):
        agreement_plot(mu,arrays[f'rank{rank}_output_mean'],title=f'{method.upper()}, rank {rank}',xlabel='MC logit mean',ylabel='moment logit mean',ax=ax[0,i])
        agreement_plot(np.diag(C),np.diag(arrays[f'rank{rank}_covariance']),title=f'{method.upper()}, rank {rank}',xlabel='MC logit variance',ylabel='moment logit variance',ax=ax[1,i])
    save(fig,out,method+'_forward');plt.close(fig)
    fig,ax=plt.subplots(2,2,figsize=(8,7))
    for i,rank in enumerate([1,64]):
        for k,stat in enumerate(['mean','centered']):agreement_plot(arrays['mc_'+stat],arrays[f'rank{rank}_{stat}'],title=f'{method.upper()}, rank {rank}',xlabel='MC '+('mean signal' if stat=='mean' else 'variance sensitivity'),ylabel='moment approximation',ax=ax[k,i])
    save(fig,out,method+'_backward');plt.close(fig)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-only',action='store_true',help='Keep saved moment results; refine only their MC reference.')
    parser.add_argument('--batches',type=int,default=8192)
    args=parser.parse_args()
    for method in ['mfvi','ep']:run(method,batches=args.batches,reference_only=args.reference_only);jax.clear_caches();gc.collect()

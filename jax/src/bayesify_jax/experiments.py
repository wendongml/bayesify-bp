"""Native JAX figures. Every plotted point is saved beside its generating code.

These algebra/approximation studies are separate from the full data benchmarks.
No R/PyTorch prediction files or rendered figures are loaded.
"""
from pathlib import Path
import time,json
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from .gaussian import Moments,FullMoments,encode,mfvi_update
from .gaussian_chain import candidate
from .gaussian_reference import chain_reference,ep_site,condition
from . import models,messages,heads,inference,covariance
from .plots import agreement_plot


def save(fig,folder,name):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    fig.tight_layout();fig.savefig(folder/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(folder/(name+'.pdf'),bbox_inches='tight')
    return fig


def exact_equivalence(folder='results/semantics'):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(41)
    from .gaussian_chain import frozen_mlp
    layers,weights,initial,H,y,R,checkpoint_activations=frozen_mlp()
    forward,activation,weight=candidate(layers,weights,initial,H,y,R)
    ref=chain_reference(*jax.tree_util.tree_map(np.asarray,(layers,weights,initial,H,y,R)))
    fig,axes=plt.subplots(2,2,figsize=(8,7));arrays={};rows=[]
    for i,(qs,label) in enumerate([(activation,'activation'),(weight,'weight')]):
        for k,stat in enumerate(['mean','variance']):
            actual=np.concatenate([np.asarray(q.mean if stat=='mean' else jnp.diag(q.covariance)) for q in qs])
            reference=np.concatenate(ref[label+'_mean'] if stat=='mean' else [np.diag(c) for c in ref[label+'_covariance']])
            agreement_plot(reference,actual,title=f'Gaussian {label} {stat}',xlabel='independent RTS / conditioning',ylabel='pseudo-gradient result',ax=axes[i,k])
            arrays[label+'_'+stat]=actual;arrays['reference_'+label+'_'+stat]=reference
            rows.append({'method':'exact Gaussian','quantity':label+' '+stat,'maximum_error':float(np.max(np.abs(actual-reference)))})
    save(fig,folder,'gaussian_equivalence');plt.close(fig)
    # Explicit cavities and site replacement after EVERY assimilation.
    dim=6;Hs=[rng.normal(size=(2,dim))*.3 for _ in range(6)];ys=[rng.normal(size=2) for _ in Hs];noise=np.eye(2)*.5
    fig,axes=plt.subplots(2,2,figsize=(8,7));traces={}
    for diagonal,label in [(False,'full EP'),(True,'diagonal EP')]:
        P=jnp.eye(dim);h=jnp.zeros(dim);Pr=np.eye(dim);hr=np.zeros(dim)
        sites=[(jnp.zeros((dim,dim)),jnp.zeros(dim)) for _ in Hs];refs=[(np.zeros((dim,dim)),np.zeros(dim)) for _ in Hs]
        av=[];rv=[]
        for sweep in range(3):
            for n,(J,t) in enumerate(zip(Hs,ys)):
                (P,h),sites[n],_=inference.ep_replace((P,h),sites[n],jnp.asarray(J),jnp.asarray(t),jnp.asarray(noise),diagonal)
                Pc,hc=Pr-refs[n][0],hr-refs[n][1];Cc=np.linalg.inv(Pc)
                sn,hn,_,_=ep_site(Cc@hc,Cc,J,t,noise,diagonal);refs[n]=(sn,hn);Pr,hr=Pc+sn,hc+hn
                C=np.linalg.inv(np.asarray(P));Cr=np.linalg.inv(Pr);av.append(np.r_[C@h,np.diag(C)]);rv.append(np.r_[Cr@hr,np.diag(Cr)])
        a=np.asarray(av);r=np.asarray(rv);traces[label.replace(' ','_')]=a;traces['reference_'+label.replace(' ','_')]=r
        row=0 if diagonal is False else 1
        for k,stat in enumerate(['mean','variance']):
            sl=slice(0,dim) if k==0 else slice(dim,None)
            agreement_plot(r[:,sl],a[:,sl],title=label+' '+stat,xlabel='independent explicit-site EP',ylabel='pseudo-gradient EP',ax=axes[row,k])
        rows.append({'method':label,'quantity':'all iterates','maximum_error':float(np.max(np.abs(a-r)))})
    save(fig,folder,'ep_equivalence');plt.close(fig)
    # CAVI comparison uses the learned candidate state, not the analytic optimum.
    X=jnp.asarray(rng.normal(size=(12,dim))*.3);target=jnp.asarray(rng.normal(size=12));prior=Moments(jnp.zeros(dim),jnp.ones(dim))
    q=Moments(jnp.asarray(rng.normal(size=dim)),jnp.ones(dim)*.8);Pr=np.eye(dim)+np.asarray(X.T@X);hr=np.asarray(X.T@target)
    mr=np.asarray(q.mean).copy();vr=np.ones(dim)*.8;av=[];rv=[];gaps=[]
    nodes=(messages.Node('output','dense',('input',),('weight','bias')),)
    def candidate_update(q):
        beliefs={'weight':Moments(q.mean[:,None],q.variance[:,None]),'bias':Moments(jnp.zeros(1),jnp.zeros(1))}
        states,records=messages.forward(nodes,beliefs,Moments(X,jnp.zeros_like(X)))
        _,seed=heads.gaussian_head(states['output'],target[:,None],1.,'mfvi')
        _,_,signals=messages.backward(records,seed,literal=True)
        d=signals['weight']
        from .gaussian import Signal
        return mfvi_update(q,jax.tree_util.tree_map(lambda a:a[:,0],d),prior)
    candidate_update=jax.jit(candidate_update)
    for sweep in range(20):
        for i in range(dim):
            proposal=candidate_update(q)
            q=Moments(q.mean.at[i].set(proposal.mean[i]),q.variance.at[i].set(proposal.variance[i]))
            mr[i]=(hr[i]-Pr[i]@mr+Pr[i,i]*mr[i])/Pr[i,i]
            av.append(np.r_[q.mean,q.variance]);vr[i]=1/Pr[i,i];rv.append(np.r_[mr,vr])
            gaps.append(np.linalg.norm(Pr@np.asarray(q.mean)-hr))
    a=np.array(av);r=np.array(rv);fig,axes=plt.subplots(1,2,figsize=(8,3.8))
    for k,stat in enumerate(['mean','variance']):
        sl=slice(0,dim) if k==0 else slice(dim,None)
        agreement_plot(r[:,sl],a[:,sl],title='MFVI '+stat,xlabel='coordinate-ascent VI',ylabel='canonical pseudo-gradient update',ax=axes[k])
    save(fig,folder,'mfvi_equivalence');plt.close(fig);traces.update(mfvi=a,reference_mfvi=r,mfvi_mean_residual=gaps)
    rows.append({'method':'diagonal MFVI','quantity':'all iterates','maximum_error':float(np.max(np.abs(a-r)))})
    np.savez_compressed(folder/'equivalence.npz',**arrays,**traces)
    (folder/'equivalence.json').write_text(json.dumps(rows,indent=2));return rows


def benchmark(f,*args,repeats=5):
    compiled=jax.jit(f);begin=time.perf_counter();out=compiled(*args);jax.block_until_ready(out);compilation=time.perf_counter()-begin
    times=[]
    for _ in range(repeats):
        begin=time.perf_counter();out=compiled(*args);jax.block_until_ready(out);times.append(time.perf_counter()-begin)
    return out,float(np.median(times)),compilation


def nonlinear_equivalence(folder='results/semantics'):
    """Literal layer messages versus local VJPs of the SAME nonlinear closure."""
    folder=Path(folder);folder.mkdir(exist_ok=True,parents=True)
    b=models.Builder(seed=41,dtype=jnp.float64,prior_variance=.15);parent='input'
    for i,(ni,no) in enumerate([(64,12),(12,8)],1):
        parent=b.add(f'dense{i}','dense',parent,shape=(ni,no));parent=b.add(f'relu{i}','relu',parent)
    b.add('readout','readout',parent,shape=(8,10));model=b.finish('Nonlinear message check')
    x=jnp.asarray(np.random.default_rng(41).normal(size=(16,64)));q=model.snapshot()
    states,records=messages.forward(model.nodes,q,Moments(x,jnp.zeros_like(x)))
    rows=[];arrays={}
    for method in ['ep','mfvi']:
        _,seed=heads.categorical_head(states['readout'],jnp.arange(16)%10,method)
        _,la,lw=messages.backward(records,seed,literal=True)
        _,pa,pw=messages.backward(records,seed,literal=False)
        fig,axes=plt.subplots(2,2,figsize=(8,7))
        for row,(left,right,names,kind) in enumerate([(la,pa,['relu1','relu2'],'activation'),(lw,pw,sorted(q),'weight')]):
            for col,(stat,label) in enumerate([('mean','mean signal'),('centered','variance sensitivity')]):
                ref=np.concatenate([np.asarray(getattr(left[n],stat)).ravel() for n in names])
                out=np.concatenate([np.asarray(getattr(right[n],stat)).ravel() for n in names])
                np.testing.assert_allclose(out,ref,atol=1e-10,rtol=1e-8)
                stem=f'{method}_{kind}_{stat}';arrays[stem]=out;arrays['literal_'+stem]=ref
                rows.append({'method':method.upper(),'quantity':kind+' '+label,'maximum_error':float(np.max(np.abs(out-ref)))})
                agreement_plot(ref,out,xlabel='literal message',ylabel='local VJP + Fisher encoding',title=method.upper()+' '+kind+' '+label,ax=axes[row,col])
        save(fig,folder,method+'_nonlinear_equivalence');plt.close(fig)
    np.savez_compressed(folder/'nonlinear_equivalence.npz',**arrays)
    (folder/'nonlinear_equivalence.json').write_text(json.dumps(rows,indent=2));return rows


def approximation_study(folder='results/semantics',samples=16384):
    """Common Gaussian weight belief; same regression objective and coordinates.

    MC references average complete networks. EP differentiates log(mean f),
    never mean(log f). Reference sensitivity is with respect to (m,v), then
    encoded in the SAME natural moment coordinates as the approximation.
    """
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    model=models.mlp(inputs=4,width=12,classes=2,dtype=jnp.float64);q=model.snapshot()
    q={n:Moments(p.mean,jnp.ones_like(p.variance)*.015) for n,p in q.items()}
    x=jnp.asarray(np.random.default_rng(9).normal(size=(3,4)));y=jnp.array([0,1,0]);key=jax.random.PRNGKey(42)
    noises={n:jax.random.normal(k,(samples,)+p.mean.shape,dtype=p.mean.dtype) for (n,p),k in zip(sorted(q.items()),jax.random.split(key,len(q)))}
    def mc(q):
        w={n:p.mean+jnp.sqrt(p.variance)*noises[n] for n,p in q.items()}
        z=jax.vmap(lambda w:messages.point_forward(model.nodes,w,x)[0])(w)
        return z
    z,mc_seconds,mc_compile=benchmark(mc,q,repeats=3)
    reference=FullMoments(z.mean(0),jnp.einsum('sbi,sbj->bij',z-z.mean(0),z-z.mean(0))/samples)
    reference_grad={};reference_signal={}
    for method in ['mfvi','ep']:
        g=jax.jit(jax.grad(lambda q:heads.categorical_objective(mc(q),y,method)))(q)
        reference_grad[method]=g;reference_signal[method]={n:encode(q[n],g[n]) for n in q}
    schemes=[('diagonal',None),('shared rank 4',4),('shared rank 16',16),('shared rank 64',64)]
    rows=[];arrays={'reference_mean':reference.mean,'reference_covariance':reference.covariance};saved=[]
    for name,rank in schemes:
        if rank is None:
            f=lambda q:messages.forward(model.nodes,q,Moments(x,jnp.zeros_like(x)))[0]['readout']
        else:f=lambda q,rank=rank:covariance.shared_records(model.nodes,q,x,jax.random.PRNGKey(91),rank)[0]['readout']
        out,seconds,compile_seconds=benchmark(f,q)
        stem=name.replace(' ','_');arrays[stem+'_mean']=out.mean;arrays[stem+'_covariance']=out.covariance
        row={'scheme':name,'forward_seconds':seconds,'compile_seconds':compile_seconds,
             'mean_relative_error':float(jnp.linalg.norm(out.mean-reference.mean)/jnp.linalg.norm(reference.mean)),
             'covariance_relative_error':float(jnp.linalg.norm(out.covariance-reference.covariance)/jnp.linalg.norm(reference.covariance))}
        for method in ['mfvi','ep']:
            if rank is None:gfun=lambda q:inference.moment_likelihood(model.nodes,q,x,y,method=method)[1]
            else:gfun=lambda q:covariance.shared_likelihood(model.nodes,q,x,y,jax.random.PRNGKey(91),rank,method=method)[1]
            grad,back_seconds,bc=benchmark(gfun,q,repeats=3)
            actual={n:encode(q[n],grad[n]) for n in q};reference_s=reference_signal[method]
            for stat in ['mean','second','centered']:
                a=np.concatenate([np.asarray(getattr(actual[n],stat)).ravel() for n in sorted(q)])
                r=np.concatenate([np.asarray(getattr(reference_s[n],stat)).ravel() for n in sorted(q)])
                arrays[stem+'_'+method+'_'+stat]=a;arrays['reference_'+method+'_'+stat]=r
                row[method+'_'+stat+'_relative_error']=float(np.linalg.norm(a-r)/np.linalg.norm(r))
            row[method+'_forward_backward_seconds']=back_seconds
        rows.append(row);saved.append((name,out))
    shown=[saved[0],saved[-1]]
    fig,axes=plt.subplots(2,2,figsize=(8,7))
    for i,(name,out) in enumerate(shown):
        agreement_plot(reference.mean,out.mean,title=name,xlabel='MC output mean',ylabel='moment output mean',ax=axes[0,i])
        agreement_plot(jnp.diagonal(reference.covariance,axis1=-2,axis2=-1),jnp.diagonal(out.covariance,axis1=-2,axis2=-1),title=name,xlabel='MC output variance',ylabel='moment output variance',ax=axes[1,i])
    save(fig,folder,'forward_approximation');plt.close(fig)
    for method in ['mfvi','ep']:
        fig,axes=plt.subplots(2,2,figsize=(8,7))
        for i,(name,_) in enumerate(shown):
            for k,stat in enumerate(['mean','centered']):
                agreement_plot(arrays['reference_'+method+'_'+stat],arrays[name.replace(' ','_')+'_'+method+'_'+stat],title=name,xlabel='MC '+('mean signal' if stat=='mean' else 'variance sensitivity'),ylabel='moment approximation',ax=axes[k,i])
        save(fig,folder,method+'_backward_approximation');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for row in rows:
        axes[0].scatter(row['forward_seconds']*1000,row['covariance_relative_error'],label=row['scheme'])
        axes[1].scatter(row['mfvi_forward_backward_seconds']*1000,row['mfvi_centered_relative_error'],label=row['scheme'])
    for ax,title in zip(axes,['Forward covariance','MFVI variance sensitivity']):ax.set(xlabel='execution milliseconds',ylabel='relative error against MC',title=title);ax.legend(fontsize=8);ax.grid(alpha=.2)
    save(fig,folder,'approximation_cost');plt.close(fig)
    receipt={'samples':samples,'mc_forward_seconds':mc_seconds,'mc_compile_seconds':mc_compile,'rows':rows,'note':'Untrained nonlinear fixture; no classification-performance claim. Centered variance sensitivity is 2 v^2 dPhi/dv, not a finite posterior variance change.'}
    np.savez_compressed(folder/'approximation.npz',**jax.tree_util.tree_map(np.asarray,arrays));(folder/'approximation.json').write_text(json.dumps(receipt,indent=2))
    return receipt


def residual_study(folder='results/semantics',samples=32768):
    """Same-belief residual correction: compare corrected and direct sampling cost."""
    from .residual import network_residual_backward
    from jax.flatten_util import ravel_pytree
    folder=Path(folder);model=models.mlp(inputs=4,width=12,classes=2,dtype=jnp.float64)
    q={n:Moments(p.mean,jnp.ones_like(p.variance)*.015) for n,p in model.snapshot().items()}
    x=jnp.asarray(np.random.default_rng(9).normal(size=(1,4)));label=0
    m,unravel=ravel_pytree({n:p.mean for n,p in q.items()});v,_=ravel_pytree({n:p.variance for n,p in q.items()})
    pred=lambda w:messages.point_forward(model.nodes,w,x)[0][0]
    eps=jax.random.normal(jax.random.key(811),(samples,len(m)),dtype=m.dtype)
    evaluation_eps=jax.random.normal(jax.random.key(812),(2048,len(m)),dtype=m.dtype)
    def mc(m,v,eps,method):
        logp=jax.vmap(lambda w:jax.nn.log_softmax(pred(unravel(w)))[label])(m+jnp.sqrt(v)*eps)
        return logp.mean() if method=='mfvi' else jax.scipy.special.logsumexp(logp)-jnp.log(len(logp))
    reference={};rows=[];arrays={}
    for method in ['mfvi','ep']:
        grad=jax.jit(jax.grad(lambda m,v,e:mc(m,v,e,method),argnums=(0,1)))
        gm,gv=grad(m,v,eps);reference[method]=encode(Moments(m,v),Moments(gm,gv))
        for stat in ['mean','centered']:arrays[method+'_reference_'+stat]=np.asarray(getattr(reference[method],stat))
        for draws in [128,512,2048]:
            (gm,gv),seconds,_=benchmark(grad,m,v,evaluation_eps[:draws],repeats=3)
            signal=encode(Moments(m,v),Moments(gm,gv));row={'method':method,'scheme':'direct sampling','draws':draws,'seconds':seconds}
            for stat in ['mean','centered']:row[stat+'_relative_error']=float(jnp.linalg.norm(getattr(signal,stat)-getattr(reference[method],stat))/jnp.linalg.norm(getattr(reference[method],stat)))
            rows.append(row)
    # Different quadrature noises from the MC reference. All four dependent
    # antithetic/shifted evaluations count toward the recorded draw budget.
    for draws in [128,512,2048]:
        begin=time.perf_counter();out=network_residual_backward(pred,q,label,jax.random.key(975),pairs=draws//4,head_power=16);jax.block_until_ready(out);seconds=time.perf_counter()-begin
        for method in ['mfvi','ep']:
            signal=out[method]['signal'];row={'method':method,'scheme':'Gaussian + nonlinear residual','draws':draws,'seconds_including_setup':seconds}
            for stat in ['mean','centered']:
                ref=getattr(reference[method],stat);a=getattr(signal,stat);row[stat+'_relative_error']=float(jnp.linalg.norm(a-ref)/jnp.linalg.norm(ref));arrays[method+'_'+str(draws)+'_'+stat]=np.asarray(a)
            rows.append(row)
    fig,axes=plt.subplots(2,2,figsize=(8,7))
    for i,method in enumerate(['mfvi','ep']):
        for k,stat in enumerate(['mean','centered']):agreement_plot(arrays[method+'_reference_'+stat],arrays[method+'_2048_'+stat],title=method.upper()+' residual correction',xlabel='independent MC '+('mean signal' if stat=='mean' else 'variance sensitivity'),ylabel='corrected signal',ax=axes[i,k])
    save(fig,folder,'residual_agreement');plt.close(fig)
    np.savez_compressed(folder/'residual.npz',**arrays);(folder/'residual.json').write_text(json.dumps({'samples':samples,'rows':rows,'timing_note':'Residual timings include Jacobian/quadrature setup and compilation; direct-sampling execution timing excludes compilation. Not a speedup claim.'},indent=2));return rows

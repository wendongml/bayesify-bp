"""Quality-first paired GPT-2 LoRA runs. No legacy metrics are training targets."""
import argparse,csv,hashlib,json,math,os,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import GPT2LoRA
from optimizers import Vadam,sampled_vadam_step,sampled_ep_step
from data import load

def metric(prob,y):
    return dict(accuracy=float((prob.argmax(-1)==y).float().mean()),
                nll=float(-prob[torch.arange(len(y),device=y.device),y].clamp_min(1e-30).log().mean()))

@torch.no_grad()
def evaluate(model,data,method,draws=4,batch=64,seed=123):
    ids,mask,y=data;means,vs=model.belief();g=torch.Generator(device=ids.device).manual_seed(seed)
    count=1 if method=='adamw' or method.startswith('moment') else draws;prob=torch.zeros((len(y),2),device=ids.device)
    torch.cuda.synchronize();start=time.perf_counter()
    if method.startswith('moment'):
        for j in range(0,len(y),batch):prob[j:j+batch]=model.moment_step.probabilities(ids[j:j+batch],mask[j:j+batch])
        torch.cuda.synchronize()
        return dict(**metric(prob,y),inference_seconds=time.perf_counter()-start,draws=0),prob.cpu()
    # One global weight sample is shared across all evaluation batches.
    for _ in range(count):
        ws=means if method=='adamw' else [m+v.sqrt()*torch.randn(m.shape,device=m.device,generator=g) for m,v in zip(means,vs)]
        for j in range(0,len(y),batch):prob[j:j+batch]+=model(ids[j:j+batch],mask[j:j+batch],ws).softmax(-1)/count
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    return dict(**metric(prob,y),inference_seconds=elapsed,draws=count),prob.cpu()

def save(path,model,opt,config,epoch,step,generator,**extra):
    state=dict(belief=model.belief_state(),optimizer=opt.state_dict(),config=config,
               epoch=epoch,step=step,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),
               generator=generator.get_state(),**extra)
    temp=path.with_suffix('.partial');torch.save(state,temp);temp.replace(path)

def run(config,root):
    root=Path(root);dest=root/'runs'/config['name'];dest.mkdir(parents=True,exist_ok=True)
    if (dest/'summary.json').exists():return json.loads((dest/'summary.json').read_text())
    (dest/'config.json').write_text(json.dumps(config,indent=2))
    (dest/'source_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')},indent=2))
    # Keep the actual scientific implementation, not only hashes of mutable
    # working files. Transport helpers are unnecessary in a per-run snapshot.
    (dest/'source').mkdir(exist_ok=True)
    for name in ('train_sampled.py','model.py','data.py','optimizers.py','moments.py','moment_training.py','adjoint_variance.py'):
        if (root/name).exists():shutil.copyfile(root/name,dest/'source'/name)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    data=load(root,config['limit'],config['seed']);train=tuple(x.cuda() for x in data['train']);val=tuple(x.cuda() for x in data['validation'])
    model=GPT2LoRA(root/'gpt2',config['seed'],config['variance']);method=config['method'];n=len(train[2]);lr=config['lr']
    if method.endswith('vadam'):opt=Vadam(model,n,config['tau'],config['precision'],lr,rule=config.get('rule','coupled'))
    elif method.endswith('power_ep'):
        opt=torch.optim.Adam([{'params':[p.mean for p in model.layers],'lr':lr},
                              {'params':[p.log_variance for p in model.layers],
                               'lr':config.get('variance_lr',lr),
                               'rate_multiplier':config.get('variance_lr',lr)/lr,
                               'eps':config.get('variance_epsilon',1e-8)}],betas=(.9,.999))
    elif method=='adamw':opt=torch.optim.AdamW([p.mean for p in model.layers],lr=lr,weight_decay=config['decay'])
    else:raise ValueError(method)
    if method.startswith('moment'):
        from moment_training import MomentStep
        model.moment_step=MomentStep(model,config.get('rank',4),config.get('moment_backend','shared_factors'))
    generator=torch.Generator(device='cuda').manual_seed(config['seed']+700)
    initial=0;global_step=0;best=math.inf;training_seconds=0.;rows=[]
    recovery=dest/'recovery.pt'
    if recovery.exists():
        state=torch.load(recovery,map_location='cuda',weights_only=False)
        model.restore_belief(state['belief']);opt.load_state_dict(state['optimizer'])
        initial=state['epoch'];global_step=state['step'];best=state['best'];training_seconds=state['training_seconds'];rows=state['rows']
        generator.set_state(state['generator'].cpu());torch.set_rng_state(state['rng_cpu'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in state['rng_cuda']])
    else:save(dest/'initial.pt',model,opt,config,0,0,generator)
    started=time.time();deadline=started+config.get('max_hours',4)*3600
    for epoch in range(initial,config['epochs']):
        # Same permutation for all methods; optimizer/sampling RNG cannot change it.
        order=torch.randperm(n,generator=torch.Generator().manual_seed(config['seed']+epoch)).cuda()
        torch.cuda.synchronize();tic=time.perf_counter();loss_sum=0.;seen=0
        for j in range(0,n,config['batch']):
            # Full-data continuation uses the same declared schedule for every
            # method. Pilot configurations retain their original constant rate.
            if config.get('schedule')=='warmup_cosine':
                total=config['epochs']*math.ceil(n/config['batch'])
                warm=max(1,round(.05*total));step=global_step+1
                multiplier=step/warm if step<=warm else .1+.9*.5*(1+math.cos(math.pi*(step-warm)/(total-warm)))
                if isinstance(opt,Vadam):opt.lr=lr*multiplier
                else:
                    for group in opt.param_groups:group['lr']=lr*multiplier*group.get('rate_multiplier',1.)
            ind=order[j:j+config['batch']];ids,mask,y=[x[ind] for x in train]
            if method.startswith('moment'):loss,field=model.moment_step.step(ids,mask,y,opt,method,n,config['tau'],config['precision'],generator)
            elif method=='sampled_vadam':loss,field=sampled_vadam_step(model,ids,mask,y,opt,generator)
            elif method=='sampled_power_ep':loss,field=sampled_ep_step(model,ids,mask,y,opt,n,config['tau'],config['precision'],4,generator)
            else:
                opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(ids,mask),y);loss.backward();opt.step()
            loss_sum+=float(loss)*len(y);seen+=len(y);global_step+=1
            if not math.isfinite(float(loss)):raise FloatingPointError('Nonfinite training objective')
            if global_step%25==0:
                (dest/'progress.json').write_text(json.dumps(dict(epoch=epoch+1,step=global_step,seen=seen,train_n=n,updated=time.time(),loss=float(loss))))
            if time.time()>deadline:raise TimeoutError('Declared worker training deadline reached; last complete epoch retained')
        torch.cuda.synchronize();training_seconds+=time.perf_counter()-tic
        scores,prob=evaluate(model,val,method,draws=config.get('validation_draws',4),seed=10000+epoch)
        row=dict(epoch=epoch+1,updates=global_step,training_seconds=training_seconds,training_objective=loss_sum/seen,**scores)
        rows.append(row)
        save(dest/f'epoch{epoch+1}.pt',model,opt,config,epoch+1,global_step,generator,metrics=scores)
        np.savez_compressed(dest/f'epoch{epoch+1}_predictions.npz',probabilities=prob.numpy(),labels=val[2].cpu().numpy())
        if scores['nll']<best:
            best=scores['nll'];save(dest/'best.pt',model,opt,config,epoch+1,global_step,generator,metrics=scores)
            np.savez_compressed(dest/'best_predictions.npz',probabilities=prob.numpy(),labels=val[2].cpu().numpy())
        with (dest/'learning_curves.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(rows)
        save(recovery,model,opt,config,epoch+1,global_step,generator,best=best,training_seconds=training_seconds,rows=rows)
        print(config['name'],json.dumps(row),flush=True)
    summary=dict(config=config,best=min(rows,key=lambda x:x['nll']),last=rows[-1],training_seconds=training_seconds,
                 elapsed_seconds=time.time()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                 status='fixed-budget completed, not convergence claim',train_n=n,validation_n=len(val[2]))
    (dest/'summary.json').write_text(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('config');parser.add_argument('--root',default='/content/moment_transformer')
    args=parser.parse_args();run(json.loads(Path(args.config).read_text()),args.root)

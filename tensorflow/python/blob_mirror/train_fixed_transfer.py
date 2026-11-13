"""The prescribed transfer, after the source gate; no tuning grid.

Likelihood: one released BLoB draw, or the validated conditional-moment kernel.
Update: released AdamW likelihood step, then SGD of rho's retained likelihood
adjoint plus the cyclic Gaussian KL. No Vadam or EP is substituted here.
The conditional kernel is explicitly a stochastic hybrid.
"""
from pathlib import Path
import argparse, json, math, time, random, hashlib
import numpy as np
import torch
from transfer_data import ROOT, prepare, batches
from transfer_model import make_model, logits
from source_stream_transfer import SourceStreamTransfer

p=argparse.ArgumentParser()
p.add_argument('--task',choices=['sst2','clinc'],required=True)
p.add_argument('--method',choices=['ordinary','sampled','hybrid'],required=True)
p.add_argument('--seed',type=int,default=41)
a=p.parse_args()
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
data=prepare(a.task);n=len(data['train']['labels'])
epochs,lr=(3,3e-4) if a.task=='sst2' else (6,1e-3)
model=make_model(a.task,a.method,a.seed,n,epochs,full_head=a.task=='clinc',
                 mean_lr=lr,tutorial_init=a.task=='sst2',point_decay=.01)
target_ids=data.get('target_ids')
assert not any(m.training for m in model.modules() if isinstance(m,torch.nn.Dropout))
params=[p for p in model.parameters() if p.requires_grad]
total=epochs*math.ceil(n/32);warm=max(1,round((.05 if a.task=='sst2' else .06)*total))


def multiplier(step):
    s=step+1
    if s<=warm:return s/warm
    u=(s-warm)/(total-warm)
    return .1+.9*(1+math.cos(math.pi*u))/2 if a.task=='sst2' else max(0.,1-u)


model.scheduler=torch.optim.lr_scheduler.LambdaLR(model.opt,multiplier)
if a.method!='ordinary':model.scheduler2=torch.optim.lr_scheduler.LambdaLR(model.opt2,multiplier)
folder=ROOT/a.task/f'v3_{a.method}_seed{a.seed}';folder.mkdir(parents=True,exist_ok=True)
names={name for name,p in model.named_parameters() if p.requires_grad or 'classifier' in name}
protocol={'task':a.task,'method':a.method,'seed':a.seed,'epochs':epochs,'batch_size':32,
          'initial_mean_lr':lr,'point_weight_decay':.01,'bayesian_adapter_decay':0.,
          'prior_std':.2,'rho_KL_SGD_lr':.002,'KL_scale':1.,'KL_cycle':model.M,
          'posterior':'Gaussian A with variance rho^4; point B',
          'training_kernel':'conditional-moment hybrid K2' if a.method=='hybrid' else a.method,
          'adaptation':'query/value LoRA4, fixed GPT2 readout' if a.task=='sst2' else 'query/value LoRA8, full POINT classifier; not full-weight Bayesian RoBERTa',
          'selection':'fixed epoch1 and epoch3, no report-set selection' if a.task=='sst2' else 'minimum training-only stopping NLL; same source MC10 for both Bayesian fits',
          'no_new_tuning':True,'source_commit':'7d960082efa3a680b76aac79d2a4f13d7f0f651a',
          'initialization':'paired means, restored after the source PEFT wrapper'}
(folder/'protocol.json').write_text(json.dumps(protocol,indent=2))
for name in ['train_fixed_transfer.py','source_stream_transfer.py','source_stream_moments.py','transfer_model.py','transfer_data.py']:
    (folder/name).write_bytes(Path(name).read_bytes())
(folder/'code_hashes.json').write_text(json.dumps({name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                                                for name in [p.name for p in folder.glob('*.py')]},indent=2))

epoch0=0;cursor=0;order=None;step=0;seconds=0.;best=math.inf


def save(epoch,cursor,order,path):
    state={'weights':{name:p.detach().cpu() for name,p in model.named_parameters() if name in names},
           'opt':model.opt.state_dict(),'scheduler':model.scheduler.state_dict(),
           'epoch':epoch,'cursor':cursor,'order':order,'step':step,'training_seconds':seconds,
           'best_stop_nll':best,'i':model.i,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
           'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
    if a.method!='ordinary':state.update(opt2=model.opt2.state_dict(),scheduler2=model.scheduler2.state_dict())
    tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)


cp=folder/'recovery.pt'
if cp.exists():
    st=torch.load(cp,map_location='cpu',weights_only=False);named=dict(model.named_parameters())
    with torch.no_grad():
        for name,w in st['weights'].items():named[name].copy_(w)
    model.opt.load_state_dict(st['opt']);model.scheduler.load_state_dict(st['scheduler'])
    if a.method!='ordinary':model.opt2.load_state_dict(st['opt2']);model.scheduler2.load_state_dict(st['scheduler2'])
    epoch0,cursor,order,step=st['epoch'],st['cursor'],st['order'],st['step']
    seconds,best,model.i=st['training_seconds'],st['best_stop_nll'],st['i']
    random.setstate(st['python_rng']);np.random.set_state(st['numpy_rng'])
    torch.set_rng_state(st['torch_rng']);torch.cuda.set_rng_state_all(st['cuda_rng'])
else:save(0,0,None,folder/'initial.pt')

for epoch in range(epoch0,epochs):
    if order is None:order=torch.randperm(n,generator=torch.Generator().manual_seed(90000+a.seed*101+epoch))
    for b in batches(data['train'],32,order[cursor:]):
        torch.cuda.synchronize();start=time.perf_counter()
        if a.method=='hybrid':
            z=SourceStreamTransfer(2,197+7919*step).forward_task(model,b,a.task,target_ids,
                                        checkpoint_blocks=b['input_ids'].shape[1]>96)
        else:z=logits(model,b,a.task,target_ids)
        loss=torch.nn.functional.cross_entropy(z,b['labels'])
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
        model.opt.step();model.opt.zero_grad();model.scheduler.step()
        kl_value=0.;pi=0.
        if a.method!='ordinary':
            # rho is absent from AdamW. Its likelihood adjoint is still present
            # when the source's prior/entropy derivative is accumulated here.
            kl=model.div_posterior_prior(model.base_model)
            cycle=(model.i-1)%model.M+1;pi=2.**(cycle-model.M-1)/(1-2.**(-model.M-1));model.i+=1
            (pi*kl).backward();model.opt2.step();model.opt2.zero_grad();model.scheduler2.step()
            kl_value=float(kl.detach())
        torch.cuda.synchronize();seconds+=time.perf_counter()-start
        step+=1;cursor+=len(b['labels'])
        row={'epoch':epoch+1,'step':step,'cursor':cursor,'training_seconds':seconds,
             'likelihood_loss':float(loss.detach()),'KL':kl_value,'KL_weight':pi,
             'lr':model.opt.param_groups[0]['lr']}
        with (folder/'learning.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if step%25==0:
            (folder/'progress.json').write_text(json.dumps(row));print(json.dumps(row),flush=True)
            save(epoch,cursor,order,cp)
    # Evaluation must not change subsequent training noise. CLINC's stopping
    # criterion uses the SAME source predictor for both learned Gaussian beliefs.
    selection=data['report'] if a.task=='sst2' else data['stop']
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
        torch.manual_seed(7119+a.seed*97+epoch)
        ps=[];ys=[];draws=1 if a.method=='ordinary' else 10
        torch.cuda.synchronize();start=time.perf_counter()
        for b in batches(selection,32):
            q=sum(logits(model,b,a.task,target_ids).softmax(-1) for _ in range(draws))/draws
            ps.append(q.cpu());ys.append(b['labels'].cpu())
        torch.cuda.synchronize();evaluation_seconds=time.perf_counter()-start
        q,y=torch.cat(ps).double(),torch.cat(ys)
        result={'epoch':epoch+1,'accuracy':float((q.argmax(-1)==y).double().mean()),
                'nll':float(-q[torch.arange(len(y)),y].log().mean()),'draws':draws,
                'evaluation_seconds':evaluation_seconds,'training_seconds':seconds,
                'reference':'training-only stopping set' if a.task=='clinc' else 'fixed reporting snapshot; not selected'}
    (folder/f'epoch{epoch+1}_metrics.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(folder/f'epoch{epoch+1}_predictions.npz',probabilities=q.numpy(),labels=y.numpy())
    if a.task=='clinc' and result['nll']<best:
        best=result['nll'];save(epoch+1,0,None,folder/'best.pt')
    save(epoch+1,0,None,cp)
    if a.task=='sst2' and epoch+1 in [1,3]:save(epoch+1,0,None,folder/f'epoch{epoch+1}.pt')
    print('EPOCH_COMPLETE',json.dumps(result),flush=True)
    cursor=0;order=None
(folder/'complete.json').write_text(json.dumps({'status':'TRAINING_COMPLETE','epochs':epochs,
                                               'steps':step,'training_seconds':seconds}))

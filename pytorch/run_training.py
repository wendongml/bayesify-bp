"""Full native image recipe; no imported benchmark predictions.

Examples: python run_training.py --architecture lenet --method moment_ep
          python run_training.py --architecture lenet --dataset dirty --method ordinary
Download and training are explicit commands. The default notebook only checks
math and regenerates figures from the saved numerical evidence.
"""
from pathlib import Path
from dataclasses import asdict
import argparse,copy,csv,json,time,math,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from bayesify_torch import data
from bayesify_torch.models import MomentMLP,MomentLeNet
from bayesify_torch.architectures import resnet20,densenet40
from bayesify_torch.benchmark import TrainingConfig,GaussianTrainer,evaluate_batches
from bayesify_torch.metrics import uncertainty_metrics,ood_metrics


def protocol(architecture,method,N,seed=41):
    cifar=architecture in ('resnet20','densenet40');point=method=='ordinary'
    lr=.003 if cifar else (3e-4 if architecture=='mlp' and method in ('ordinary','sampled_ep') else .001)
    return TrainingConfig('point' if point else method.split('_')[1],
        'point' if point else method.split('_')[0],N,lr,
        .003 if architecture=='mlp' and method=='sampled_ep' else lr,
        (.01 if method=='moment_ep' else .001) if cifar else .01,
        samples=4 if method=='sampled_ep' else 1,weight_decay=.005 if cifar and point else 0.,
        rank=1 if cifar else 0,direction_seed=seed+703,full_readout=True,
        curvature_refresh=1 if cifar else (4 if architecture=='mlp' else 2),
        adam_beta2=.999 if cifar else .99)


def initialize(architecture,method,N,seed,device):
    torch.manual_seed(seed)
    graph=({'mlp':MomentMLP,'lenet':MomentLeNet,'resnet20':resnet20,'densenet40':densenet40}[architecture]()).network.to(device)
    cfg=protocol(architecture,method,N,seed)
    # One prior and paired initial beliefs for every Bayesian method. Priors
    # apply equally to convolution, normalization, dense weights and biases.
    from bayesify_torch.inference import named_gaussian_parameters
    with torch.no_grad():
        for name,p in named_gaussian_parameters(graph):
            if architecture=='lenet' and name.endswith('weight'):
                p.mean.normal_(0.,math.sqrt(2/np.prod(p.mean.shape[1:])))
            if architecture in ('mlp','lenet') and name.endswith('bias'):
                p.mean.zero_()
            p.prior_mean.zero_();p.prior_variance.fill_(1e-3)
            p.log_variance.fill_(math.log(1e-5 if architecture in ('resnet20','densenet40') else 1e-4))
    return GaussianTrainer(graph,cfg)


def train(architecture,method,seed=41,dataset='clean',epochs=None,data_root='data',output='results',limit=0):
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_default_dtype(torch.float32)
    cifar=architecture in ('resnet20','densenet40')
    if cifar and dataset!='clean':raise ValueError('Dirty-MNIST uses MNIST architectures')
    arrays=(data.cifar10 if cifar else (data.dirty_mnist if dataset=='dirty' else data.mnist))(data_root)
    x,y=arrays['train'];xv,yv=arrays['validation']
    if limit:x,y=x[:limit],y[:limit];xv,yv=xv[:min(limit,len(xv))],yv[:min(limit,len(yv))]
    def tensors(x,y):
        return torch.from_numpy(np.ascontiguousarray(x.transpose(0,3,1,2))).float().to(device),torch.as_tensor(y,dtype=torch.long,device=device)
    x,y=tensors(x,y);xv,yv=tensors(xv,yv)
    trainer=initialize(architecture,method,len(y),seed,device);cfg=trainer.config
    batch=125 if cifar else 128;epochs=epochs or (100 if cifar else (40 if architecture=='mlp' else 30))
    folder=Path(output)/f'{architecture}_{method}_{dataset}_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    manifest=dict(architecture=architecture,method=method,seed=seed,dataset=dataset,epochs=epochs,batch=batch,
                  training=asdict(cfg),device=torch.cuda.get_device_name() if device=='cuda' else 'CPU',
                  train_n=len(y),validation_n=len(yv),selection='minimum validation predictive NLL',limit=limit,
                  torch_version=torch.__version__)
    (folder/'config.json').write_text(json.dumps(manifest,indent=2))
    def sync():
        if device=='cuda':torch.cuda.synchronize()
    def batches(a,b):return ((a[i:i+128],b[i:i+128]) for i in range(0,len(b),128))
    def evaluate(a,b,draws=20):
        trainer.graph.eval();sync();t=time.perf_counter()
        r=evaluate_batches(trainer,batches(a,b),draws,torch.Generator(device=device).manual_seed(seed+1103))
        sync();r['seconds']=time.perf_counter()-t;return r
    def checkpoint(epoch):
        opt=trainer.optimizer
        state=opt.state_dict() if isinstance(opt,torch.optim.Optimizer) else {k:v for k,v in vars(opt).items() if k!='parameters'}
        return dict(model=trainer.graph.state_dict(),optimizer=state,curvature=trainer.curvature,
                    updates=trainer.updates,epoch=epoch,configuration=manifest,rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else [],
                    data_generator=rng.get_state(),posterior_generator=sample_rng.get_state())
    rng=torch.Generator(device=device).manual_seed(seed)
    sample_rng=torch.Generator(device=device).manual_seed(seed+101)
    rows=[];best=float('inf');train_seconds=0.
    torch.save(checkpoint(0),folder/'initial.pt')
    for epoch in range(1,epochs+1):
        # All methods share the declared schedule; no test-set stopping/tuning.
        rate=.03+.97*.5*(1+math.cos(math.pi*(epoch-1)/max(1,epochs-1)))
        opt=trainer.optimizer
        if isinstance(opt,torch.optim.Optimizer):
            for group in opt.param_groups:group['lr']=cfg.learning_rate*rate
        else:
            opt.learning_rate=cfg.learning_rate*rate
            if hasattr(opt,'variance_learning_rate'):opt.variance_learning_rate=cfg.variance_learning_rate*rate
        trainer.graph.train();sync();start=time.perf_counter();order=torch.randperm(len(y),device=device,generator=rng);losses=[]
        for ids in order.split(batch):
            a=x[ids]
            if cifar:
                # Standard crop/flip; the one augmented example is shared by
                # all posterior networks in this minibatch.
                padded=F.pad(a,(4,4,4,4))
                offsets=torch.randint(0,9,(len(a),2),device=device,generator=rng)
                crop_rows=offsets[:,0,None,None]+torch.arange(32,device=device)[None,:,None]
                crop_cols=offsets[:,1,None,None]+torch.arange(32,device=device)[None,None,:]
                a=padded[torch.arange(len(a),device=device)[:,None,None],:,crop_rows,crop_cols].permute(0,3,1,2)
                flip=torch.rand(len(a),device=device,generator=rng)<.5;a=torch.where(flip[:,None,None,None],a.flip(-1),a)
            losses.append(trainer.step(a,y[ids],sample_rng))
        sync();train_seconds+=time.perf_counter()-start;r=evaluate(xv,yv)
        row=dict(epoch=epoch,training_seconds=train_seconds,validation_seconds=r['seconds'],
                 validation_nll=r['nll'],validation_accuracy=r['accuracy'],internal_objective=float(np.mean(losses)))
        rows.append(row)
        with (folder/'learning_curves.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=row);writer.writeheader();writer.writerows(rows)
        if r['nll']<best:
            best=r['nll'];torch.save(checkpoint(epoch),folder/'best.pt')
        torch.save(checkpoint(epoch),folder/'last.pt')
        print(json.dumps(dict(run=folder.name,**row)),flush=True)
    best_state=torch.load(folder/'best.pt',map_location=device,weights_only=False);trainer.graph.load_state_dict(best_state['model'])
    xt,yt=tensors(*arrays['test']);r=evaluate(xt,yt);p=r['probability'].numpy();labels=r['label'].numpy()
    xo,yo=(data.svhn if cifar else data.fashion)(data_root);xo,yo=tensors(xo,yo);po=evaluate(xo,yo)['probability'].numpy()
    saved=dict(p=p,y=labels,p_ood=po);score=dict(uncertainty_metrics(p,labels),**ood_metrics(p,po))
    if dataset=='dirty':
        xa,ya=arrays['ambiguous_test'];xa,unused=tensors(xa,np.zeros(len(xa),dtype='int64'))
        pa=evaluate(xa,unused)['probability'].numpy();saved.update(p_ambiguous=pa,ambiguous_labels=ya)
        # Each ambiguous image has ten labels. Count these for proper scores,
        # but count unique images ONCE for OOD rankings.
        score=dict(uncertainty_metrics(np.concatenate((p,np.repeat(pa,10,axis=0))),np.concatenate((labels,ya.ravel()))),
                   **ood_metrics(np.concatenate((p,pa)),po))
    np.savez_compressed(folder/'predictions.npz',**saved)
    score.update(training_seconds=train_seconds,inference_seconds=r['seconds'],selected_epoch=best_state['epoch'],
                 status='complete_fixed_budget' if not limit else 'execution_control_only')
    (folder/'metrics.json').write_text(json.dumps(score,indent=2));return score


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--architecture',choices=['mlp','lenet','resnet20','densenet40'],default='lenet')
    p.add_argument('--method',choices=['ordinary','sampled_mfvi','moment_mfvi','sampled_ep','moment_ep'],default='moment_ep')
    p.add_argument('--dataset',choices=['clean','dirty'],default='clean');p.add_argument('--seed',type=int,default=41)
    p.add_argument('--epochs',type=int);p.add_argument('--data',default='data');p.add_argument('--output',default='results');p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();train(a.architecture,a.method,a.seed,a.dataset,a.epochs,a.data,a.output,a.limit)

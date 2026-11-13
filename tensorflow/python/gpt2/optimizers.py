"""Reuse the CIFAR Vadam recurrence and tied power-EP field, in ordinary Torch.

Vadam uses likelihood E[g], E[g²]. EP uses its own normalizer objective and
Adam-normalized mean/log-variance derivatives. These are NOT the same update.
"""
import torch


def cavity_field(means,variances,n,precision,alpha,tau):
    k=alpha/n
    assert 0<k<1
    cm=[];cv=[];field=means[0].new_zeros(())
    for m,v in zip(means,variances):
        d=1+k*(precision*v-1)
        cm.append((1-k)*m/d);cv.append(v/d)
        field=field+.5*tau/alpha*(torch.log1p(k*(precision*v-1))
            -k*(precision*v).log()+k*(1-k)*precision*m.square()/d).sum()
    return cm,cv,field


def field_pullback(means,variances,gm,gv,n,p,alpha,tau):
    k=alpha/n;first=[];second=[]
    for m,v,a,b in zip(means,variances,gm,gv):
        d=1+k*(p*v-1);mc=(1-k)*m/d
        first.append((1-k)*a/d+tau/n*(1-k)*p*m/d)
        second.append((1-k)*b/d.square()-a*mc*k*p/d+
                      .5*tau/n*(p/d-1/v-k*(1-k)*p*p*m.square()/d.square()))
    return first,second


class Vadam:
    """Same recurrence as efficiency/modules.py, with checkpointable EMA state."""
    def __init__(self,model,n,tau,precision,lr,betas=(.9,.99),rule='coupled'):
        self.layers=model.layers; self.n=n/tau; self.p=precision;self.lr=lr
        self.betas=betas;self.rule=rule;self.t=0
        self.r=[torch.zeros_like(p.mean) for p in self.layers]
        self.s=[(1-precision*p.log_variance.exp())/(self.n*p.log_variance.exp()) for p in self.layers]
        assert all(bool((s>=0).all()) for s in self.s),'Initial variance exceeds prior variance.'

    @torch.no_grad()
    def step(self,first,second):
        self.t+=1;b1,b2=self.betas;lam=self.p/self.n
        for p,r,s,g,g2 in zip(self.layers,self.r,self.s,first,second):
            r.mul_(b1).add_((1-b1)*(g+lam*p.mean if self.rule=='coupled' else g))
            s.mul_(b2).add_((1-b2)*g2)
            metric=(s/(1-b2**self.t)).sqrt()+lam
            new=p.mean-self.lr*(r/(1-b1**self.t))/metric
            if self.rule=='prior_prox': new=new/(1+self.lr*lam/metric)
            elif self.rule!='coupled':raise ValueError(self.rule)
            p.mean.copy_(new);p.log_variance.copy_((1/(self.n*s+self.p)).log())

    def state_dict(self):
        return dict(t=self.t,r=self.r,s=self.s,lr=self.lr,n=self.n,p=self.p,rule=self.rule)

    def load_state_dict(self,state):
        self.t=state['t'];self.lr=state['lr']
        for a,b in zip(self.r,state['r']):a.copy_(b)
        for a,b in zip(self.s,state['s']):a.copy_(b)


def sampled_vadam_step(model,ids,mask,y,opt,generator):
    means,vs=model.belief()
    weights=[(m.detach()+v.detach().sqrt()*torch.randn(m.shape,device=m.device,generator=generator)).requires_grad_()
             for m,v in zip(means,vs)]
    loss=torch.nn.functional.cross_entropy(model(ids,mask,weights),y)
    gradients=torch.autograd.grad(loss,weights)
    opt.step(gradients,[g.square() for g in gradients])
    return loss.detach(),0.


def sampled_ep_step(model,ids,mask,y,opt,n,tau,precision,draws,generator):
    means,vs=model.belief(); alpha=tau  # power alpha/tau = 1, separately tunable later.
    cm,cv,field=cavity_field(means,vs,n,precision,alpha,tau)
    opt.zero_grad(set_to_none=True);ll=[]
    for _ in range(draws):
        w=[m+v.sqrt()*torch.randn(m.shape,device=m.device,generator=generator) for m,v in zip(cm,cv)]
        ll.append(model(ids,mask,w).log_softmax(-1).gather(1,y[:,None]).squeeze(1))
    loglikes=torch.stack(ll)
    objective=-(torch.logsumexp(loglikes,0)-__import__('math').log(draws)).mean()+field
    objective.backward();opt.step()
    return objective.detach(),field.detach()

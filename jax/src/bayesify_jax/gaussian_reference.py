"""Independent NumPy/SciPy conditioning and RTS; JAX candidate below is separate."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve,block_diag


def condition(mean,covariance,H,target,noise):
    S=noise+H@covariance@H.T
    K=cho_solve(cho_factor(S,lower=True),H@covariance).T
    return mean+K@(target-H@mean),covariance-K@H@covariance


def chain_reference(layers,weights,input_belief,H,y,R):
    means=[input_belief[0]];covs=[input_belief[1]]
    for (A,B,c),(m,V) in zip(layers,weights):
        means.append(A@means[-1]+B@m+c);covs.append(A@covs[-1]@A.T+B@V@B.T)
    sm=[None]*len(means);sv=[None]*len(means);wm=[None]*len(weights);wv=[None]*len(weights)
    sm[-1],sv[-1]=condition(means[-1],covs[-1],H,y,R)
    for i in reversed(range(len(layers))):
        A,B,_=layers[i];K=cho_solve(cho_factor(covs[i+1],lower=True),A@covs[i]).T
        Kw=cho_solve(cho_factor(covs[i+1],lower=True),B@weights[i][1]).T
        d=sm[i+1]-means[i+1];D=sv[i+1]-covs[i+1]
        sm[i]=means[i]+K@d;sv[i]=covs[i]+K@D@K.T
        wm[i]=weights[i][0]+Kw@d;wv[i]=weights[i][1]+Kw@D@Kw.T
    # Dense joint conditioning is independent of the smoothing recursion.
    blocks=[input_belief]+list(weights);m=np.concatenate([q[0] for q in blocks]);C=block_diag(*[q[1] for q in blocks])
    F=np.zeros((len(input_belief[0]),len(m)));F[:,:len(input_belief[0])]=np.eye(len(input_belief[0]));offset=np.zeros(len(input_belief[0]));cursor=len(input_belief[0])
    for (A,B,c),(w,V) in zip(layers,weights):
        F=A@F;F[:,cursor:cursor+len(w)]+=B;offset=A@offset+c;cursor+=len(w)
    dm,dC=condition(m,C,H@F,y-H@offset,R)
    return dict(forward_mean=means,forward_covariance=covs,activation_mean=sm,activation_covariance=sv,weight_mean=wm,weight_covariance=wv,dense_mean=dm,dense_covariance=dC)


def cavi(precision,information,mean,sweeps=1):
    m=mean.copy();v=1/np.diag(precision)
    for _ in range(sweeps):
        for i in range(len(m)):m[i]=(information[i]-precision[i]@m+precision[i,i]*m[i])/precision[i,i]
    return m,v


def ep_site(cavity_mean,cavity_covariance,H,y,R,diagonal=False):
    m,C=condition(cavity_mean,cavity_covariance,H,y,R)
    if diagonal:C=np.diag(np.diag(C))
    P=np.linalg.solve(C,np.eye(len(m)));Pc=np.linalg.solve(cavity_covariance,np.eye(len(m)))
    return P-Pc,P@m-Pc@cavity_mean,m,C

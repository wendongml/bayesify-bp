"""Rebuild the cross-implementation comparison from saved numerical arrays.

This is an audit of identical learned beliefs, not a model-selection benchmark.
No checkpoint, R figure, or network training is required to rebuild the plots.
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / 'evidence'


def compare():
    rows = []
    for method in ('mfvi', 'ep'):
        a = np.load(ROOT / f'{method}_comparison.npz')
        for backend in ('tensorflow', 'pytorch', 'jax'):
            for rank in (1, 4, 16, 64):
                for name in ('output_mean', 'covariance', 'mean_signal', 'variance_signal'):
                    target = a['mc_' + name]
                    estimate = a[f'{backend}_r{rank}_{name}']
                    rows.append(dict(method=method, backend=backend, rank=rank, quantity=name,
                        relative_error=float(np.linalg.norm(estimate-target)/np.linalg.norm(target))))
    with (ROOT / 'approximation_errors.csv').open('w') as f:
        out = csv.DictWriter(f, fieldnames=rows[0]);out.writeheader();out.writerows(rows)
    return rows


def coordinates():
    """Plain arrays for independently rendered R and Python figures."""
    rows=[]
    for method in ('mfvi','ep'):
        a=np.load(ROOT/f'{method}_comparison.npz')
        for backend in ('tensorflow','pytorch','jax'):
            for quantity in ('covariance','mean_signal','variance_signal'):
                target=a['mc_'+quantity]
                target=np.diag(target) if quantity=='covariance' else target.ravel()
                schemes=('r1','r64') if quantity=='covariance' else ('r64','corrected_s4096','corrected_s16384')
                for scheme in schemes:
                    key=scheme if scheme.startswith('corrected') else backend+'_'+scheme
                    value=a[key+'_'+quantity]
                    value=np.diag(value) if quantity=='covariance' else value.ravel()
                    for i,(x,y) in enumerate(zip(target,value)):
                        rows.append(dict(method=method,backend=backend,quantity=quantity,
                            scheme=scheme,coordinate=i,reference=float(x),estimate=float(y)))
    with (ROOT/'coordinates.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)


def figures(rows, backend='pytorch'):
    # One backend is displayed because independent native computations agree
    # to float64 precision; the CSV retains all three rather than overplotting.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, method in zip(axes, ('mfvi', 'ep')):
        for name, label in [('covariance', 'Forward covariance'),
                            ('mean_signal', 'Readout mean signal'),
                            ('variance_signal', 'Readout variance signal')]:
            r = [x for x in rows if x['method']==method and x['backend']==backend and x['quantity']==name]
            ax.plot([x['rank'] for x in r], [100*x['relative_error'] for x in r], 'o-', label=label)
        ax.set(xscale='log', yscale='log', xlabel='Retained directions', ylabel='Relative error (%)', title=method.upper()+' — closure ablation')
        ax.set_xticks([1,4,16,64], ['1','4','16','64']);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(ROOT/f'covariance_and_message_error.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, method in zip(axes, ('mfvi','ep')):
        a=np.load(ROOT/f'{method}_comparison.npz');x=np.diag(a['mc_covariance'])
        ys=[np.diag(a[f'{backend}_r{rank}_covariance']) for rank in (1,64)]
        for y,label in zip(ys, ('1 direction','64 directions')):ax.scatter(x,y,label=label,s=24)
        hi=1.05*max(x.max(),*(y.max() for y in ys))
        ax.plot([0,hi],[0,hi],'k--',lw=1)
        ax.set(xlim=(0,hi),ylim=(0,hi),xlabel='MC logit variance',ylabel='Moment logit variance',title=method.upper()+' — closure ablation')
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(ROOT/f'forward_agreement.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for col, method in enumerate(('mfvi','ep')):
        a = np.load(ROOT/f'{method}_comparison.npz')
        for row, quantity in enumerate(('mean_signal','variance_signal')):
            ax=axes[row,col];x=a['mc_'+quantity].ravel()
            ys=[a['corrected_s4096_'+quantity].ravel(),a['corrected_s16384_'+quantity].ravel()]
            for y, label in zip(ys, ('Paired correction: 4,096','Paired correction: 16,384')):
                ax.scatter(x,y,s=7,alpha=.55,label=label)
            lo=min(x.min(),*(y.min() for y in ys));hi=max(x.max(),*(y.max() for y in ys))
            margin=.05*(hi-lo);lo-=margin;hi+=margin
            ax.plot([lo,hi],[lo,hi],'k--',lw=1)
            ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='MC reference',ylabel='Computed signal',
                   title=method.upper()+': '+('mean signal' if row==0 else 'centered variance signal'))
            ax.set_aspect('equal',adjustable='box');ax.ticklabel_format(style='sci',axis='both',scilimits=(-2,2))
            ax.legend(fontsize=9);ax.grid(alpha=.15)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(ROOT/f'readout_agreement.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    rows=compare();coordinates();figures(rows)
    print('Rebuilt',len(rows),'approximation comparisons and three figures.')

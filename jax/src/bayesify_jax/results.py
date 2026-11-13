"""Tables and figures from THIS implementation's saved numerical predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .plots import calibration_plot,risk_coverage_plot,ood_roc_plot
from .uncertainty import uncertainty_metrics,ood_metrics,ranking_curve

LABELS={'ordinary':'Ordinary','sampled_mfvi':'Sampled MFVI','moment_mfvi':'Moment MFVI','sampled_ep':'Sampled EP','moment_ep':'Moment EP','hybrid':'Conditional moment MFVI','sampled':'Sampled conditional MFVI'}


def collect(root='results'):
    rows=[]
    for p in sorted(Path(root).glob('*/metrics.json')):
        cfg=json.loads((p.parent/'config.json').read_text());receipt=json.loads((p.parent/'receipt.json').read_text())
        if receipt.get('status')!='full_fixed_budget':continue
        m=json.loads(p.read_text());method=cfg['method'];name=cfg.get('architecture',cfg.get('task'))
        rows.append({'architecture':name,'dataset':cfg.get('dataset','clean'),'method':method,'seed':cfg['seed'],'optimizer':cfg.get('optimizer','AdamW' if method=='ordinary' else 'BLoB split AdamW/SGD'),
            'learning_rate':cfg.get('learning_rate',3e-4 if name=='sst2' else (.001 if name=='clinc' else None)),
            'variance_rate':None if method=='ordinary' or cfg.get('optimizer')=='vadam' else cfg.get('variance_rate',.002 if 'task' in cfg else None),
            'weight_decay':cfg.get('weight_decay',.01 if method=='ordinary' or name=='clinc' else 0.),
            'temperature':None if method=='ordinary' else cfg.get('temperature'),'training_draws':(1 if method in ('sampled','hybrid') else 0) if 'task' in cfg else (cfg['samples'] if method.startswith('sampled') else 0),
            'conditional_pairs':2 if method=='hybrid' else 0,
            'inference_draws':0 if method=='ordinary' else (cfg.get('prediction_samples',10) if ('task' in cfg or method.startswith('sampled')) else 0),
            'epochs':receipt.get('completed_epochs',cfg.get('epochs')),'compile_seconds':receipt.get('compile_seconds'),**m,'folder':str(p.parent)})
    frame=pd.DataFrame(rows)
    if len(frame):
        for (arch,dataset),group in frame.groupby(['architecture','dataset']):
            base=group[group.method=='ordinary']
            if len(base):
                chosen=(frame.architecture==arch)&(frame.dataset==dataset)
                frame.loc[chosen,'training_multiple']=frame.loc[chosen,'training_seconds']/base.iloc[0].training_seconds
                frame.loc[chosen,'inference_multiple']=frame.loc[chosen,'inference_seconds']/base.iloc[0].inference_seconds
        frame.to_csv(Path(root)/'benchmark_table.csv',index=False)
    return frame


def plots(root='results',output='figures'):
    from .experiments import save
    frame=collect(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if not len(frame):return frame
    for (arch,dataset),group in frame.groupby(['architecture','dataset'],sort=False):
        name=arch+('_dirty' if dataset=='dirty' else '')
        fig,grid=plt.subplots(2,2,figsize=(9.6,8));axes=grid.ravel();learning,curves=plt.subplots(2,2,figsize=(9,7))
        has_ood=all((Path(folder)/'ood_predictions.npz').exists() for folder in group.folder)
        for _,row in group.iterrows():
            folder=Path(row.folder);a=np.load(folder/'test_predictions.npz');p,y=a['probabilities'],a['labels'];label=LABELS.get(row.method,row.method)
            p_id=p
            if dataset=='dirty':
                ambiguous=np.load(folder/'ambiguous_predictions.npz');pa=ambiguous['probabilities']
                p_id=np.concatenate((p,pa))
                p=np.concatenate((p,np.repeat(pa,10,axis=0)));y=np.concatenate((y,ambiguous['labels'].ravel()))
            calibration_plot(p,y,ax=axes[0],label=label);risk_coverage_plot(p,y,ax=axes[1],label=label)
            error=ranking_curve(1-p.max(1),p.argmax(1)!=y)
            axes[2].plot(error.fpr,error.tpr,label=f'{label} ({error.auroc:.3f})')
            if has_ood:
                po=np.load(folder/'ood_predictions.npz')['probabilities'];entropy=lambda p:-(p*np.log(np.maximum(p,1e-30))).sum(-1)
                ood_roc_plot(entropy(p_id),entropy(po),ax=axes[3],label=label)
            else:
                axes[3].hist(p.max(1),bins=np.linspace(0,1,31),histtype='step',density=True,label=label)
            c=pd.read_json(folder/'curves.json').rename(columns=lambda x:x.replace('selection_','validation_'))
            for col,(x,xlabel) in enumerate([('epoch','epoch'),('training_seconds','training seconds')]):
                for r,(y,title) in enumerate([('validation_nll','Validation NLL'),('validation_accuracy','Validation accuracy')]):
                    curves[r,col].plot(c[x],c[y],label=label);curves[r,col].set(xlabel=xlabel,ylabel=title);curves[r,col].grid(alpha=.2)
                    if y=='validation_nll':curves[r,col].set_yscale('log')
        axes[2].plot([0,1],[0,1],'k--',lw=.7)
        axes[2].set(title='Error detection ROC (AUROC)',xlabel='correct cases flagged',ylabel='errors flagged',xlim=(0,1),ylim=(0,1),aspect='equal')
        for ax in axes:ax.legend(fontsize=8,loc='best');ax.grid(alpha=.15)
        axes[3].set_title(('Fashion-MNIST OOD (entropy)' if arch in ('mlp','lenet') else 'OOD detection (entropy)') if has_ood else 'Confidence distribution')
        fig.suptitle(name+' — native JAX predictions; seed 41',fontsize=11);save(fig,output,name+'_uncertainty');plt.close(fig)
        for ax in curves.ravel():ax.legend(fontsize=10,loc='best')
        save(learning,output,name+'_learning');plt.close(learning)
    return frame


def reader_table(frame,architecture,dataset='clean'):
    cols=['method','accuracy','nll','ece','brier','aurc','error_auroc','entropy_auroc','training_seconds','training_multiple','inference_seconds','inference_multiple']
    cols=[c for c in cols if c in frame]
    out=frame[(frame.architecture==architecture)&(frame.dataset==dataset)][cols].copy();out['method']=out.method.map(lambda x:LABELS.get(x,x))
    return out.rename(columns={'accuracy':'Accuracy ↑','nll':'NLL ↓','ece':'ECE ↓','brier':'Brier ↓','aurc':'AURC ↓','error_auroc':'Error AUROC ↑','entropy_auroc':'OOD AUROC ↑','training_seconds':'Train s ↓','training_multiple':'Train × ↓','inference_seconds':'Infer s ↓','inference_multiple':'Infer × ↓'})


def protocol_table(frame,architecture,dataset='clean'):
    cols=['method','optimizer','learning_rate','variance_rate','weight_decay','temperature','training_draws','conditional_pairs','inference_draws','epochs','selected_epoch','compile_seconds']
    out=frame[(frame.architecture==architecture)&(frame.dataset==dataset)][[c for c in cols if c in frame]].copy()
    out['method']=out.method.map(lambda x:LABELS.get(x,x))
    return out.rename(columns={'variance_rate':'log variance / rho LR','weight_decay':'point / head decay','temperature':'KL / external field strength','training_draws':'training samples','inference_draws':'stochastic prediction passes','compile_seconds':'compile s'})


def learned_fidelity(root='results/learned_fidelity',output='figures'):
    """One forward and one backward comparison; all displayed points are saved data."""
    from .plots import agreement_plot
    from .experiments import save
    root=Path(root);output=Path(output);output.mkdir(exist_ok=True,parents=True)
    if not all((root/(m+'.npz')).exists() for m in ['mfvi','ep']):return pd.DataFrame()
    rows=[];forward,fa=plt.subplots(2,2,figsize=(8,7));backward,ba=plt.subplots(2,2,figsize=(8,7))
    for row,method in enumerate(['mfvi','ep']):
        a=np.load(root/(method+'.npz'));receipt=json.loads((root/(method+'.json')).read_text())
        for column,rank in enumerate([1,64]):
            title=f'{method.upper()} — {rank} shared direction'+('s' if rank>1 else '')
            agreement_plot(np.diag(a['mc_covariance']),np.diag(a[f'rank{rank}_covariance']),title=title,xlabel='MC logit variance',ylabel='moment logit variance',ax=fa[row,column])
            agreement_plot(a['mc_centered'],a[f'rank{rank}_centered'],title=title,xlabel='MC variance sensitivity',ylabel='moment variance sensitivity',ax=ba[row,column])
        for values in receipt['rows']:
            rows.append({'method':method.upper(),**values,'MC draws':receipt['networks'],'MC seconds':receipt['MC_execution_seconds'],'MC relative SE':receipt['readout_relative_MCSE']['centered']})
    save(forward,output,'learned_forward');save(backward,output,'learned_backward');plt.close(forward);plt.close(backward)
    table=pd.DataFrame(rows);table.to_csv(output/'learned_fidelity.csv',index=False);return table

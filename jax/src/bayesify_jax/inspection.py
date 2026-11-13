"""Architecture inspection from real Flax modules and traced JAX functions."""
from pathlib import Path
import json,re
import jax
import jax.numpy as jnp
from flax import nnx
from . import models,transformers as T


def summary(model,example,folder,name):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    args=example if isinstance(example,tuple) else (example,)
    inputs=dict(ids=args[0],mask=args[1]) if len(args)==2 else dict(x=args[0])
    only_layers=lambda row:row.type.__name__ in {'MomentModel','MomentLayer','Transformer','TransformerBlock'} and not any(str(x) in {'base_arrays','snapshot','statistics','priors'} for x in row.path)
    # Rich otherwise bypasses string capture in IPython and writes an HTML
    # display directly, leaving the downloadable summary file empty.
    console={'width':160,'force_terminal':False,'force_jupyter':False,'color_system':None}
    text=nnx.tabulate(model,**inputs,depth=None,row_filter=only_layers,
                     console_kwargs=console,column_kwargs={'overflow':'fold'})
    text=re.sub(r'\x1b\[[0-9;]*m','',text);(folder/(name+'.txt')).write_text(text)
    def boundary(row):
        if not only_layers(row):return False
        if not row.path or name in ('mlp','lenet'):return True
        last=str(row.path[-1])
        if name=='resnet20':return last in {'stem_conv','global_pool','classifier'} or bool(re.fullmatch(r'stage[123]_block3_relu2',last))
        if name=='densenet40':return last in {'stem_conv','global_pool','classifier'} or bool(re.fullmatch(r'dense[123]_12_concat|transition[12]_pool',last))
        return last in {'0','5','11'}
    short=nnx.tabulate(model,**inputs,depth=None,row_filter=boundary,
                      console_kwargs={**console,'width':110},column_kwargs={'overflow':'fold'})
    (folder/(name+'_compact.txt')).write_text(re.sub(r'\x1b\[[0-9;]*m','',short))
    return text


def export_onnx(model,example,path):
    """Optional maintained exporter; output is checked against the live model.

    Run in an environment with jax2onnx + ONNX Runtime, then inspect in Netron.
    The Bayesian state remains in JAX; this export describes the mean network.
    """
    import numpy as np
    import onnxruntime as ort
    from jax2onnx import to_onnx
    args=example if isinstance(example,tuple) else (example,)
    to_onnx(model,list(args),return_mode='file',output_path=str(path))
    session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    expected=np.asarray(model(*args));actual=session.run(None,{x.name:np.asarray(v) for x,v in zip(session.get_inputs(),args)})[0]
    np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=3e-4)
    return {'maximum_absolute_error':float(np.max(np.abs(actual-expected))),'bytes':Path(path).stat().st_size}


def all_summaries(folder='assets/architectures'):
    records=[]
    for name,shape in [('mlp',(1,28,28,1)),('lenet',(1,28,28,1)),('resnet20',(1,32,32,3)),('densenet40',(1,32,32,3))]:
        model=getattr(models,name)();summary(model,jnp.zeros(shape,dtype=jnp.float32),folder,name)
        records.append({'architecture':name,'parameters':models.parameter_count(model),'shape':shape});del model
    for name in ['gpt2','roberta']:
        c=T.Config(architecture=name,rank=4 if name=='gpt2' else 8,alpha=32 if name=='gpt2' else 8,classes=2 if name=='gpt2' else 150,vocabulary=50257 if name=='gpt2' else 50265,positions=1024 if name=='gpt2' else 514)
        base=T.random_base(c);q=T.initial_beliefs(c);model=T.Transformer(c,base,q)
        summary(model,(jnp.full((1,8),2,dtype=jnp.int32),jnp.ones((1,8),dtype=jnp.int32)),folder,name)
        records.append({'architecture':name,'base_parameters':sum(x.size for x in base.values()),'adapter_mean_parameters':sum(x.mean.size for x in q.values()),'random_initialization_for_inspection':True});del model,base,q
    Path(folder,'manifest.json').write_text(json.dumps(records,indent=2));return records

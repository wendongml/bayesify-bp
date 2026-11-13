"""Actual transformer mean graphs, with weights as inputs rather than 500 MB assets."""
from pathlib import Path
import json,gc
import jax,jax.numpy as j,numpy as np,onnxruntime as ort
from jax2onnx import to_onnx,onnx_function
from bayesify_jax import transformers as T
@onnx_function(type='TransformerBlock')
def exported_block(h,mask,*arrays,base_names,weight_names,cfg,index):
 base=dict(zip(base_names,arrays[:len(base_names)]))
 weights=dict(zip(weight_names,arrays[len(base_names):]))
 return T.block(h,mask,base,weights,cfg,index)

root=Path('assets/architectures/onnx');root.mkdir(parents=True,exist_ok=True);rows=[]
for name in ['gpt2','roberta']:
 c=T.Config(architecture=name,rank=4 if name=='gpt2' else 8,alpha=32 if name=='gpt2' else 8,classes=2 if name=='gpt2' else 150,vocabulary=50257 if name=='gpt2' else 50265,positions=1024 if name=='gpt2' else 514)
 base=T.random_base(c);weights={n:q.mean for n,q in T.initial_beliefs(c).items()};bn=tuple(sorted(base));wn=tuple(sorted(weights))
 ids=j.full((1,8),2,dtype=j.int32);mask=j.ones_like(ids);targets=j.array([0,1])
 # Same block function as the executable model, annotated only for graph viewing.
 def model(ids,mask,*arrays):
  b=dict(zip(bn,arrays[:len(bn)]));w=dict(zip(wn,arrays[len(bn):]));h=T.embeddings(b,ids,mask,c)
  for i in range(c.layers):
   keys_b=tuple(n for n in bn if n.startswith(f"block{i}."));keys_w=tuple(n for n in wn if n.startswith(f"block{i}."))
   h=exported_block(h,mask,*[b[n] for n in keys_b],*[w[n] for n in keys_w],base_names=keys_b,weight_names=keys_w,cfg=c,index=i)
  return T.readout(h,b,w,mask,c,targets)
 values=[ids,mask]+[base[n] for n in bn]+[weights[n] for n in wn]
 path=root/(name+'.onnx');to_onnx(model,[jax.ShapeDtypeStruct(v.shape,v.dtype) for v in values],return_mode='file',output_path=str(path),input_names=['token_ids','attention_mask']+list(bn)+list(wn))
 session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider']);feed={x.name:np.asarray(v) for x,v in zip(session.get_inputs(),values)}
 expected=np.asarray(T.point(base,weights,ids,mask,c,targets));actual=session.run(None,feed)[0];np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=3e-4)
 row={'architecture':name,'maximum_absolute_error':float(abs(actual-expected).max()),'bytes':path.stat().st_size,'weights':'explicit graph inputs; not bundled','input_names':bn+wn};rows.append(row);print({k:v for k,v in row.items() if k!='input_names'},flush=True)
 (root/'transformer_manifest.json').write_text(json.dumps(rows,indent=2));del session,feed,base,weights,values;gc.collect();jax.clear_caches()

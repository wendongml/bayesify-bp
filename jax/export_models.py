"""Native mean-network graphs, with parameters as inputs; no checkpoint payload."""
from pathlib import Path
import json,gc
import numpy as np
import jax,jax.numpy as j,onnxruntime as ort
from jax2onnx import to_onnx
from bayesify_jax import models,messages
root=Path('assets/architectures/onnx');root.mkdir(parents=True,exist_ok=True);rows=[]
for name in ['mlp','lenet','resnet20','densenet40']:
 model=getattr(models,name)();x=j.ones((1,28,28,1) if name in ['mlp','lenet'] else (1,32,32,3),j.float32)*.1
 q=model.snapshot();names=tuple(sorted(q));statistics=model.statistics();nodes=model.nodes
 def point(x,*weights):return messages.point_forward(nodes,dict(zip(names,weights)),x,statistics)[0]
 values=[x]+[q[n].mean for n in names];path=root/(name+'.onnx')
 to_onnx(point,[jax.ShapeDtypeStruct(a.shape,a.dtype) for a in values],return_mode='file',output_path=str(path),input_names=['images']+list(names))
 session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
 actual=session.run(None,{n.name:np.asarray(v) for n,v in zip(session.get_inputs(),values)})[0];expected=np.asarray(model(x))
 np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=3e-4)
 row={'architecture':name,'maximum_absolute_error':float(abs(actual-expected).max()),'relative_norm_error':float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-12)),'bytes':path.stat().st_size,'weights':'explicit graph inputs; not bundled'};rows.append(row);print(row,flush=True)
 (root/'manifest.json').write_text(json.dumps(rows,indent=2));del model,q,session,values;gc.collect();jax.clear_caches()

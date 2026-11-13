"""Match the likelihood and coordinates, before comparing any optimizer."""
from pathlib import Path
import argparse,json
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();rows=[]
# Predeclared float32 tolerances: probability absolute5e-5, objective relative1e-4;
# aggregated derivative relativeL2 1e-3, with absolute1e-6 for near-zero blocks.
for method in ['mfvi','ep']:
 x=np.load(a.root/(method+'_predictions.npz'));y=np.load(a.root/(method+'_torch_predictions.npz'))
 assert np.array_equal(x['labels'],y['labels'])
 error=np.abs(x['probabilities']-y['probabilities'])
 rows.append(dict(method=method,quantity='probabilities',max_abs=float(error.max()),relative_l2=float(np.linalg.norm(error)/np.linalg.norm(y['probabilities'])),passed=bool(error.max()<5e-5)))
 x=np.load(a.root/(method+'_gradients.npz'));y=np.load(a.root/(method+'_torch_gradients.npz'))
 assert set(x.files)==set(y.files)
 for part in ['objective','mean','variance']:
  keys=['objective'] if part=='objective' else sorted(k for k in x.files if k.endswith('__'+part))
  u=np.concatenate([np.asarray(x[k],np.float64).ravel() for k in keys]);v=np.concatenate([np.asarray(y[k],np.float64).ravel() for k in keys])
  err=u-v;rel=np.linalg.norm(err)/max(np.linalg.norm(v),1e-300)
  limit=1e-4 if part=='objective' else 1e-3
  passed=bool(rel<=limit or np.max(np.abs(err))<=1e-6)
  rows.append(dict(method=method,quantity=part,max_abs=float(np.max(np.abs(err))),relative_l2=float(rel),passed=passed))
report={'scope':'Saved Gaussian A+B beliefs, same base/input/token targets; no training or optimizer.','arithmetic':'float32; highest matmul precision; no TF32','rows':rows,'passed':all(r['passed'] for r in rows)}
(a.root/'native_agreement.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

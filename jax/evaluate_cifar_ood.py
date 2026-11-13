"""Fixed learned beliefs, full SVHN test set; no training or checkpoint selection."""
import os
os.environ.setdefault('XLA_FLAGS','--xla_disable_hlo_passes=constant_folding')
from pathlib import Path
import json,gc
import numpy as np
import jax
from flax import serialization
from bayesify_jax import data
from bayesify_jax.training import Protocol,initialize,make_predictor,predict_array
from bayesify_jax.uncertainty import ood_metrics
x,_=data.svhn();rows=[]
for folder in sorted(Path('results').glob('*net*_seed*')):
    if not (folder/'metrics.json').exists():continue
    cfg=Protocol(**json.loads((folder/'config.json').read_text()))
    if cfg.architecture not in ('resnet20','densenet40'):continue
    model,prior,state,probes=initialize(cfg,45000)
    saved=serialization.from_bytes({'state':state,'epoch':0},(folder/'best.msgpack').read_bytes())
    fn=make_predictor(model,cfg);p=predict_array(fn,saved['state'],x)
    np.savez_compressed(folder/'ood_predictions.npz',probabilities=p)
    idp=np.load(folder/'test_predictions.npz')['probabilities'];scores=ood_metrics(idp,p)
    original=json.loads((folder/'metrics.json').read_text());original.update(scores)
    (folder/'metrics.json').write_text(json.dumps(original,indent=2))
    (folder/'ood_protocol.json').write_text(json.dumps({'ID':'complete CIFAR-10 test','OOD':'complete SVHN test','ID_count':len(idp),'OOD_count':len(p),'preprocessing':'CIFAR training mean/std','selection':'no OOD tuning','checkpoint_epoch':saved['epoch']},indent=2))
    print(folder.name,scores,flush=True);del model,state,saved,fn,p;jax.clear_caches();gc.collect()

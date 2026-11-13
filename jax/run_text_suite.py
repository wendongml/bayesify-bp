"""Full declared text protocols, independent of imported checkpoint evaluation."""
import os
os.environ.setdefault('XLA_FLAGS','--xla_disable_hlo_passes=constant_folding')

import argparse,gc,jax
from bayesify_jax.text_training import train
p=argparse.ArgumentParser();p.add_argument('--tasks',nargs='+',default=['clinc','sst2']);p.add_argument('--methods',nargs='+',default=['ordinary','sampled','hybrid']);p.add_argument('--seed',type=int,default=41)
a=p.parse_args()
for task in a.tasks:
 for method in a.methods:
  train(task,method,a.seed);jax.clear_caches();gc.collect()

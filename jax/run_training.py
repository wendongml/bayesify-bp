"""Example: python run_training.py --architecture mlp --method moment_ep."""
import argparse
from dataclasses import replace
from bayesify_jax.training import protocol,train
p=argparse.ArgumentParser();p.add_argument('--architecture',choices=['mlp','lenet','resnet20','densenet40'],default='mlp')
p.add_argument('--method',choices=['ordinary','sampled_mfvi','moment_mfvi','sampled_ep','moment_ep'],default='moment_mfvi')
p.add_argument('--seed',type=int,default=41);p.add_argument('--epochs',type=int);p.add_argument('--limit-train',type=int,default=0);p.add_argument('--limit-validation',type=int,default=0)
p.add_argument('--dataset',choices=['clean','dirty'],default='clean');p.add_argument('--data',default='data');p.add_argument('--output',default='results');a=p.parse_args()
cfg=protocol(a.architecture,a.method,a.seed);cfg=replace(cfg,limit_train=a.limit_train,limit_validation=a.limit_validation,dataset=a.dataset)
if a.epochs is not None:cfg=replace(cfg,epochs=a.epochs)
train(cfg,a.data,a.output)

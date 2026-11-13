import argparse
from bayesify_jax.text_training import train
p=argparse.ArgumentParser();p.add_argument('--task',choices=['sst2','clinc'],required=True);p.add_argument('--method',choices=['ordinary','sampled','hybrid'],default='hybrid');p.add_argument('--seed',type=int,default=41);p.add_argument('--limit-train',type=int,default=0);a=p.parse_args()
train(a.task,a.method,a.seed,limit_train=a.limit_train)

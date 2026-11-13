"""The original fixed settings; preserve the three-epoch learning-rate schedule.

The tutorial reports the saved FIRST epoch. Writing that snapshot is additional
logging only; it does not change the optimizer, data order or schedule.
"""
from pathlib import Path
import argparse,json
from datasets import load_dataset
from huggingface_hub import snapshot_download
import torch
import train_sampled
from vadam_normalization import VadamLikelihoodEMA
p=argparse.ArgumentParser();p.add_argument('--method',choices=['adamw','sampled_mfvi','moment_ep'],required=True);a=p.parse_args()
root=Path(__file__).resolve().parent
snapshot_download('gpt2',revision='607a30d783dfa663caf39e06633721c8d4cfcd7e',local_dir=root/'gpt2',
                  allow_patterns=['config.json','model.safetensors','vocab.json','merges.txt','tokenizer*'])
data=load_dataset('stanfordnlp/sst2',revision='8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb')
for split in ['train','validation']:
    if not (root/f'sst2_{split}').exists():data[split].save_to_disk(str(root/f'sst2_{split}'))
config=json.loads((root/f'{a.method}.json').read_text())
# A wall-time guard from the old GPU queue is not part of the scientific method.
config['max_hours']=24
if a.method=='sampled_mfvi':train_sampled.Vadam=VadamLikelihoodEMA
train_sampled.run(config,root)

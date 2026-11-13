"""Reconstruct the published MRPC experiment using the authors' model and updates.

Only orchestration is ours: four data-parallel ranks are evaluated serially on
one A100, gradients are averaged, and the original optimizers each step once.
No posterior, likelihood, Flipout, KL, or optimizer formula is replaced.
"""
import os, sys, math, time, json, csv, random, argparse, copy
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, RandomSampler
from accelerate.data_loader import BatchSamplerShard
from accelerate.utils import set_seed
from transformers import RobertaConfig, RobertaForSequenceClassification
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'official'))
from modelwrappers.blob import BLoB, get_parser as blob_parser
from modelwrappers.mle import MLE, get_parser as map_parser

class Local:
    """The small Accelerator interface used by the original constructors/tests."""
    device = torch.device('cuda')
    num_processes = 1
    is_local_main_process = is_main_process = True
    use_fp16 = False
    def wait_for_everyone(self): pass
    def main_process_first(self): return nullcontext()
    def backward(self, loss): loss.backward()
    def gather(self, x): return x


def arguments(method, seed):
    argv = ['--dataset-type','bertds','--dataset','mrpc','--model-type','seqcls',
            '--model','roberta-base','--modelwrapper', 'blob' if method=='blob' else 'mle',
            '--lr','5e-4','--batch-size','32','--opt','adamw','--warmup-ratio','0.06',
            '--max-seq-len','256','--seed',str(seed),'--apply-classhead-lora',
            '--lora-r','8','--lora-alpha','8','--lora-dropout','0',
            '--max-train-steps','5000','--eval-per-steps','6000','--nowand']
    if method=='blob':
        argv += ['--bayes-klreweighting','--bayes-datasetrescaling','--bayes-eps','0.05',
                 '--bayes-beta','0.2','--bayes-gamma','8','--bayes-kllr','0.002',
                 '--bayes-train-n-samples','1','--bayes-eval-n-samples','10',
                 '--bayes-eval-n-samples-final','10']
    else: argv += ['--opt-wd','0.00001']
    args=(blob_parser() if method=='blob' else map_parser()).parse_args(argv)
    args.epoch=0;args.num_workers=0  # Missing upstream CLI attribute; deterministic loader.
    return args


def map_forward(self, batch, sample=True, n_samples=1, **kwargs):
    # Mechanical repair: upstream mistakenly calls self.base_model(**batch.logits).
    return self.base_model(**batch).logits.unsqueeze(1)
MLE.forward_logits = map_forward


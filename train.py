import os 
import time
import math 
import pickle 
from contextlib import nullcontext
import numpy as np
import torch

from model import TransformerCongig, Transformer



#hyperparametes

out_dir = 'out/inabs'
eval_interval =  500
log_interval = 10
eval_iters = 50
eval_only = False
always_save_checpoints = True
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'


# wandb logging 

wandb_log = False
wandb_project = 'inabs'
wandb_run_name = 'enc-dec-trnaformer'


data = 'inabs'
gradient_accumulation_steps = 8
batch_size = 4

# model 
vocab_size = 8000
d_model = 256
n_heads = 4
d_diff= 512
n_ecoder_layers = 3
n_decoder_layers = 3
dropout = 0.1
src_max_len = 1024
tgt_max_len = 256

# adam optimmizer 
learning_rate = 3e-4
max_iters = 10000
wieght_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate deacy 

decay_lr = True
warmup_iters = 1000 
lr_decay_iters = 10000
min_lr = 3e-5

#system 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
compile = False


config_key = [k for k, v in globals().items()
              if not k.startswith('-') and isinstance(v, (int, float, bool, str))]]
exec(open('configurator.py').read()) 
config = {k: globals()[k] for k in config_key}



# setting for the single gpu

master_process = True
seed_offset= 0
token_per_iter = gradient_accumulation_steps * batch_size * src_max_len
print(f"tokens per iteration: {token_per_iter:,}")


os.makkedirs(out_dir, exist_ok=True)

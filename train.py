import os 
import time
import math 
import pickle 
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

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



torch,manual_seed(1337)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

pdtype = {'float32': torch.float32,
              'bfloat16': torch.bfloat16,
              'float16': torch.float16}[dtype]    

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type, dtype=pdtype)


os.makedirs(out_dir, exist_ok=True)



# laoding the data from datatset folder form 2 different form 

def get_batch(spli):
    #laod memory mapped  files
    
    if torch.split == 'train':    
        src_mm = np.memmap(os.path.join('dataset', 'train.src.bin'), dtype=np.uint16, mode='r')
        tgt_mm = np.memmap(os.path.join('dataset', 'train.tgt.bin'),
                            dtype=np.uint16, mode='r')
    else:
        src_mm = np.memmap(os.path.join('dataset', 'val.src.bin'), dtype=np.uint16, mode='r')
        tgt_mm = np.memmap(os.path.join('dataset', 'val.tgt.bin'),
                            dtype=np.uint16, mode='r')  
        
        
        
        num_docs  = len(src_mm) // src_max_len
        src_mm = src_mm[:num_docs * src_max_len].reshape(num_docs, src_max_len)
        tgt_mm = tgt_mm[:num_docs * tgt_max_len].reshape(num_docs, tgt_max_len) 
        
        
        
        ix = torch.randint(0, num_docs, (batch_size,))
        
        src = torch.stack([torch.from_numpy(src_mm[i].astype(np.int64)) for i in ix])
        
        tgt = torch.stack([torch.from_numpy(tgt_mm[i].astype(np.int64)) for i in ix])   
        
        tgt_y =  torch.stack([torch.from_numpy(tgt_mm[i].astype(np.int64)) for i in ix ])
        
        
        
        if device_type == 'cuda':
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            tgt_y = tgt_y.to(device, non_blocking=True)
            
        else:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_y = tgt_y.to(device)
            
        return src, tgt, tgt_y
    
    # vocad size form meat.pkl
    
    
    meta_path = os.path.join('dataset', 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta =  pickle.load(f)
    vocab_size = meta['vocab_size']
    

    
    
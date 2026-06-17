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
    
    meta_path = os.path.join('data_dir', 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta['vocab_size']
        print(f"vocab size = {vocab_size}( read from meta.pkl )")
    else:
        print(f"vocab size = {vocab_size}( hard coded )")
        
        
        
#MODEL INIT

model_args = dict(
    vocab_size = vocab_size,
    d_model = d_model,
    n_heads = n_heads,
    d_diff = d_diff,
    n_encoder_layers = n_ecoder_layers,
    n_decoder_layers = n_decoder_layers,
    dropout = dropout,
    src_max_len = src_max_len,
    tgt_max_len = tgt_max_len
    
)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    cfg  = TransformerCongig(**model_args)
    model = Transformer(cfg)
    
elif init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    print(f"Resuming training from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    
    for k in model_args:
        model_args[k] = checkpoint['model_args'][k]
        
        cfg = TransformerCongig(**model_args)
        model = Transformer(cfg)
        
        
        state_dict = checkpoint['model']
        for k in list(state_dict.keys()):
            if k.startswith('decoder.'):
                state_dict[k[len('decoder.'):]] = state_dict.pop(k)
                
                
        model.load_status_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from iteration {iter_num} with best val loss {best_val_loss}")
        
        
model.to(device)

n_params = sum(p.numel() for p in model.parameters())
print("number of parameters: %.2fM" % (n_params/1e6))


# grad SCALER

scaler  = torch.cuda.amp.GradScaler(enabled=(device_type == 'float16'))



# OPTIMISER 


decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]


print(f"number of decay params: {len(decay_params)}")
print(f"number of nodecay params: {len(nodecay_params)}")

optimiser = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": wieght_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ],
    
    lr=learning_rate, betas=(beta1, beta2)
    
)


if init_from == 'resume':
    optimiser.load_state_dict(checkpoint['optimizer'])
checkpoint = None

if compile:
    print("compiling the model")
    model = torch.compile(model)
    
# LOSS ESTIMATION 


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses  = torch.zeros(eval_iters)
        for k in range(eval_iters):
            src, tgt, tgt_y = get_batch(split)
            with ctx:
                logits, loss = model(src, tgt, tgt_y)
                
            losses[k] = loss.item()
        out[split] = losses.item()
    model.train()
    return out 



# LEARNING RATE SCHEDULE


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * ( it+1) / (warmup_iters+1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff  = 0.5 * ( 1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)



# WANDB 

if wandb_log:
    import wandb 
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    wandb.watch(model, log='all')
    
    
# TRANING LOOP 

item_num = 0  if init_from == 'scratch' else iter_num
best_val_loss = 1e9 if init_from == 'scratch' else best_val_loss


src, tgt, tgt_y = get_batch('train')
t0= time.time()

while True:
    # LR UPDATE
    
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimiser.param_groups:
        param_group['lr'] = lr
        
        
    #EVAL + CHECKPOINTs
    
    if iter_num % eval_interval == 0:
        losses  = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")    
        if wandb_Log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr
            })
            
            
        if losses['val'] < best_val_loss or always_save_checkpoints:
            best=val_loss = losses['val']
            if iter_num > 0:
                ckpt  = {
                    'model' : model.state_dict(),
                    'optimiser': optimiser.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                        
                }
                
                save_path = os.path.join(out_dir, 'ckpt.pt')
                torch.save(ckpt, save_path)
                print(f"Saved checkpoint to {save_path}")
                
                
    if iter_num == 0 and eval_only:
        break
    
    
# FORWARS + BACKWARD 


for micro_step in range(gradient_accumulation_steps):
    with ctx:
        logits, loss = model(src, tgt, tgt_y)
        loss = loss / gradient_accumulation_steps
          
          
    src, tgt, tgt_y = get_batch('train')
    
    scaler.sacler(loss).baclward()
    
# Gradient CLIPPING 

if grad_clip != 0.0:
    scaler.unscale_(optimiser)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    
scaler.step(opitimiser)

t1 = time.time()
dt  = t1- t0
t0 = t1

if iter_num % log_interval == 0:
    lossf  = loss.item() *  gradient_accumulation_steps
    print(f"iter {iter_num}: loss {lossf:.4f}, time per iter {dt*1000:.2f}ms, lr {lr:e}")
    
iter_num += 1

if iter_num > max_iters:
    break
print("Training complete")

    
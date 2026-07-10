import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from model import TransformerConfig, Transformer

# -----------------------------------------------------------------------------
# hyperparameters — defaults, overridden by config/train_inabs.py
out_dir   = 'out-inabs'
init_from = 'scratch'   # 'scratch' or 'resume'

eval_interval          = 500
log_interval           = 10
eval_iters             = 50
eval_only              = False
always_save_checkpoint = True

wandb_log      = False
wandb_project  = 'inabs'
wandb_run_name = 'enc-dec-transformer'

dataset                     = 'inabs'
gradient_accumulation_steps = 8
batch_size                  = 4

# model
vocab_size   = 8000
d_model      = 256
n_heads      = 4
d_ff         = 512
n_enc_layers = 3
n_dec_layers = 3
dropout      = 0.1
src_max_len  = 1024
tgt_max_len  = 256

# optimizer
learning_rate = 3e-4
max_iters     = 10000
weight_decay  = 0.1
beta1         = 0.9
beta2         = 0.95
grad_clip     = 1.0

# lr schedule
decay_lr       = True
warmup_iters   = 400
lr_decay_iters = 10000
min_lr         = 3e-5

# system
device  = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype   = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False

# -----------------------------------------------------------------------------
# configurator — overrides above defaults from config file + command line args
config_keys = [k for k, v in globals().items()
               if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype     = {'float32': torch.float32,
               'bfloat16': torch.bfloat16,
               'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

os.makedirs(out_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# data loading
data_dir  = os.path.join('data', dataset)
meta_path = os.path.join(data_dir, 'meta.pkl')

if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    print(f"vocab_size = {vocab_size} (read from meta.pkl)")
else:
    print(f"vocab_size = {vocab_size} (no meta.pkl found, using default)")


def get_batch(split):
    # file names must match exactly what prepare.py wrote
    src_file = 'train_src.bin' if split == 'train' else 'val_src.bin'
    tgt_file = 'train_tgt.bin' if split == 'train' else 'val_tgt.bin'

    src_mm = np.memmap(os.path.join(data_dir, src_file), dtype=np.uint16, mode='r')
    tgt_mm = np.memmap(os.path.join(data_dir, tgt_file), dtype=np.uint16, mode='r')

    num_docs = len(src_mm) // src_max_len
    src_mm   = src_mm[:num_docs * src_max_len].reshape(num_docs, src_max_len)
    tgt_mm   = tgt_mm[:num_docs * tgt_max_len].reshape(num_docs, tgt_max_len)

    ix = torch.randint(0, num_docs, (batch_size,))

    src   = torch.stack([torch.from_numpy(src_mm[i].astype(np.int64))       for i in ix])
    # teacher forcing: decoder input = all tokens except last
    tgt   = torch.stack([torch.from_numpy(tgt_mm[i, :-1].astype(np.int64)) for i in ix])
    # decoder target = all tokens except first (shifted by 1)
    tgt_y = torch.stack([torch.from_numpy(tgt_mm[i, 1:].astype(np.int64))  for i in ix])

    if device_type == 'cuda':
        src   = src.pin_memory().to(device, non_blocking=True)
        tgt   = tgt.pin_memory().to(device, non_blocking=True)
        tgt_y = tgt_y.pin_memory().to(device, non_blocking=True)
    else:
        src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)

    return src, tgt, tgt_y


# -----------------------------------------------------------------------------
# model init
model_args = dict(
    vocab_size   = vocab_size,
    d_model      = d_model,
    n_heads      = n_heads,
    d_ff         = d_ff,
    n_enc_layers = n_enc_layers,
    n_dec_layers = n_dec_layers,
    dropout      = dropout,
    src_max_len  = src_max_len,
    tgt_max_len  = tgt_max_len,
)

if init_from == 'scratch':
    print("initializing new model from scratch")
    cfg   = TransformerConfig(**model_args)
    model = Transformer(cfg)
    iter_num      = 0
    best_val_loss = 1e9

elif init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    print(f"resuming from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    for k in model_args:
        model_args[k] = checkpoint['model_args'][k]

    cfg   = TransformerConfig(**model_args)
    model = Transformer(cfg)

    state_dict = checkpoint['model']
    for k in list(state_dict.keys()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    iter_num      = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    print(f"resumed at iter {iter_num}, best val loss {best_val_loss:.4f}")

else:
    raise ValueError(f"unknown init_from: {init_from}")

model.to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"model parameters: {n_params/1e6:.2f}M")

# -----------------------------------------------------------------------------
# grad scaler — only active for float16
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer — weight decay only on 2D params
decay_params   = [p for n, p in model.named_parameters() if p.dim() >= 2]
nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
print(f"decay params: {sum(p.numel() for p in decay_params):,}")
print(f"no-decay params: {sum(p.numel() for p in nodecay_params):,}")

optimizer = torch.optim.AdamW(
    [
        {'params': decay_params,   'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ],
    lr=learning_rate,
    betas=(beta1, beta2),
)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free memory

if compile:
    print("compiling model...")
    model = torch.compile(model)


# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            src, tgt, tgt_y = get_batch(split)
            with ctx:
                logits, loss = model(src, tgt, tgt_y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it):
    # linear warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # past decay → min_lr
    if it > lr_decay_iters:
        return min_lr
    # cosine decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# -----------------------------------------------------------------------------
# training loop
src, tgt, tgt_y = get_batch('train')
t0 = time.time()

while True:

    # set lr
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # eval + checkpoint
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if wandb_log:
            wandb.log({"iter": iter_num, "train/loss": losses['train'],
                       "val/loss": losses['val'], "lr": lr})

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                ckpt = {
                    'model':         model.state_dict(),
                    'optimizer':     optimizer.state_dict(),
                    'model_args':    model_args,
                    'iter_num':      iter_num,
                    'best_val_loss': best_val_loss,
                    'config':        config,
                }
                save_path = os.path.join(out_dir, 'ckpt.pt')
                torch.save(ckpt, save_path)
                print(f"checkpoint saved → {save_path}")

    if iter_num == 0 and eval_only:
        break

    # forward + backward with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(src, tgt, tgt_y)
            loss = loss / gradient_accumulation_steps
        src, tgt, tgt_y = get_batch('train')
        scaler.scale(loss).backward()

    # gradient clip
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f} | time {dt*1000:.1f}ms | lr {lr:.2e}")

    iter_num += 1

    if iter_num > max_iters:
        break

print("training complete.")
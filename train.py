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
# FIX: was 'TransformerCongig' -- model.py actually defines 'TransformerConfig'
# (confirmed by the ImportError), so this was a genuine typo, not a matching name.

# -----------------------------------------------------------------------------
# hyperparameters
out_dir = 'out/inabs'
eval_interval = 500
log_interval = 10
eval_iters = 50
eval_only = False
always_save_checkpoints = True   # FIX: was 'always_save_checpoints' but used
                                  # later in the file as 'always_save_checkpoints'
                                  # -> NameError. Now the name is consistent.
init_from = 'scratch'  # 'scratch' or 'resume'

# wandb logging
wandb_log = False
wandb_project = 'inabs'
wandb_run_name = 'enc-dec-transformer'

dataset = 'inabs'
gradient_accumulation_steps = 8
batch_size = 4

# model
vocab_size = 8000
d_model = 256
n_heads = 4
d_ff = 512
n_encoder_layers = 3
n_decoder_layers = 3
dropout = 0.1
src_max_len = 1024
tgt_max_len = 256

# adam optimizer
learning_rate = 3e-4
max_iters = 10000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay
decay_lr = True
warmup_iters = 1000
lr_decay_iters = 10000
min_lr = 3e-5

# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
compile = False

config_keys = [k for k, v in globals().items()
               if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

torch.manual_seed(1337)   # FIX: was 'torch,manual_seed(1337)' (comma instead of
                           # dot) -> this actually called an undefined function
                           # 'manual_seed' and built a tuple, raising NameError.

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

ptdtype = {'float32': torch.float32,
           'bfloat16': torch.bfloat16,
           'float16': torch.float16}[dtype]

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type, dtype=ptdtype)

os.makedirs(out_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# data loading
# NOTE: your original had two different literal folder names ('dataset' for
# the .bin files and 'data_dir' for meta.pkl) which almost certainly pointed
# at the wrong place for one of them. Unified both under a single directory
# built from the `dataset` config var -- adjust this to match wherever your
# prep pipeline actually wrote train.src.bin / train.tgt.bin / val.src.bin /
# val.tgt.bin / meta.pkl.
data_dir = os.path.join('data', dataset)

meta_path = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    print(f"vocab_size = {vocab_size} (read from meta.pkl)")
else:
    print(f"vocab_size = {vocab_size} (hard-coded, no meta.pkl found)")


def get_batch(split):
    # FIX: original signature was 'def get_batch(spli):' (typo'd, unused
    # parameter) and the body checked 'if torch.split == 'train':' -- torch.split
    # is a real torch function, so that comparison was always False and the
    # function *always* fell into the 'val' branch regardless of what was
    # requested. The 'train' branch also never reached a return statement, so
    # calling get_batch('train') would have returned None, None, None.
    src_file = 'train.src.bin' if split == 'train' else 'val.src.bin'
    tgt_file = 'train.tgt.bin' if split == 'train' else 'val.tgt.bin'

    src_mm = np.memmap(os.path.join(data_dir, src_file), dtype=np.uint16, mode='r')
    tgt_mm = np.memmap(os.path.join(data_dir, tgt_file), dtype=np.uint16, mode='r')

    num_docs = len(src_mm) // src_max_len
    src_mm = src_mm[:num_docs * src_max_len].reshape(num_docs, src_max_len)
    tgt_mm = tgt_mm[:num_docs * tgt_max_len].reshape(num_docs, tgt_max_len)

    ix = torch.randint(0, num_docs, (batch_size,))

    src = torch.stack([torch.from_numpy(src_mm[i].astype(np.int64)) for i in ix])
    tgt = torch.stack([torch.from_numpy(tgt_mm[i].astype(np.int64)) for i in ix])
    tgt_y = torch.stack([torch.from_numpy(tgt_mm[i].astype(np.int64)) for i in ix])
    # NOTE (not fixed, just flagging): tgt and tgt_y are built identically here,
    # both the raw target sequence. For teacher forcing you'd usually want
    # tgt = decoder input (e.g. <bos> + tokens[:-1]) and tgt_y = shifted labels
    # (tokens[1:] + <eos>) -- unless your Transformer.forward() already does
    # that shift internally. Worth double-checking against model.py.

    if device_type == 'cuda':
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        tgt_y = tgt_y.to(device, non_blocking=True)
    else:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_y = tgt_y.to(device)

    return src, tgt, tgt_y


# -----------------------------------------------------------------------------
# model init
model_args = dict(
    vocab_size=vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    d_ff=d_ff,
    n_encoder_layers=n_encoder_layers,
    n_decoder_layers=n_decoder_layers,
    dropout=dropout,
    src_max_len=src_max_len,
    tgt_max_len=tgt_max_len,
)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    cfg = TransformerConfig(**model_args)
    model = Transformer(cfg)
    iter_num = 0
    best_val_loss = 1e9
elif init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    print(f"Resuming training from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # FIX: this whole block (cfg/model/state_dict creation) was previously
    # nested *inside* the 'for k in model_args' loop, so it re-ran (and
    # re-created the model from scratch) once per key in model_args. It only
    # needs to happen once, after the loop finishes overwriting model_args.
    for k in model_args:
        model_args[k] = checkpoint['model_args'][k]

    cfg = TransformerConfig(**model_args)
    model = Transformer(cfg)

    state_dict = checkpoint['model']
    for k in list(state_dict.keys()):
        if k.startswith('decoder.'):
            state_dict[k[len('decoder.'):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)   # FIX: was 'load_status_dict' (typo, no such method)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    print(f"Resumed from iteration {iter_num} with best val loss {best_val_loss}")
else:
    raise ValueError(f"unknown init_from: {init_from}")

model.to(device)

n_params = sum(p.numel() for p in model.parameters())
print("number of parameters: %.2fM" % (n_params / 1e6,))

# grad scaler
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
# FIX: was 'enabled=(device_type == 'float16')' -- device_type is only ever
# 'cuda' or 'cpu', so that condition could never be true and the scaler was
# silently disabled even when you actually wanted float16 mixed precision.

# optimizer
decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]

print(f"number of decay params: {len(decay_params)}")
print(f"number of nodecay params: {len(nodecay_params)}")

optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ],
    lr=learning_rate, betas=(beta1, beta2),
)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free up memory

if compile:
    print("compiling the model")
    model = torch.compile(model)


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
        out[split] = losses.mean().item()   # FIX: was 'losses.item()', which only
                                             # works on a single-element tensor and
                                             # would raise on eval_iters > 1.
    model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    wandb.watch(model, log='all')

# -----------------------------------------------------------------------------
# training loop
src, tgt, tgt_y = get_batch('train')
t0 = time.time()

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:   # FIX: was 'wandb_Log' (wrong case) -> NameError
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
            })

        if losses['val'] < best_val_loss or always_save_checkpoints:
            best_val_loss = losses['val']   # FIX: was 'best=val_loss = losses['val']',
                                             # which is a chained assignment that set
                                             # two *different* variables ('best' and
                                             # 'val_loss') instead of 'best_val_loss'.
            if iter_num > 0:
                ckpt = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),   # FIX: key now matches
                                                            # what's read back on resume
                                                            # (was saved as 'optimiser'
                                                            # but loaded as 'optimizer').
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

    # FIX: everything from here down used to sit outside the 'while True:' loop
    # at top-level indentation. That meant: (a) it only ever ran once, after the
    # loop somehow ended, instead of once per iteration, and (b) since iter_num
    # was never incremented *inside* the loop, 'while True' had no way to exit
    # on its own -- an infinite loop that never actually trained anything.
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(src, tgt, tgt_y)
            loss = loss / gradient_accumulation_steps
        src, tgt, tgt_y = get_batch('train')
        scaler.scale(loss).backward()   # FIX: was 'scaler.sacler(loss).baclward()' (typos)

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)   # FIX: was 'scaler.step(opitimiser)' -- undefined name, NameError
    scaler.update()          # FIX: was missing entirely -- required after scaler.step()
    optimizer.zero_grad(set_to_none=True)   # FIX: was missing entirely -- without this,
                                             # gradients from every micro-step and every
                                             # iteration just keep accumulating forever.

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time per iter {dt*1000:.2f}ms, lr {lr:e}")

    iter_num += 1

    if iter_num > max_iters:
        break

print("Training complete")
# config/train_inabs.py
# run with: python train.py config/train_inabs.py

# I/O
out_dir   = 'out-inabs'
init_from = 'scratch'

# eval / logging
eval_interval = 500
log_interval  = 10
eval_iters    = 50
eval_only     = False
always_save_checkpoint = True

# data
dataset = 'inabs'

# batch
batch_size                  = 4
gradient_accumulation_steps = 8

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
weight_decay  = 1e-1
beta1         = 0.9
beta2         = 0.95
grad_clip     = 1.0

# lr schedule
decay_lr       = True
warmup_iters   = 400
lr_decay_iters = 10000
min_lr         = 3e-5

# system
compile = False
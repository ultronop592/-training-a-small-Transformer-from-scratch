out_dir    = 'out-inabs'
dataset    = 'inabs'

eval_interval = 500
log_interval  = 10
eval_iters    = 50

batch_size                  = 4
gradient_accumulation_steps = 8

vocab_size   = 8000
d_model      = 256
n_heads      = 4
d_ff         = 512
n_enc_layers = 3
n_dec_layers = 3
dropout      = 0.1
src_max_len  = 1024
tgt_max_len  = 256

learning_rate  = 3e-4
max_iters      = 10000
weight_decay   = 1e-1
warmup_iters   = 400
lr_decay_iters = 10000
min_lr         = 3e-5
compile        = False
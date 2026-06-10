# model 

from sympy import false


from sympy import false


vocal_size  = 8000
d_model = 256
heads = 4
d_ff = 512
n_enc_layers = 3
n_dec_layers = 3
dropout = 0.1


# sequence 

src_max_len = 1024
tgt_max_len = 256

#training
batch_size =  4
grad_accum = 8
learning_rate = 3e-4
max_ters = 100000
eval_interval =  500
eval_iters = 50


# system 
device = 'cuda'
dtype = 'float16'
compile = false


out_dir = 'out/inabs'
log_dir = 'logs/inabs'

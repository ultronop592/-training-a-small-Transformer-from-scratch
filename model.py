# MODEL ENCODER - DECODER TRANSFORMER 

import math 
from dataclasses import dataclass
from networkx import config
import torch
from torch import nn
from torch.nn import functional as F

@dataclass 
class TransformerConfig:
    vocal_size: int =8000
    d_model: int = 256
    n_heads: int = 4
    d_diff : int = 512
    n_encoder_layers: int = 3
    n_decoder_layers: int = 3
    dropout: float = 0.1
    src_max_len: int = 1024
    tgt_max_len: int = 256
    


#SINGLE ATTENTION HEAD 


class Head(nn.Module):
    
    def __init__(self, config, head_size, causal=False):
        super().__init__()
        self.causal = causal
        d_model = config.d_model
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
        
        if causal:
            max_lem = config.src_max_len
            self.register_buffer('tril', torch.tril(torch.ones(max_lem, max_lem)))
            
     
    def forward(self, x , context  = None, key_padding_mask = None):
        if context is None:
            context = x
            
        B, T, _ = x.shape
        _, S, _ = context.shape
        
        q= self.query(x)
        k= self.key(context)
        v= self.value(context)
        
        
        head_size = q.shape[-1]
        wei = q @ k.transpose(-2, -1) * head_size ** -0.5
        
        if self.causal:
            wei  = wei.masked_fill(self.tril[:T, :S] == 0, float('-inf'))
        if key_padding_mask is not None:
            wei = wei.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        out =  wei @ v
        return out
    
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, congig, causal=False):
        super().__init__()
        self.causal = causal
        d_model = config.d_model
        n_heads = config.n_heads
        head_size = d_model // n_heads
        
        self.heads = nn.ModuleList([Head(config, head_size, causal) for _ in range(n_heads)])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        
    def  forward (self, x, context = None, key_padding_mask = None):
        out = torch.cat([h(x, context, key_padding_mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
    
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config.d_model
        d_diff = config.d_diff
        self.net = nn.Sequential(
            nn.Linear(d_model, d_diff),
            nn.ReLU(),
            nn.Linear(d_diff, d_model),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, x):
        return self.net(x)
    
    
    
class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(config, causal=False)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
    def forward(self, x, key_padding_mask = None):
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.ffwd(self.ln2(x))
        return x
    
    
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding =  nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding  = nn.Module(congig.tgt_max_len, config.d_model)
        self.dropout= nn .Dropout(config.dropout)
        self.blocks = nn.ModuleList([Decode(config) for _ in range(config.n_decoder_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        
        self._lm_head= nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        
    def forward(self, tgt_ids, enc_out, src_padding_maxk = None):
        B,T  =tgt_ids.shape
        device  = tgt_ids.device
        
        tok_emb  =  self.token_embedding(tgt_ids)
        pos_emb = self.pos_embedding(torch.arange(T, device=device))
        x  = self.dropout(tok_emb + pos_emb)
        
        for block in self.blocks:
            x  = block(x, enc_out, src_padding_mask)
            
        x  = self.ln_f(x)
        logits  = self._lm_head(x) 
        return logits
    
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config  = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
 
        
        self.decoder.lm_head.weight = self.decoder.token_embedding.weight
 
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Transformer initialized: {n_params/1e6:.2f}M parameters")
 
    def forward(self, src_ids, tgt_ids, tgt_y=None, pad_id=0):
       
        src_padding_mask = (src_ids == pad_id)   # (B, T_src), dtype=bool
 
        enc_out = self.encoder(src_ids, src_padding_mask=src_padding_mask)       
        logits  = self.decoder(tgt_ids, enc_out, src_padding_mask=src_padding_mask) 
 
        loss = None
        if tgt_y is not None:
            B, T, C = logits.shape
            # flatten batch and time dims, same as Karpathy's bigram.py loss computation
            logits_flat = logits.view(B * T, C)
            targets_flat = tgt_y.view(B * T)
            # ignore_index=pad_id: don't penalize the model for padding positions
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_id)
 
        return logits, loss
 
    @torch.no_grad()
    def generate(self, src_ids, bos_id, eos_id, pad_id=0, max_new_tokens=256):
        
        self.eval()
 
        src_padding_mask = (src_ids == pad_id)
        enc_out = self.encoder(src_ids, src_padding_mask=src_padding_mask)   # encode ONCE, reuse every step
 
        B = src_ids.shape[0]
        device = src_ids.device
 
        # start each sequence with the BOS (beginning-of-summary) token
        tgt_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
 
        for _ in range(max_new_tokens):
            logits = self.decoder(tgt_ids, enc_out, src_padding_mask=src_padding_mask)  # (B, T, vocab_size)
            logits = logits[:, -1, :]                                                    # only need the LAST step's prediction
            probs  = F.softmax(logits, dim=-1)
            next_id = torch.argmax(probs, dim=-1, keepdim=True)   # greedy: pick most likely token
            tgt_ids = torch.cat([tgt_ids, next_id], dim=1)
 
            # stop early if every sequence in the batch has generated EOS
            if (next_id == eos_id).all():
                break
 
        self.train()
        return tgt_ids
        
    
        
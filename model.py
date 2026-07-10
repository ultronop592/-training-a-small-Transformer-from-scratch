"""
model.py — Encoder-Decoder Transformer for Indian Legal Summarization
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class TransformerConfig:
    vocab_size:   int   = 8000   # fixed: was 'vocal_size'
    d_model:      int   = 256
    n_heads:      int   = 4
    d_ff:         int   = 512    # fixed: was 'd_diff'
    n_enc_layers: int   = 3      # fixed: was 'n_encoder_layers'
    n_dec_layers: int   = 3      # fixed: was 'n_decoder_layers'
    dropout:      float = 0.1
    src_max_len:  int   = 1024
    tgt_max_len:  int   = 256


# =============================================================================
# SINGLE ATTENTION HEAD
# =============================================================================
class Head(nn.Module):
    def __init__(self, config, head_size, causal=False):
        super().__init__()
        self.causal = causal
        self.key    = nn.Linear(config.d_model, head_size, bias=False)
        self.query  = nn.Linear(config.d_model, head_size, bias=False)
        self.value  = nn.Linear(config.d_model, head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        if causal:
            # causal mask for decoder self-attention
            # fixed: was using src_max_len for decoder — must use tgt_max_len
            max_len = config.tgt_max_len
            self.register_buffer('tril', torch.tril(torch.ones(max_len, max_len)))

    def forward(self, x, context=None, key_padding_mask=None):
        if context is None:
            context = x

        B, T, _ = x.shape
        _, S, _ = context.shape

        q = self.query(x)        # (B, T, head_size)
        k = self.key(context)    # (B, S, head_size)
        v = self.value(context)  # (B, S, head_size)

        head_size = q.shape[-1]
        wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)  # (B, T, S)

        if self.causal:
            # fixed: was [:T, :S] which is wrong for self-attention — must be [:T, :T]
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        if key_padding_mask is not None:
            # fixed: was using 4D indexing — key_padding_mask is (B, S), unsqueeze to (B, 1, S)
            wei = wei.masked_fill(key_padding_mask.unsqueeze(1), float('-inf'))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v  # (B, T, head_size)


# =============================================================================
# MULTI-HEAD ATTENTION
# =============================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, config, causal=False):  # fixed: was 'congig' typo
        super().__init__()
        d_model   = config.d_model
        n_heads   = config.n_heads
        head_size = d_model // n_heads
        assert d_model % n_heads == 0

        self.heads   = nn.ModuleList([Head(config, head_size, causal) for _ in range(n_heads)])
        self.proj    = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, context=None, key_padding_mask=None):
        out = torch.cat([h(x, context, key_padding_mask) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


# =============================================================================
# FEEDFORWARD
# =============================================================================
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),  # fixed: was config.d_diff
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),  # fixed: was config.d_diff
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# ENCODER BLOCK
# =============================================================================
class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config, causal=False)
        self.ln2  = nn.LayerNorm(config.d_model)
        self.ff   = FeedForward(config)

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.ff(self.ln2(x))
        return x


# =============================================================================
# DECODER BLOCK
# =============================================================================
class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1        = nn.LayerNorm(config.d_model)
        self.self_attn  = MultiHeadAttention(config, causal=True)   # causal self-attention
        self.ln2        = nn.LayerNorm(config.d_model)
        self.cross_attn = MultiHeadAttention(config, causal=False)  # cross-attention
        self.ln3        = nn.LayerNorm(config.d_model)
        self.ff         = FeedForward(config)

    def forward(self, x, enc_out, src_padding_mask=None):
        x = x + self.self_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), context=enc_out, key_padding_mask=src_padding_mask)
        x = x + self.ff(self.ln3(x))
        return x


# =============================================================================
# ENCODER
# =============================================================================
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding   = nn.Embedding(config.src_max_len, config.d_model)
        self.dropout         = nn.Dropout(config.dropout)
        self.blocks          = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_enc_layers)])
        self.ln_f            = nn.LayerNorm(config.d_model)

    def forward(self, src_ids, src_padding_mask=None):
        B, T = src_ids.shape
        device = src_ids.device

        tok_emb = self.token_embedding(src_ids)
        pos_emb = self.pos_embedding(torch.arange(T, device=device))
        x = self.dropout(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x, key_padding_mask=src_padding_mask)

        return self.ln_f(x)


# =============================================================================
# DECODER
# =============================================================================
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding   = nn.Embedding(config.tgt_max_len, config.d_model)  # fixed: was nn.Module
        self.dropout         = nn.Dropout(config.dropout)
        self.blocks          = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_dec_layers)])  # fixed: was Decode + n_decoder_layers
        self.ln_f            = nn.LayerNorm(config.d_model)
        self.lm_head         = nn.Linear(config.d_model, config.vocab_size, bias=False)  # fixed: was _lm_head

    def forward(self, tgt_ids, enc_out, src_padding_mask=None):  # fixed: was src_padding_maxk typo
        B, T = tgt_ids.shape
        device = tgt_ids.device

        tok_emb = self.token_embedding(tgt_ids)
        pos_emb = self.pos_embedding(torch.arange(T, device=device))
        x = self.dropout(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x, enc_out, src_padding_mask)

        x = self.ln_f(x)
        return self.lm_head(x)


# =============================================================================
# TRANSFORMER
# =============================================================================
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config  = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # weight tying — lm_head shares weights with decoder token embedding
        self.decoder.lm_head.weight = self.decoder.token_embedding.weight

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Transformer initialized: {n_params/1e6:.2f}M parameters")

    def forward(self, src_ids, tgt_ids, tgt_y=None, pad_id=0):
        src_padding_mask = (src_ids == pad_id)  # (B, T_src)

        enc_out = self.encoder(src_ids, src_padding_mask=src_padding_mask)
        logits  = self.decoder(tgt_ids, enc_out, src_padding_mask=src_padding_mask)

        loss = None
        if tgt_y is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, C),
                tgt_y.view(B * T),
                ignore_index=pad_id,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, src_ids, bos_id, eos_id, pad_id=0, max_new_tokens=256):
        self.eval()

        src_padding_mask = (src_ids == pad_id)
        enc_out = self.encoder(src_ids, src_padding_mask=src_padding_mask)

        B      = src_ids.shape[0]
        device = src_ids.device
        tgt_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            logits  = self.decoder(tgt_ids, enc_out, src_padding_mask=src_padding_mask)
            logits  = logits[:, -1, :]
            next_id = torch.argmax(F.softmax(logits, dim=-1), dim=-1, keepdim=True)
            tgt_ids = torch.cat([tgt_ids, next_id], dim=1)
            if (next_id == eos_id).all():
                break

        self.train()
        return tgt_ids
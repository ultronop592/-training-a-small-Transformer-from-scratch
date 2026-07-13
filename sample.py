

import os
import sys
import pickle
import argparse
import glob

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import TransformerConfig, Transformer



# ARGUMENTS


parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint',        type=str,   default='out-inabs/ckpt.pt')
parser.add_argument('--judgment',          type=str,   default=None)
parser.add_argument('--max_new_tokens',    type=int,   default=256)
parser.add_argument('--device',            type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')


parser.add_argument('--beam_size',         type=int,   default=4,
                    help='beam search width (1 = greedy, 4 = recommended)')
parser.add_argument('--top_k',             type=int,   default=0,
                    help='top-k sampling (0 = disabled, use beam search instead)')
parser.add_argument('--temperature',       type=float, default=1.0,
                    help='sampling temperature (lower = more focused)')

parser.add_argument('--rep_penalty',       type=float, default=1.3,
                    help='repetition penalty (1.0 = off, 1.3 = recommended)')


parser.add_argument('--length_penalty',    type=float, default=0.8,
                    help='beam search length penalty (>1 = longer, <1 = shorter)')
parser.add_argument('--min_length',        type=int,   default=30,
                    help='minimum summary length in tokens')

args = parser.parse_args()




# LOAD CHECKPOINT

print(f"loading checkpoint: {args.checkpoint}")

if not os.path.exists(args.checkpoint):
    print(f"ERROR: checkpoint not found at {args.checkpoint}")
    sys.exit(1)

checkpoint = torch.load(args.checkpoint, map_location=args.device)
model_args = checkpoint['model_args']

cfg   = TransformerConfig(**model_args)
model = Transformer(cfg)

state_dict = checkpoint['model']
for k in list(state_dict.keys()):
    if k.startswith('_orig_mod.'):
        state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.to(args.device)
model.eval()

print(f"checkpoint loaded (iter {checkpoint['iter_num']}, "
      f"best val loss {checkpoint['best_val_loss']:.4f})")



# LOAD TOKENIZER + META

data_dir  = os.path.join('data', 'inabs')
meta_path = os.path.join(data_dir, 'meta.pkl')

if not os.path.exists(meta_path):
    print(f"ERROR: meta.pkl not found at {meta_path}")
    sys.exit(1)

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

pad_id      = meta['pad_id']
bos_id      = meta['bos_id']
eos_id      = meta['eos_id']
src_max_len = meta['src_max_len']

tokenizer = Tokenizer.from_file(meta['tokenizer_path'])
print(f"tokenizer loaded (vocab_size={tokenizer.get_vocab_size()})")


# LOAD JUDGMENT

if args.judgment:
    judgment_path = args.judgment
else:
    test_dir   = os.path.join('Dataset', 'train-data', 'judgement')
    test_files = sorted(glob.glob(os.path.join(test_dir, '*.txt')))
    if not test_files:
        print(f"ERROR: no .txt files in {test_dir}")
        sys.exit(1)
    judgment_path = test_files[0]
    print(f"no --judgment given, using: {judgment_path}")

with open(judgment_path, 'r', encoding='utf-8', errors='replace') as f:
    judgment_text = f.read().strip()

print(f"\n{'='*60}")
print("INPUT JUDGMENT (first 500 chars):")
print(judgment_text[:500])
print(f"{'='*60}\n")


# ENCODE

enc     = tokenizer.encode(judgment_text, add_special_tokens=False)
src_ids = enc.ids[:src_max_len]
src_ids = src_ids + [pad_id] * (src_max_len - len(src_ids))
src     = torch.tensor([src_ids], dtype=torch.long, device=args.device)



# HELPER — REPETITION PENALTY



def apply_repetition_penalty(logits, generated_ids, penalty):
    
    if penalty == 1.0:
        return logits
    for token_id in set(generated_ids):
        if logits[0, token_id] > 0:
            logits[0, token_id] /= penalty
        else:
            logits[0, token_id] *= penalty
    return logits



# DECODING STRATEGY 1 — BEAM SEARCH


def beam_search(src, enc_out, src_padding_mask):
    
    
    B         = src.shape[0]   
    beam_size = args.beam_size
    device    = src.device


    beams = [(0.0, [bos_id])]   
    completed = []              

    for step in range(args.max_new_tokens):
        all_candidates = []

        for score, tokens in beams:
            if tokens[-1] == eos_id:
                all_candidates.append((score, tokens))
                continue

    
            tgt_tensor = torch.tensor([tokens], dtype=torch.long, device=device)

            with torch.no_grad():
                logits = model.decoder(tgt_tensor, enc_out,src_padding_mask=src_padding_mask)
            logits = logits[:, -1, :]  

        
            logits = apply_repetition_penalty(logits, tokens, args.rep_penalty)
            if len(tokens) < args.min_length:
                logits[0, eos_id] = float('-inf')

            if args.temperature != 1.0:
                logits = logits / args.temperature

            log_probs = F.log_softmax(logits, dim=-1)  

            top_log_probs, top_ids = torch.topk(log_probs, beam_size, dim=-1)

            for i in range(beam_size):
                next_token    = top_ids[0, i].item()
                next_log_prob = top_log_probs[0, i].item()
                new_score     = score + next_log_prob
                new_tokens    = tokens + [next_token]
                all_candidates.append((new_score, new_tokens))

        def length_normalized_score(candidate):
            sc, toks = candidate
            length = max(len(toks), 1)
            return sc / (length ** args.length_penalty)

        all_candidates.sort(key=length_normalized_score, reverse=True)
        beams = all_candidates[:beam_size]

        still_running = []
        for sc, toks in beams:
            if toks[-1] == eos_id:
                completed.append((sc, toks))
            else:
                still_running.append((sc, toks))

        beams = still_running

        if len(beams) == 0:
            break

    completed.extend(beams)
    completed.sort(key=lambda x: x[0] / max(len(x[1]), 1) ** args.length_penalty,
                   reverse=True)
    best_tokens = completed[0][1]
    return best_tokens



# DECODING STRATEGY 2 — TOP-K SAMPLING WITH REPETITION PENALTY

def topk_sample(src, enc_out, src_padding_mask):
    tgt_ids = [bos_id]
    device  = src.device

    for _ in range(args.max_new_tokens):
        tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model.decoder(tgt_tensor, enc_out,
                                   src_padding_mask=src_padding_mask)
        logits = logits[:, -1, :]

        
        logits = apply_repetition_penalty(logits, tgt_ids, args.rep_penalty)

        
        if len(tgt_ids) < args.min_length:
            logits[0, eos_id] = float('-inf')

        logits = logits / args.temperature

        top_k_logits, _ = torch.topk(logits, args.top_k, dim=-1)
        logits = logits.masked_fill(logits < top_k_logits[:, -1:], float('-inf'))

        probs   = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        tgt_ids.append(next_id)

        if next_id == eos_id:
            break

    return tgt_ids




# GENERATE

print(f"generating summary (beam_size={args.beam_size}, "
      f"rep_penalty={args.rep_penalty}, "
      f"length_penalty={args.length_penalty})...")

src_padding_mask = (src == pad_id)

with torch.no_grad():
    enc_out = model.encoder(src, src_padding_mask=src_padding_mask)

if args.top_k > 0:
    generated = topk_sample(src, enc_out, src_padding_mask)
else:
    generated = beam_search(src, enc_out, src_padding_mask)




# DECODE AND PRINT




if generated and generated[0] == bos_id:
    generated = generated[1:]

if eos_id in generated:
    generated = generated[:generated.index(eos_id)]
generated = [t for t in generated if t != pad_id]

summary = tokenizer.decode(generated)

print(f"\n{'='*60}")
print("GENERATED SUMMARY:")
print(f"{'='*60}")
print(summary)
print(f"{'='*60}")


# COMPARE WITH REFERENCE

fname    = os.path.basename(judgment_path)
ref_path = os.path.join('Dataset', 'train-data', 'summary', fname)

if os.path.exists(ref_path):
    with open(ref_path, 'r', encoding='utf-8', errors='replace') as f:
        ref_summary = f.read().strip()
    print("\nREFERENCE SUMMARY:")
    print(f"{'='*60}")
    print(ref_summary)
    print(f"{'='*60}")
    print("\n(compare the two to judge model quality)")
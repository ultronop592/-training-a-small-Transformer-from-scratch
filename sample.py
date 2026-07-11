
import os
import sys
import pickle
import argparse
import glob

import torch
from tokenizers import Tokenizer

from model import TransformerConfig, Transformer


# ARGUMENTS

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint',     type=str, default='out-inabs/ckpt.pt')
parser.add_argument('--judgment',       type=str, default=None)
parser.add_argument('--max_new_tokens', type=int, default=256)
parser.add_argument('--top_k',         type=int, default=1)
parser.add_argument('--device',        type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')

args = parser.parse_args()

# LOAD CHECKPOINT

print(f"loading checkpoint from: {args.checkpoint}")

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


# LOAD TOKENIZER + METa

data_dir  = os.path.join('data', 'inabs')
meta_path = os.path.join(data_dir, 'meta.pkl')

if not os.path.exists(meta_path):
    print(f"ERROR: meta.pkl not found at {meta_path}")
    print("Run: python data\\inabs\\prepare.py")
    sys.exit(1)

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

pad_id      = meta['pad_id']
bos_id      = meta['bos_id']
eos_id      = meta['eos_id']
src_max_len = meta['src_max_len']

tokenizer = Tokenizer.from_file(meta['tokenizer_path'])
print(f"tokenizer loaded (vocab_size={tokenizer.get_vocab_size()})")


# LOAD JUDGMENT TEXT

if args.judgment:
    judgment_path = args.judgment
else:
    test_dir   = os.path.join('Dataset', 'train-data', 'judgement')
    test_files = sorted(glob.glob(os.path.join(test_dir, '*.txt')))
    if not test_files:
        print(f"ERROR: no .txt files found in {test_dir}")
        sys.exit(1)
    judgment_path = test_files[0]
    print(f"no --judgment given, using: {judgment_path}")

with open(judgment_path, 'r', encoding='utf-8', errors='replace') as f:
    judgment_text = f.read().strip()

print(f"\n{'='*60}")
print(f"INPUT JUDGMENT (first 500 chars):")
print(judgment_text[:500])
print(f"{'='*60}\n")

# ENCODE


enc     = tokenizer.encode(judgment_text, add_special_tokens=False)
src_ids = enc.ids

if len(src_ids) > src_max_len:
    src_ids = src_ids[:src_max_len]

src_ids = src_ids + [pad_id] * (src_max_len - len(src_ids))
src     = torch.tensor([src_ids], dtype=torch.long, device=args.device)

#
# GENERATE

print("generating summary...")

with torch.no_grad():
    if args.top_k == 1:
        # greedy
        output_ids = model.generate(
            src,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        # top-k sampling
        src_padding_mask = (src == pad_id)
        enc_out = model.encoder(src, src_padding_mask=src_padding_mask)
        tgt_ids = torch.full((1, 1), bos_id, dtype=torch.long, device=args.device)

        for _ in range(args.max_new_tokens):
            logits = model.decoder(tgt_ids, enc_out, src_padding_mask=src_padding_mask)
            logits = logits[:, -1, :]

            top_k_logits, _ = torch.topk(logits, args.top_k, dim=-1)
            logits = logits.masked_fill(logits < top_k_logits[:, -1:], float('-inf'))

            probs   = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            tgt_ids = torch.cat([tgt_ids, next_id], dim=-1)

            if next_id.item() == eos_id:
                break

        output_ids = tgt_ids

# DECODE AND PRINT

generated = output_ids[0].tolist()

if generated and generated[0] == bos_id:
    generated = generated[1:]
if eos_id in generated:
    generated = generated[:generated.index(eos_id)]
generated = [t for t in generated if t != pad_id]

summary = tokenizer.decode(generated)

print(f"{'='*60}")
print("GENERATED SUMMARY:")
print(f"{'='*60}")
print(summary)
print(f"{'='*60}")


# COMPARE WITH REFERENCE (if available)

fname    = os.path.basename(judgment_path)
ref_path = os.path.join('Dataset', 'train-data', 'summary', fname)

if os.path.exists(ref_path):
    with open(ref_path, 'r', encoding='utf-8', errors='replace') as f:
        ref_summary = f.read().strip()
    print("\nREFERENCE SUMMARY:")
    print(f"{'='*60}")
    print(ref_summary)
    print(f"{'='*60}")
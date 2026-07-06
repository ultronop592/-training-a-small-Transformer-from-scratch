import os 
import sys
import pickle 
import argparse

from matplotlib.pylab import pad
import torch 
from tokenizers import Tokenizer

from model import Transformer, TransformerConfig


# Arguments 

parser =  argparse.ArgumentParser(description='Generate summaries for INABS dataset using a trained transformer model')
parser.add_argument('--checkpoint', type=str, default = 'out/inabs/ckt.pt', help='Path to the trained model checkpoint')
parser.add_argument('--judgement', type=str, default=None, help='Path to the input text file for summarization')
parser.add_argument('--max_new_tokens', type=int, default=256, help='Maximum number of new tokens to generate')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on (cpu or cuda)')
parser.add_argument('--topk', type = int , default =1, help='Top-k sampling for text generation')

args  = parser.parse_args()

# LOAD CHECKPOINTS 

print(f'Loading model checkpoint from {args.checkpoint}...')
checkpoint = torch.load(args.checkpoint, map_location=args.device)
model_args =  checkpoint['model_args']
print(f'Model arguments: {model_args}')


cfg  = TransformerConfig(**model_args)
model = Transformer(cfg)

state_dict = checkpoint['model_state_dict']
for k in list(state_dict.keys()):
    if k.startswith('module.'):
        state_dict[k[len('module.'):]] = state_dict.pop(k)
        
model.load_state_dict(state_dict)
model.to(args.device)
model.eval()

print(f"checkpoint loaded trained for {checkpoint['item_num']} iters, with best val loss: {checkpoint['best_val_loss']:.4f}")


# LOAD TOKEINIZER + META 


data_dir = os.path.dirname(args.checkpoint)
tokenizer_path = os.path.join(data_dir, 'tokenizer.json')


if not os.path.exists(tokenizer_path):
    print(f"Tokenizer file not found at {tokenizer_path}. Please ensure the tokenizer is trained and saved.")
    print("did ypu run: python data/inabs/prepare.py --train_tokenizer")
    sys.exit(1)
    
with open(sys.meta_path, 'rb') as f:
    meta = pickle.load(f)
    

pad_id  = meta['pad_id ']
bos_id  = meta['bos_id']
eos_id  = meta['eos_id']
src_max_len = meta['src_max_len']

tokenizer = Tokenizer.from_file(tokenizer_path)
print(f"Tokenizer loaded from {tokenizer_path} with vocab size: {tokenizer.get_vocab_size()}")


# LOAD THE JUDGEMENT TEXT 

if args.judgement:
    judgement_path = args.judgement
else:
    test_dir = os.path.join('Dataset', 'inabs', 'test-data', 'judgement')
    if not os.path.exists(test_dir):
        print(f"Test directory not found at {test_dir}. Please ensure the test data is available.")
        sys.exit(1)
    import glob 
    test_files = sorted(glob.glob(os.path.join(test_dir, '*.txt')))
    if not test_files:
        print(f"No test files found in {test_dir}. Please ensure the test data is available.")
        sys.exit(1)
    judgement_path = test_files[0]
    print(f"No judgement file specified. Using the first test file found: {judgement_path}")
    
with open(judgement_path, 'r', encoding='utf-8', error = 'replace') as f:
    judgement_text = f.read().strip()
    
print(f"\n{'='*60}")
print(f"Judgement text loaded from {judgement_path}:\n")
print(judgement_text[:500])
print(f"\n{'='*60}\n")



# ENCODE THE JUDGEMENT TEXT

enc =  tokenizer.encode(judgement_text, add_special_tokens=False)
src_idS  = enc.ids 

if len(src_ids) > src_max_len:
    src_ids  = src_ids[:src_max_len]
    
    
pad_len = src_max_len - len(src_ids)
src_ids  = src_ids + [pad_id] * pad_len

src  = torch.tensor([src_ids], dtype=torch.long, device=args.device)

# Generate 


print('generating summary...')

with torch.no_grad():
    if args.top_k == 1:
        # greedy decoding 
        
        output_ids = model.generate(src, max_new_tokens=args.max_new_tokens, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id)
    else:
        src_padding_mask = (src  == pad_id)
        enc_out = model.encode(src, src_padding_mask)
        
        tgt_ids = torch.full((1,1), bos_id, dtype = torch.long, device = args.device)
        
        for _ in range(args.max_new_tokens):
            logits  = model.decoder(tgt_ids, enc_out, src_padding_mask = src_padding_mask)
            logits  = logits[:, -1, :]
            
            
            top_k_logits, _ = torch.topk(logits, args.top_k, dim=-1)
            min_top_k = top_k_logits[:, -1].unsqueeze(-1)
            logits[logits < min_top_k] = -float('Inf')
            
            probs  = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            tgt_ids = torch.cat([tgt_ids, next_id], dim=-1)
            
            
            if next_id.item() ==  eos_id:
                break
            
    output_ids = tgt_ids
    
    
#DECODER AND PRINT 

generated  = output_ids[0].tolist()

if generated and generated[0]== bos_id:
    generated = generated[1:]
    
if eos_id in generated:
    generated = generated[:generated.index(eos_id)]
 
# remove any remaining padding
generated = [t for t in generated if t != pad_id]
 
# decode token ids → text
summary = tokenizer.decode(generated)
 
print(f"{'='*60}")
print("GENERATED SUMMARY:")
print(f"{'='*60}")
print(summary)
print(f"{'='*60}")
    

    
# COMAPRE WITH REFERNCE 
if args.judgment is None or 'test-data' in judgment_path:
    fname = os.path.basename(judgment_path)
    ref_path = os.path.join('Dataset', 'IN-Abs', 'test-data', 'summary', fname)
    if os.path.exists(ref_path):
        with open(ref_path, 'r', encoding='utf-8', errors='replace') as f:
            ref_summary = f.read().strip()
        print("\nREFERENCE SUMMARY (from dataset):")
        print(f"{'='*60}")
        print(ref_summary)
        print(f"{'='*60}")
        print("\n(compare the two above to judge model quality)")
          
          
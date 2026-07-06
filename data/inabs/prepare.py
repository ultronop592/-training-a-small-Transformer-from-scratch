import os
import glob 
import pickle 
import random 
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import whitespace
from tokenizers.processors import TemplateProcessing






#congiguration 

DATA_ROOT = os.path.join('Datasets', 'inabs')
TRAIN_DATA = os.path.join(DATA_ROOT, 'train')
TEST_DATA = os.path.join(DATA_ROOT, 'test')



OUT_DIR = os.path.join('out', 'inabs')
os.makedirs(OUT_DIR, exist_ok=True)


VOCAB_SIZE = 8000
SRC_MAX_LEN = 1024
TGT_MAX_LEN = 256

PAD_TOKEN = '[PAD]'
BOS_TOKEN = '[BOS]'
EOS_TOKEN = '[EOS]'
UNK_TOKEN = '[UNK]'

VAL_SPLIT = 0.1
random.seed(1337)





#  load file pairs

 
def load_pairs(split_dir):
    """
    Returns list of (judgment_text, summary_text) tuples.
    split_dir = 'Dataset/IN-Abs/train-data' or 'test-data'
    """
    judgement_dir = os.path.join(split_dir, 'judgement')
    summary_dir   = os.path.join(split_dir, 'summary')
 
    
    judgement_files = sorted(glob.glob(os.path.join(judgement_dir, '*.txt')))
    pairs = []
    skipped = 0
 
    for jpath in judgement_files:
        fname   = os.path.basename(jpath)
        spath   = os.path.join(summary_dir, fname)
 
        if not os.path.exists(spath):
            print(f"  WARNING: no matching summary for {fname}, skipping")
            skipped += 1
            continue
 
        with open(jpath, 'r', encoding='utf-8', errors='replace') as f:
            judgment_text = f.read().strip()
        with open(spath, 'r', encoding='utf-8', errors='replace') as f:
            summary_text  = f.read().strip()
 
    
        if not judgment_text or not summary_text:
            skipped += 1
            continue
 
        pairs.append((judgment_text, summary_text))
 
    print(f"  loaded {len(pairs)} pairs from {split_dir} ({skipped} skipped)")
    return pairs
 
 
print("=" * 60)
print("Loading file pairs...")
train_pairs = load_pairs(TRAIN_DIR)

random.shuffle(train_pairs)
split_idx   = int(len(train_pairs) * (1 - VAL_SPLIT))
val_pairs   = train_pairs[split_idx:]
train_pairs = train_pairs[:split_idx]
 
print(f"  train pairs: {len(train_pairs)}")
print(f"  val pairs:   {len(val_pairs)}")
 
 

# STEP 3 — TRAIN BPE TOKENIZER

print("=" * 60)
print(f"Training BPE tokenizer (vocab_size={VOCAB_SIZE})...")
 

all_texts = [j for j, s in train_pairs] + [s for j, s in train_pairs]
 

temp_corpus = os.path.join(OUT_DIR, '_corpus_temp.txt')
with open(temp_corpus, 'w', encoding='utf-8') as f:
    for text in all_texts:
        f.write(text + '\n')
print(f"  wrote {len(all_texts)} texts to temp corpus file")
 

tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))

tokenizer.pre_tokenizer = Whitespace()
 

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN],
    min_frequency=2,     
    show_progress=True,
)
 
tokenizer.train(files=[temp_corpus], trainer=trainer)

bos_id = tokenizer.token_to_id(BOS_TOKEN)
eos_id = tokenizer.token_to_id(EOS_TOKEN)
pad_id = tokenizer.token_to_id(PAD_TOKEN)
unk_id = tokenizer.token_to_id(UNK_TOKEN)
 
tokenizer.post_processor = TemplateProcessing(
    single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
    special_tokens=[
        (BOS_TOKEN, bos_id),
        (EOS_TOKEN, eos_id),
    ],
)
 

tokenizer_path = os.path.join(OUT_DIR, 'tokenizer.json')
tokenizer.save(tokenizer_path)
print(f"  tokenizer saved → {tokenizer_path}")
print(f"  actual vocab size: {tokenizer.get_vocab_size()}")
print(f"  pad_id={pad_id}, bos_id={bos_id}, eos_id={eos_id}, unk_id={unk_id}")
 
# clean up temp corpus file
os.remove(temp_corpus)
 
 

# STEP 4 — ENCODE + PAD/TRUNCATE → numpy arrays


 
def encode_and_pad(texts, max_len, is_target=False):
    """
    Tokenize a list of texts.
    Truncate if longer than max_len.
    Pad with pad_id if shorter than max_len.
 
    is_target=True → add BOS/EOS special tokens around the sequence
    is_target=False → no special tokens (encoder doesn't need them)
 
    Returns: numpy array of shape (N, max_len), dtype=uint16
    """
    encodings = tokenizer.encode_batch(texts, add_special_tokens=is_target)
 
    rows = []
    for enc in encodings:
        ids = enc.ids
 
    
        if len(ids) > max_len:
            ids = ids[:max_len]
            if is_target:
                ids[-1] = eos_id
 
        pad_len = max_len - len(ids)
        ids = ids + [pad_id] * pad_len
 
        rows.append(ids)
 
    arr = np.array(rows, dtype=np.uint16)
    return arr
 
 
def write_split(pairs, src_bin_path, tgt_bin_path, split_name):
    judgments = [j for j, s in pairs]
    summaries = [s for j, s in pairs]
 
    print(f"\nEncoding {split_name} split ({len(pairs)} pairs)...")
 
    src_arr = encode_and_pad(judgments, SRC_MAX_LEN, is_target=False)
    tgt_arr = encode_and_pad(summaries, TGT_MAX_LEN, is_target=True)
 
    
    src_flat = src_arr.flatten()
    tgt_flat = tgt_arr.flatten()
 
    src_flat.tofile(src_bin_path)
    tgt_flat.tofile(tgt_bin_path)
 
    print(f"  src: {src_arr.shape} → {src_bin_path}  ({src_flat.nbytes / 1e6:.1f} MB)")
    print(f"  tgt: {tgt_arr.shape} → {tgt_bin_path}  ({tgt_flat.nbytes / 1e6:.1f} MB)")
 
   
    reloaded = np.fromfile(src_bin_path, dtype=np.uint16)
    assert reloaded.shape[0] == len(pairs) * SRC_MAX_LEN, "src bin file shape mismatch!"
    print(f"  sanity check passed ✓")
 
 
print("=" * 60)
 
write_split(
    train_pairs,
    src_bin_path=os.path.join(OUT_DIR, 'train_src.bin'),
    tgt_bin_path=os.path.join(OUT_DIR, 'train_tgt.bin'),
    split_name='train',
)
 
write_split(
    val_pairs,
    src_bin_path=os.path.join(OUT_DIR, 'val_src.bin'),
    tgt_bin_path=os.path.join(OUT_DIR, 'val_tgt.bin'),
    split_name='val',
)
 
 

# STEP 5 — SAVE meta.pkl

meta = {
    'vocab_size': tokenizer.get_vocab_size(),
    'pad_id':     pad_id,
    'bos_id':     bos_id,
    'eos_id':     eos_id,
    'unk_id':     unk_id,
    'src_max_len': SRC_MAX_LEN,
    'tgt_max_len': TGT_MAX_LEN,
    'tokenizer_path': tokenizer_path,
}
 
meta_path = os.path.join(OUT_DIR, 'meta.pkl')
with open(meta_path, 'wb') as f:
    pickle.dump(meta, f)
 
print("=" * 60)
print(f"meta.pkl saved → {meta_path}")
print(f"  vocab_size : {meta['vocab_size']}")
print(f"  pad_id     : {meta['pad_id']}")
print(f"  bos_id     : {meta['bos_id']}")
print(f"  eos_id     : {meta['eos_id']}")
 

# STEP 6 — QUICK DECODE VERIFICATION


 
print("=" * 60)
print("Decode verification (first training example):")
 

src_check = np.fromfile(os.path.join(OUT_DIR, 'train_src.bin'), dtype=np.uint16)
src_check = src_check.reshape(-1, SRC_MAX_LEN)
 
first_src_ids = src_check[0].tolist()

first_src_ids = [i for i in first_src_ids if i != pad_id]
decoded_src = tokenizer.decode(first_src_ids)
print(f"\nJudgment (first 200 chars):\n  {decoded_src[:200]}...")
 
tgt_check = np.fromfile(os.path.join(OUT_DIR, 'train_tgt.bin'), dtype=np.uint16)
tgt_check = tgt_check.reshape(-1, TGT_MAX_LEN)
 
first_tgt_ids = tgt_check[0].tolist()
first_tgt_ids = [i for i in first_tgt_ids if i != pad_id]
decoded_tgt = tokenizer.decode(first_tgt_ids)
print(f"\nSummary (first 200 chars):\n  {decoded_tgt[:200]}...")
 
print("\n" + "=" * 60)
print("prepare.py complete. You can now run:")
print("  python train.py config/train_inabs.py")
print("=" * 60)
 
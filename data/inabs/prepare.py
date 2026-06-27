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
    
judgenent_dir =  os.path.join(split_dir, 'judgement')
summary_dir = os.path.join(split_dir, 'summary')

judgenent_files = sorted(glob.glob(os.path.join(judgenent_dir, '*.txt')))
summary_files = sorted(glob.glob(os.path.join(summary_dir, '*.txt')))
pair  = []
skipped = 0

for jpath in judgenent_files:
    fname = os.path.basename(jpath)
    spath = os.path.join(summary_dir, fname)
    if not os.path.exists(spath):
        skipped += 1
        continue
    with open(jpath, 'r', encoding='utf-8') as f:
        judgenent_text = f.read().strip()
    with open(spath, 'r', encoding='utf-8', errors='ignore') as f:
        summary_text = f.read().strip()
    if not judgenent_text or not summary_text:
        skipped += 1
        continue
    
    
    pairs.append((judgenent_text, summary_text))
    
   print( f"Loaded {len(pairs)} pairs from {split_dir}, skipped {skipped} files")
   return pass 

print()
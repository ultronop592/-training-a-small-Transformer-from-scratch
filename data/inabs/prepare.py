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
    

# MODEL ENCODER - DECODER TRANSFORMER 

import math 
from dataclasses import dataclass
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
    
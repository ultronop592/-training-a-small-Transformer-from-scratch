import os 
import time
import math 
import pickle 
from contextlib import nullcontext
import numpy as np
import torch

from model import TransformerCongig, Transformer



#hyperparametes

out_dir = 'out/inabs'
eval_interval =  500
log_interval = 10
eval_iters = 50
eval_only = False



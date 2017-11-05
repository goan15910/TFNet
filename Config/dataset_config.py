from easydict import EasyDict as edict
import numpy as np
from math import ceil


config = edict()

# Basics
config.shuffle = True # shuffle train or not

# Queue related
config.use_q = False
config.q_maxsize = 50
config.q_min_frac = 0.4
config.n_idx_threads = 1
config.n_batch_threads = 2

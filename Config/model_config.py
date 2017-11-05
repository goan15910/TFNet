from easydict import EasyDict as edict
import numpy as np
from math import ceil


config = edict()

# Training process related
config.batch_size = None
config.max_steps = None
config.lr = 1e-3

# Optimization related
config.wd = 5*1e-4

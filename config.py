from easydict import EasyDict as edict
import numpy as np
from math import ceil


config = edict()

# Training process related
config.batch_size = edict()
config.batch_size.train = None
config.batch_size.val = None
config.batch_size.test = None
config.max_steps = None
config.lr = None

# Dataset related
config.shuffle = None # shuffle train or not

# Optimization related
config.wd = None

from easydict import EasyDict as edict

# Import future models here
from simple_segnet import Simple_SegNet

model_table = edict()
model_table['simple_segnet'] = Simple_SegNet

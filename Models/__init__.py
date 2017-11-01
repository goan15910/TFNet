from easydict import EasyDict as edict

# Import future models here
from SegNets.segnet import SegNet
from SegNets.simple_segnet import Simple_SegNet

model_table = edict()
model_table['segnet'] = SegNet
model_table['simple_segnet'] = SegNet

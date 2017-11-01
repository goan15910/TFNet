from easydict import EasyDict as edict

# Import future dataset here
from camvid import CamVid


dataset_table = edict()
dataset_table['camvid'] = CamVid

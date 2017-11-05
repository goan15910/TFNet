import os,sys

import numpy as np
import cv2
from easydict import EasyDict as edict
from DS_base.dataset import Dataset, SET
import statistics as stats
from Config.dataset_config import config


class CamVid(Dataset):
  """
  Camvid dataset, ref:
  http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
  Dataset info:

  """
  def __init__(self,
               root_dir,
               use_sets):
    # customize cfg here
    cfg = config
    # no customize setting

    # setup dataset skeleton
    Dataset.__init__(self,
                     root_dir,
                     cfg,
                     use_sets)
    # name
    self.name = 'CamVid'

    # batch-key
    self._batch_keys = ('clips', 'images', 'labels')
    self._datum_shapes = (None, [360, 480, 3], [360, 480, 1])

    # fname-dict
    self.fname_dict = edict()
    self.fname_dict[SET.TRAIN] = \
        os.path.join(root_dir, 'train.txt')
    self.fname_dict[SET.TEST] = \
        os.path.join(root_dir, 'test.txt')
    self.fname_dict[SET.VAL] = \
        os.path.join(root_dir, 'val.txt')

    # dataset info
    self._n_cls = 11
    self._cls_names = np.array([
      'Sky', 'Building', 'Pole',
      'Road_marking', 'Road', 'Pavement',
      'Others', 'Others', 'Others',
      'Others', 'Others',])
    self._cls_colors = np.array(
     [[128, 128, 128], [128, 0, 0], [192, 192, 128],
      [255, 69, 0], [128, 64, 128], [60, 40, 222],
      [0, 0, 0], [0, 0, 0], [0, 0, 0],
      [0, 0, 0], [0, 0, 0],],
      dtype=np.uint8)


  def _read_fnames(self, fname):
    """
    Read-func for train/val/test txt file
    Args:
      fname: filename of txt file
    Return:
      a list of (clip, image, label)
    """
    with open(fname, 'r') as f:
      lines = f.readlines()

    # clip_name, img_path, la_path
    fnames = []
    for line in lines:
      im, la = line.strip().split(' ')
      clip = im.split('/')[-1].split('_')[0]
      fnames.append((clip, im, la)) # as order of batch-key
    return fnames


  def _decode_func(self, fnames):
    # Define the decode function
    def camvid_decode(idxs):
      """
      Read-func of decoding idxs to batch
      Args:
        idxs: a list of index
      """
      clips = []
      images = []
      labels = []
      for idx in idxs:
        clip, im, la = fnames[idx]
        clips.append(clip)
        image = cv2.imread(im)
        label = cv2.imread(la)[..., 0]
        images.append(image)
        labels.append(label)
      return (clips, images, labels)

    return camvid_decode

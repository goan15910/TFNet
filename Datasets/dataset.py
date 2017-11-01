import os, sys
from math import ceil
from Queue import Queue
from threading import Thread

import numpy as np
import cv2
from easydict import EasyDict as edict
from base_queue import BaseQueue, push_idxs, push_batch


# Flags for SET
SET = edict()
SET.TRAIN = 'train'
SET.VAL = 'val'
SET.TEST = 'test'


class Dataset:
  """Base class of dataset object"""
  def __init__(self,
               config,
               n_threads):
    # Basic info
    self.config = config
    self.name = None
    self.n_threads = n_threads
    self._batch_key = None
    self._datum_shape = None
    self._min_queue_num = None

    # Train / val / test sets
    self._set = edict()
    for skey in SET.keys():
      self._set[skey] = edict()
      self._set[skey].fnames = None
      self._set[skey].idx_q = None
      self._set[skey].batch_q = None

    # Dataset info
    self._n_cls = None
    self._cls_names = None
    self._cls_colors = None


  @property
  def batch_key(self):
    if self._batch_key is None:
      raise NotImplementedError
    return self._batch_key


  @property
  def datum_shape(self):
    if self._datum_shape is None:
      raise NotImplementedError
    return self._datum_shape


  @property
  def min_queue_num(self):
    if self._min_queue_num is None:
      raise NotImplementedError
    return self._min_queue_num


  @property
  def num_classes(self):
    if self._n_cls is None:
      raise NotImplementedError
    return self._n_cls


  @property
  def class_names(self):
    if self._cls_names is None:
      raise NotImplementedError
    return self._cls_names


  @property
  def class_colors(self):
    if self._cls_colors is None:
      raise NotImplementedError
    return self._cls_colors


  def epoch_info(self, skey):
    """Compute epoch steps and total number of examples"""
    self._check_set_key(skey)
    n_examples = len(self._set[skey].fnames)
    batch_size = self.config.batch_size
    epoch_steps = np.ceil(n_examples / batch_size).astype(np.int64)
    return (epoch_steps, n_examples)


  def start(self):
    self._load_idxs(self.fname_dict)
    self._load_batches()


  def done(self):
    for _set in self._set.values():
      _set.idx_q.done()
      _set.batch_q.done()


  def batch(self, skey):
    """Get a mini-batch data"""
    self._check_set_key(skey)
    batch_item = self._set[skey].pop()
    batch = zip(self.batch_key, self.batch_item)
    return edict(dict(batch_dict))


  def _load_idxs(self):
    """Start loading idx-queue"""
    fname_dict = self.fname_dict
    for skey in set(fname_dict.keys()):
      self._check_set_key(skey)
      fnames = self._read_fnames(fname_dict[skey])
      batch_size = self.config.batch_size[skey]
      if skey == SET.TRAIN:
        shuffle = self.config.shuffle
      else:
        shuffle = False

      idx_q = BaseQueue(self.n_threads, batch_size)
      push_idxs_args = (len(fnames), shuffle)
      idx_q.start(push_idxs, args)
      self._set[skey].idx_q = idx_q
      self._set[skey].fnames = fnames


  def _load_batches(self):
    """Start loading batch-queue"""
    batch_queue = BaseQueue(self.n_threads)
    push_batch_args = \
        (self.idx_q.queue, self._read_func)
    batch_queue.start(push_batch,
                      push_batch_args)


  def _read_fnames(self, fname):
    raise NotImplementedError


  def _decode_func(self, idxs):
    raise NotImplementedError


  def _check_skey(self, skey):
    assert skey in SETS.values(), \
        "Invalid set {}".format(skey)

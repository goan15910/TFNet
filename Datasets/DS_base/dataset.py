import os, sys
from math import ceil
from Queue import Queue
from threading import Thread

import numpy as np
import cv2
from easydict import EasyDict as edict
from data_queue import DataQueue
from data_list import DataList


# Flags for SET
SET = edict()
SET.TRAIN = 'train'
SET.VAL = 'val'
SET.TEST = 'test'


class Dataset:
  """Base class of dataset object"""
  def __init__(self,
               root_dir,
               cfg,
               use_sets,
               use_q=None):
    # Basic info
    self.name = None
    self.root_dir = root_dir
    self._batch_keys = None
    self._batch_size = None
    self.cfg = cfg
    if use_q is not None:
      self.cfg.use_q = use_q

    # Train / val / test sets
    self._use_sets = use_sets
    self.dc = edict() # data container
    for skey in use_sets:
      self.dc[skey] = None

    # Dataset info
    self._n_cls = None
    self._cls_names = None
    self._cls_colors = None


  @property
  def batch_keys(self):
    if self._batch_keys is None:
      raise NotImplementedError
    return self._batch_keys


  @property
  def datum_shapes(self):
    if self.cfg.datum_shapes is None:
      raise NotImplementedError
    return self.cfg.datum_shapes


  @property
  def n_sets(self):
    return len(self._use_sets)


  @property
  def batch_size(self):
    if self._batch_size is None:
      raise NotImplementedError
    return self._batch_size


  @property
  def np_out(self):
    if self.cfg.datum_shape is not None:
      return True
    else:
      return False


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


  def batch_shape(self, batch):
    raise NotImplementedError


  def epoch_steps(self, skey):
    """Steps to run through whole epoch"""
    self._check_set_key(skey)
    return self.dc[skey].epoch_steps


  def set_batch_size(self, batch_size):
    self._batch_size = batch_size


  def start(self):
    for skey in self._use_sets:
      # basic info
      self._check_skey(skey)
      if skey == SET.TRAIN:
        shuffle = self.cfg.shuffle
      else:
        shuffle = False
      filename = self.fname_dict[skey]

      # data queue
      if self.cfg.use_q:
        self.dc[skey] = DataQueue(
            filename,
            self.batch_size,
            self._read_fnames,
            self._decode_func,
            self.np_out,
            self.cfg.n_idx_threads,
            self.cfg.n_batch_threads,
            self.cfg.q_min_frac,
            self.cfg.q_maxsize,
            shuffle=shuffle,
            name=skey)
      else:
        self.dc[skey] = DataList(
            filename,
            self.batch_size,
            self._read_fnames,
            self._decode_func,
            np_out=self.np_out,
            shuffle=shuffle,
            name=skey)
      self.dc[skey].start()


  def done(self):
    for dc in self.dc.values():
      dc.done()


  def batch(self, skey):
    """Get a mini-batch data"""
    self._check_skey(skey)
    batch_items = self.dc[skey].pop_batch()
    batch_pairs = zip(self.batch_keys, batch_items)
    return edict(dict(batch_pairs))


  def _read_fnames(self, fname):
    raise NotImplementedError


  def _decode_func(self, idxs):
    raise NotImplementedError


  def _check_skey(self, skey):
    assert skey in SET.values(), \
        "Invalid set {}".format(skey)

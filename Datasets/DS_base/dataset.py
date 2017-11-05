import os, sys
from math import ceil
from Queue import Queue
from threading import Thread

import numpy as np
import cv2
from easydict import EasyDict as edict
from data_queue import DataQueue


# Flags for SET
SET = edict()
SET.TRAIN = 'train'
SET.VAL = 'val'
SET.TEST = 'test'


class Dataset:
  """Base class of dataset object"""
  def __init__(self,
               root_dir,
               use_sets,
               n_threads=6):
    # Basic info
    self.name = None
    self.root_dir = root_dir
    self.n_threads = max(6, n_threads)
    self._batch_keys = None
    self._datum_shape = None
    self._q_thres = None

    # Train / val / test sets
    self._set = edict()
    for skey in use_sets:
      self._set[skey] = edict()
      self._set[skey].data_q = None
      self._set[skey].

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
    if self._datum_shapes is None:
      raise NotImplementedError
    return self._datum_shapes


  @property
  def n_sets(self):
    return len(self._set.keys())


  @property
  def batch_size(self):
    return self.config.batch_size


  @property
  def n_q_threads(self):
    return  self.n_threads / self.n_sets


  @property
  def q_thres(self):
    if self._q_thres is None:
      raise NotImplementedError
    return self._q_thres


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


  def epoch_steps(self, skey):
    """Steps to run through whole epoch"""
    self._check_set_key(skey)
    return data_container.epoch_steps


  def set_config(self, config):
    self.config = config

    # thres for queue
    self._q_thres = \
        self.config.q_thres


  def start(self):
    self._load()


  def done(self):
    for _set in self._set.values():
      _set.data_q.done()


  def batch(self, skey):
    """Get a mini-batch data"""
    self._check_skey(skey)
    batch_items = self._set[skey].data_q.pop_batch()
    batch_pairs = zip(self.batch_keys, batch_items)
    return edict(dict(batch_pairs))


  def _load(self):
    """Start loading idx/batch queue"""
    fname_dict = self.fname_dict
    for skey in self._set.keys():
      self._check_skey(skey)
      batch_size = self.config.batch_size
      if skey == SET.TRAIN:
        shuffle = self.config.shuffle
      else:
        shuffle = False

      # data queue
      data_q = DataQueue(
                   fname_dict[skey],
                   batch_size,
                   self._read_fnames,
                   self._decode_func,
                   n_threads=self.n_q_threads,
                   thres=self.q_thres,
                   shuffle=shuffle)
      data_q.start()
      self._set[skey].data_q = data_q


  def _read_fnames(self, fname):
    raise NotImplementedError


  def _decode_func(self, idxs):
    raise NotImplementedError


  def _check_skey(self, skey):
    assert skey in SET.values(), \
        "Invalid set {}".format(skey)

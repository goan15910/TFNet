import os, sys
from datetime import datetime

import tensorflow as tf
import numpy as np

from easydict import EasyDict as edict

# modules
from Dataset import d_table
from Dataset.dataset import SET
from Models import m_table
from config import config
from initer import Initer
from vizer import Vizer


MODE = edict()
MODE.TRAIN = 'train'
MODE.TEST = 'test'


class Master:
  """Master coordinates all sub-components"""
  def __init__(self,
               FLAGS):
    # Setup path & flags from FLAGS
    self.mode = FLAGS.mode
    self.dataset_dir = FLAGS.dataset_dir
    self.save_dir = FLAGS.save_dir
    self.n_threads = int(FLAGS.threads / 2)
    self.pretrained = FLAGS.pretrained
    # TODO
    #self.log_dir = FLAGS.log_dir
    #self.test_ckpt = FLAGS.test_ckpt

    # Config, model, dataset
    self.config = config
    self.m_class = m_table[FLAGS.model]
    self.d_class = d_table[FLAGS.dataset]

    # Some checking
    self._check_mode(self.mode)
    self._check_exists(self.dataset_dir)
    self._check_exists(self.save_dir)


  def run(self):
    """Run the specified mode"""
    self._setup()
    if self.mode == MODE.TRAIN:
      self.model.build()
      self.model.train()
    elif self.mode == MODE.TEST:
      # TODO: restore ckpt first
      self.model.build()
      self.model.eval(SET.TEST)

    # Done
    self._done()


  def _setup(self):
    """Setup all components"""
    # dataset
    self.dataset = self.d_class(
                       self.dataset_dir,
                       self.config,
                       self.n_threads)
    print "Loading {} with {} threads".format(self.dataset.name, self.n_threads * 2)
    self.dataset.start()

    with tf.Graph().as_default():
      # initer
      self.initer = Initer(self.config,
                           self.pretrained)

      # vizer
      self.vizer = Vizer(self.save_dir)

      # model
      self.model = self.m_class(
                       self.config,
                       self.dataset,
                       self.initer,
                       self.vizer,
                       self.save_dir)


  def _done(self):
    """All job done"""
    print "Terminate all processes and threads."
    self.model.done()


  def _check_exists(self, path):
    assert os.path.exists(path), \
        "{} not exist!".format(path)


  def _check_mode(self, mkey):
    assert mkey in MODE.values(), \
        "Invalid mode {}".format(mkey)

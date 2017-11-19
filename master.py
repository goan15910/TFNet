import os, sys
from datetime import datetime

import tensorflow as tf
import numpy as np

from easydict import EasyDict as edict

# modules
from Datasets import dataset_table as d_table
from Models import model_table as m_table
from initer import Initer
from vizer import Vizer

from Datasets.DS_base.dataset import SET


MODE = edict()
MODE.TRAIN = 'train'
MODE.TEST = 'test'


class Master:
  """Master coordinates all sub-components"""
  def __init__(self,
               FLAGS):
    # Some checking
    self._check_mode(FLAGS.mode)
    self._check_exists(FLAGS.dataset_dir)
    self._check_exists(FLAGS.save_dir)

    # Setup path & flags from FLAGS
    self.mode = FLAGS.mode
    self.dataset_dir = FLAGS.dataset_dir
    self.save_dir = FLAGS.save_dir
    self.n_threads = FLAGS.threads
    self.pretrained = FLAGS.pretrained
    self.use_q = FLAGS.queue
    self.ckpt = FLAGS.ckpt

    # initiate classes
    self.m_class = m_table[FLAGS.model]
    self.d_class = d_table[FLAGS.dataset]


  def run(self):
    """Run the specified mode"""
    self._setup()
    if self.mode == MODE.TRAIN:
      print "Start training ..."
      self.model.build()
      self.model.run_init(ckpt=self.ckpt)
      self.model.train()
    elif self.mode == MODE.TEST:
      print "Start testing ..."
      self.model.build()
      self.model.run_init(ckpt=self.ckpt)
      self.model.eval(SET.TEST)

    # Done
    self._done()


  def _setup(self):
    """Setup all components"""
    # dataset
    if self.mode == MODE.TRAIN:
      use_sets = (SET.TRAIN, SET.VAL)
    elif self.mode == MODE.TEST:
      use_sets = (SET.TEST)
    self.dataset = self.d_class(
                       self.dataset_dir,
                       use_sets,
                       self.use_q)

    # initer
    self.initer = Initer(self.pretrained)

    # vizer
    self.vizer = Vizer(self.save_dir)

    # model
    self.model = self.m_class(
                     self.dataset,
                     self.initer,
                     self.vizer,
                     self.save_dir)

    # Start loading dataset
    print "Loading {}".format(self.dataset.name)
    self.dataset.start()


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

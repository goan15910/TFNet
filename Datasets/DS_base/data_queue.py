import numpy as np
from easydict import EasyDict as edict
from data_container import DataContainer
from base_queue import BaseQueue
from push_func import push_idxs, push_seq_idxs, push_batch


class DataQueue(DataContainer):
  """
  Object managing index-queue & batch-queue
  """
  def __init__(self,
               filename,
               batch_size,
               read_fnames_func,
               decode_func,
               np_out,
               n_idx_threads=1,
               n_batch_threads=2,
               min_frac=0.4,
               maxsize=50,
               shuffle=False,
               name=''):
    # basics
    DataContainer.__init__(self,
                           filename,
                           batch_size,
                           read_fnames_func,
                           decode_func,
                           np_out,
                           shuffle,
                           name)

    # idx-q
    self.idx_q = BaseQueue(
                     n_idx_threads,
                     min_frac=min_frac,
                     maxsize=maxsize)
    self.idx_q_args = (self.n_fname,
                       self._shuffle)

    # batch-q
    self.batch_q = BaseQueue(
                       n_batch_threads,
                       min_frac=min_frac,
                       maxsize=maxsize)
    self.batch_q_args = (self.idx_q,
                         self._bsize,
                         self._decode,
                         self.fnames)


  def start(self):
    print self._start_str.format(self._name)
    self.idx_q.start(push_idxs, self.idx_q_args)
    self.batch_q.start(push_batch, self.batch_q_args)


  def pop_batch(self):
    return self.batch_q.pop()


  def done(self):
    print self._done_str.format(self._name)
    self.idx_q.done()
    self.batch_q.done()

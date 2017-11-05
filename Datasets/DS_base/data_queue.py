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
               n_threads=3,
               thres=20,
               shuffle=False,
               ratio=2,
               name=''):
    # basics
    DataContainer.__init__(self,
                           filename,
                           batch_size,
                           read_fnames_func,
                           decode_func,
                           shuffle,
                           name)

    # idx-q / batch-q n_threads
    idx_n_threads = max(n_threads / (1 + ratio), 1)
    batch_n_threads = ratio * idx_n_threads

    # idx-q
    self.idx_q = BaseQueue(
                     idx_n_threads,
                     thres=thres,
                     batch_size=self._bsize)
    self.idx_q_args = (self.n_fname, self._shuffle)

    # batch-q
    self.batch_q = BaseQueue(
                       batch_n_threads,
                       thres=thres,
                       batch_size=self._bsize)
    self.batch_q_args = \
      (self.idx_q, self._decode, self.fnames)


  def start(self):
    print self._start_str.format(self._name)
    self.idx_q.start(push_idxs, self.idx_q_args)
    self.batch_q.start(push_batch, self.batch_q_args)


  def pop_batch(self):
    return self.batch_q.pop_batch()


  def done(self):
    print self._stop_str.format(self._name)
    self.idx_q.done()
    self.batch_q.done()

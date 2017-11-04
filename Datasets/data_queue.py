import numpy as np
from easydict import EasyDict as edict
from base_queue import push_idxs, push_batch, BaseQueue


class DataQueue:
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
               ratio=2):
    # basics
    self._fnames = read_fnames_func(filename)
    self._shuffle = shuffle

    # idx-q / batch-q n_threads
    idx_n_threads = max(n_threads / (1 + ratio), 1)
    batch_n_threads = ratio * idx_n_threads

    # idx-q
    self.idx_q = BaseQueue(
                     idx_n_threads,
                     thres=thres,
                     batch_size=batch_size)
    self.idx_q_args = (len(self.fnames), shuffle)

    # batch-q
    self.batch_q = BaseQueue(
                       batch_n_threads,
                       thres=thres,
                       batch_size=batch_size)
    self.batch_q_args = \
        (self.idx_q, decode_func(self.fnames))
  
 
  @property
  def fnames(self):
    return self._fnames

  
  def start(self):
    self.idx_q.start(push_idxs, self.idx_q_args)
    self.batch_q.start(push_batch, self.batch_q_args)


  def pop_batch(self):
    return self.batch_q.pop_batch()


  def done(self):
    self.idx_q.done()
    self.batch_q.done()

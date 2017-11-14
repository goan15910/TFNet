import numpy as np
from easydict import EasyDict as edict


class DataContainer:
  """
  Base class for containing data
  """
  def __init__(self,
               filename,
               batch_size,
               read_fnames_func,
               decode_func,
               np_out,
               shuffle=False,
               name=''):
    # basics
    self._name = name
    self._fnames = read_fnames_func(filename)
    self._decode = decode_func(self._fnames)
    self._bsize = batch_size
    self._np_out = np_out
    self._shuffle = shuffle
    self._epoch_steps = \
      int(np.ceil(batch_size / self.n_fname))

    # info str
    self._start_str = '{} data container job starts ...'
    self._done_str = '{} data container job ends ...'


  @property
  def fnames(self):
    return self._fnames


  @property
  def n_fname(self):
    return len(self._fnames)


  @property
  def epoch_steps(self):
    return self._epoch_steps


  def start(self):
    raise NotImplementedError


  def pop_batch(self):
    raise NotImplementedError


  def done(self):
    raise NotImplementedError

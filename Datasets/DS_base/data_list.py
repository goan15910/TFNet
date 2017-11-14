import numpy as np
from easydict import EasyDict as edict
from data_container import DataContainer


class DataList(DataContainer):
  """
  Object loading and poping data
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
    DataContainer.__init__(self,
                           filename,
                           batch_size,
                           read_fnames_func,
                           decode_func,
                           np_out,
                           shuffle,
                           name)

    # data / indexes
    self._data = None
    self._idxs = None 
    self._cur_bidx = 0 # current batch index


  def start(self):
    print self._start_str.format(self._name)
    self._idxs = np.arange(self.n_fname)
    data = zip(*self._decode(self._idxs))
    self._data = np.array(data, dtype=object)


  def pop_batch(self):
    """Pop a batch out"""
    batch_idxs = self._get_batch_idxs()
    if self._is_last_batch:
      if self._shuffle:
        np.random.shuffle(self._idxs)
      self._cur_bidx = 0
    else:
      self._cur_bidx += 1
    batch_data = self._data[batch_idxs]
    batch_data = zip(*batch_data)
    if self._np_out:
      batch_data = \
        map(lambda x: np.hstack(x), batch_data)
    return batch_data


  def done(self):
    print self._done_str.format(self._name)


  def _get_batch_idxs(self):
    start = self._cur_bidx * self._bsize
    end = start + self._bsize
    idxs = self._idxs[start:end]
    res_len = self._bsize - len(idxs)
    if res_len > 0:
      pad_idxs = self._idxs[:res_len]
      idxs = np.hstack([idxs, pad_idxs])
    return idxs


  @property
  def _is_last_batch(self):
    """whether it's the last batch"""
    return (self._cur_bidx == self._epoch_steps-1)

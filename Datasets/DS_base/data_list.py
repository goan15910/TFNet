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
               shuffle=False,
               name=''):
    # basics
    DataContainer.__init__(self,
                           filename,
                           batch_size,
                           read_fnames_func,
                           decode_func,
                           shuffle,
                           name)

    # data / indexes
    self._data = None
    self._idxs = np.arange(self.n_fname)
    self._cur_bidx = 0 # current batch index


  def start(self):
    print self._start_str.format(self._name)
    data = []
    for fname in self.fnames:
      decoded_item = self._decode(fname)
      data.append(decoded_item)
    self._data = np.array(data)


  def pop_batch(self):
    """Pop a batch out"""
    batch_idxs = self._get_batch_idxs()
    if self._is_last_batch:
      if self._shuffle:
        np.random.shuffle(self._idxs)
      self._cur_bidx = 0
    else:
      self._cur_bidx += 1
    return self._data[batch_idxs]


  def done(self):
    print self._stop_str.format(self._name)


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

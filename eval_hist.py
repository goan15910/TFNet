import numpy as np


class Eval_hist:
  def __init__(self, n_cls=None):
    self.valid = False
    self.set_n_cls(n_cls)


  @property
  def n_examples(self):
    return self.hist.sum()


  @property
  def gt_pos(self):
    return self.hist.sum(0)


  @property
  def my_pos(self):
    return self.hist.sum(1)


  @property
  def true_pos(self):
    return np.diag(self.hist)


  @property
  def acc(self):
    return self.true_pos.sum() / self.n_examples


  @property
  def iu(self):
    return self.true_pos / (self.my_pos + self.gt_pos - self.true_pos)


  @property
  def mean_iu(self):
    return np.nanmean(self.iu)


  def set_n_cls(self, n_cls):
    if n_cls is not None:
      self.n_cls = n_cls
      self.shape = (n_cls, n_cls)
      self.hist = np.zeros(self.shape)
      self.valid = True


  def clean(self):
    """Clean hist"""
    self._validate()
    self.hist = np.zeros(self.shape)


  def update(self,
             preds,
             labels,
             clean=False):
    """Update histogram by new examples"""
    self._validate()
    if clean:
      self.clean()
    N = preds.shape[0]
    for i in xrange(N):
      labels = labels[i].flatten()
      preds = preds[i].argmax(2).flatten()
      self.hist += fast_hist(labels, preds)


  def printout(self):
    """Print out summary of histogram"""
    self._validate()
    print ('accuracy = %f' % self.acc)
    print ('mean IU  = %f' % self.mean_iu)
    for ii in xrange(self.n_cls):
        if self.my_pos[ii] == 0:
          acc = 0.0
        else:
          acc = self.true_pos[ii] / self.my_pos[ii]
        print("    class # %d accuracy = %f "%(ii, acc))


  def _fast_hist(self, a, b):
    k = (a >= 0) & (a < self.n_cls)
    flat_hist = np.bincount(
                    n_cls * a[k].astype(int) + b[k],
                    minlength=n_cls**2)
    return flat_hist.reshape(self.shape)


  def _validate(self):
    assert self.valid, \
        "Set number of class first!"

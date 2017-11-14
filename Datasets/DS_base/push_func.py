import numpy as np


def push_idxs(q,
              n_idx,
              shuffle):
  """Push idxs into queue"""
  cur = 0
  idxs = np.arange(n_idx)
  if shuffle:
    np.random.shuffle(idxs)

  while True:
    if cur == (n_idx-1):
      if shuffle:
        np.random.shuffle(idxs)
      cur = 0
    q.put(idxs[cur])
    cur += 1


def push_seq_idxs(q,
                  n_idx,
                  shuffle):
  """Push idxs as seq into queue"""
  # TODO
  pass


def push_batch(q,
               idx_q,
               bsize,
               decode,
               fnames):
  while True:
    idxs = idx_q.pop_n(bsize)
    batch = decode(idxs)
    batch = np.array(batch)
    q.put(batch)

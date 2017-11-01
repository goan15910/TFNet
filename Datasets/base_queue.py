import numpy as np
from easydict import EasyDict as edict
from Queue import Queue
from threading import Thread


def push_idxs(q, n_idx, shuffle):
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


def push_seq_idxs(q, n_idx, shuffle):
  """Push idxs as seq into queue"""
  # TODO
  pass


def push_batch(q, idx_q, decode_func):
  while True:
    batch = decode_func(idx_q.pop_batch())
    q.put(batch)


class BaseQueue:
  """
  Queue object
  """
  def __init__(self,
               n_threads,
               min_queue_num,
               batch_size=1):
    # basics
    self.queue = None
    self.n_threads = n_threads
    self.min_queue_num
    self.batch_size = batch_size
    self.workers = None


  def start(self,
            push_func,
            args):
    """Start loading queue"""
    self.queue = Queue()
    full_args = [self.queue]
    for i in xrange(n_threads):
      full_args.extend(args)
      worker = Thread(target=push_func,
                      args=full_args)
      worker.setDaemon(True)
      worker.start()
      self.workers = worker


  def done(self):
    """Worker job done"""
    for worker in self.workers:
      worker.task_done()
    self.queue.join()


  def pop_batch(self):
    """Pop items of batch size"""
    pop_items = []
    for i in xrange(self.batch_size):
      pop_items.append(self._pop())
    return pop_items


  def pop(self):
    """Pop one index out"""
    while (self.queue.qsize() > self.min_queue_num):
      try:
        return self.queue.get()
      except:
        pass

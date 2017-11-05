import numpy as np
from easydict import EasyDict as edict
from Queue import Queue
from threading import Thread


class BaseQueue:
  """
  Queue object
  """
  def __init__(self,
               n_threads,
               thres,
               batch_size,
               maxsize=50):
    # basics
    self.queue = None
    self.n_threads = n_threads
    self.thres = thres
    self.batch_size = batch_size
    self.maxsize = maxsize
    self.workers = None


  def start(self,
            push_func,
            args):
    """Start loading queue"""
    self.workers = []
    self.queue = Queue(maxsize=self.maxsize)
    full_args = [self.queue] + list(args)
    for i in xrange(self.n_threads):
      worker = Thread(target=push_func,
                      args=full_args)
      worker.setDaemon(True)
      worker.start()
      self.workers.append(worker)


  def done(self):
    """Worker job done"""
    for worker in self.workers:
      worker.task_done()
    self.queue.join()


  def pop_batch(self):
    """Pop items of batch size"""
    pop_items = []
    for i in xrange(self.batch_size):
      pop_item = self.pop()
      pop_items.append(pop_item)
    return pop_items


  def pop(self):
    """Pop one index out"""
    while (self.queue.qsize() > self.thres):
      try:
        return self.queue.get()
      except:
        pass

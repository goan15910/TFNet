import sys,os

import numpy as np
from Datasets import dataset_table as d_table
from Datasets.dataset import SET
from config import config

config.batch_size = 16

ITERS = 10
N_THREADS = 6


if __name__ == '__main__':

  # Args
  if len(sys.argv[1:]) != 2:
    print """
    usage: python test_dataset.py [dataset name] [root dir]
    """
    sys.exit()
  else:
    dname = sys.argv[1]
    root_dir = sys.argv[2]

  # Setup dataset
  d_class = d_table[dname]
  use_sets = (SET.TRAIN, SET.VAL)
  dataset = d_class(root_dir, use_sets, N_THREADS)
  dataset.set_config(config)
  dataset.start()

  # Pop batch
  print "Testing dataset {}".format(dataset.name)
  for skey in use_sets:
    print "Loading for {} set".format(skey)
    for i in xrange(ITERS):
      batch_dict = dataset.batch(skey)
      for k,v in batch_dict.iteritems():
        print "{}: {}".format(k, v)

  # End testing
  print "Pass the test"
  dataset.done()

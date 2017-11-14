import sys,os
import argparse
import time

import numpy as np
from Datasets import dataset_table as d_table
from Datasets.DS_base.dataset import SET


BATCH_SIZE = 2
ITERS = 4

parser = argparse.ArgumentParser(description='Test dataset module')
parser.add_argument('dataset', type=str,
                    default='camvid',
                    help='name of used dataset')
parser.add_argument('root_dir', type=str,
                    default='/tmp3/jeff/Camvid',
                    help='dataset root directory')
parser.add_argument('queue', type=bool,
                    nargs='?',
                    default=False,
                    help='use queue or not')

args = parser.parse_args()


if __name__ == '__main__':

  # Args
  dname = args.dataset
  root_dir = args.root_dir
  use_queue = args.queue

  # Setup dataset
  d_class = d_table[dname]
  use_sets = (SET.TRAIN, SET.VAL, SET.TEST)
  dataset = d_class(root_dir, use_sets, use_queue)
  dataset.set_batch_size(BATCH_SIZE)

  # Start loading
  print "Load up dataset ..."
  tic = time.time()
  dataset.start()
  toc = time.time() - tic
  if not use_queue:
    print "loading for {0:.1f} sec".format(toc)

  # Pop batch
  print "\nTesting dataset {}".format(dataset.name)
  print "With batch size {}".format(BATCH_SIZE)
  print "     iterations {}".format(ITERS)
  try:
    datum_shapes = dataset.datum_shapes
  except:
    datum_shapes = None
  print "     datum shape {}".format(datum_shapes)

  for skey in use_sets:
    print "\n============================================="
    for i in xrange(ITERS):
      print "\n{} batch {}".format(skey, i)
      batch_dict = dataset.batch(skey)
      for k in batch_dict.keys():
        shapes = dataset.batch_shape(batch_dict, k)
        print "{}: {}".format(k, shapes)

  # End testing
  print "\n============================================="
  print ""
  dataset.done()
  print "\nTest ended"

import tensorflow as tf
from tensorflow import GraphKeys as GKeys

import re
import numpy as np
from easydict import EasyDict as edict


SummaryType = edict()
SummaryType.SCALAR = 'scalar'
SummaryType.HISTOGRAM = 'histogram'
SummaryType.IMAGE = 'image'


class Vizer:
  """
  Vizer for graph. Responsible for summary and dump visualization.
  """
  def __init__(self, log_dir):
    # ouput dir
    self.log_dir = log_dir

    # summary graph
    self.summary_writer = \
        tf.summary.FileWriter(self.log_dir)


  def add_summary(self,
                   sess,
                   sum_op,
                   feed_dict,
                   step):
    """Add summaries"""
    sum_str = sess.run(sum_op,
                       feed_dict=feed_dict)
    self.summary_writer.add_summary(sum_str, step)


  def _sum_tensor(self, x, name=None):
    """Add summaries (histogram/sparsity) for tensor x."""
    if name is None:
      name = x.op.name
    tf.summary.histogram(name + '/histogram', x)
    tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(x))


  def _sum_losses(self, total_loss):
    """Add losses summaries."""
    loss_avg = tf.train.ExponentialMovingAverage(
                   0.9,
                   name='avg')
    loss_list = tf.get_collection(GKeys.LOSSES)
    loss_avg_op = loss_avg.apply(loss_list)

    for l in loss_list:
      avg_l = loss_avg.average(l)
      tf.summary.scalar(l.op.name+'_raw', l)
      tf.summary.scalar(l.op.name+'_avg', avg_l)

    return loss_avg_op

import tensorflow as tf
from tf import GraphKeys as GKeys

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
    self.summary_op = tf.summary.merge_all()

    # placeholders & summarize op
    self.plds = edict()
    self.pld_sum_ops = edict()


  def add_pld(self,
              name,
              shape=[],
              sum_type=SummaryType.SCALAR,
              dtype=tf.float32,
              kwargs):
    """Add placeholder and its summary op"""
    pld = tf.placeholder(tf.float32, shape)
    assert sum_type in SummaryType.keys(), \
        "Invalid summary type {}".format(sum_type)
    if sum_type == SummaryType.SCALAR:
      sum_op = tf.summary.scalar(name,
                                 pld,
                                 **kwargs)
    elif sum_type == SummaryType.IMAGE:
      sum_op = tf.summary.image(name,
                                pld,
                                **kwargs)
    elif sum_type == SummaryType.HISTOGRAM:
      sum_op = tf.summary.histogram(name,
                                    pld,
                                    **kwargs)
    self.plds[name] = pld
    self.pld_sum_ops[name] = sum_op


  def add_model_summary(self,
                        sess,
                        feed_dict,
                        step):
    """Add model graph summaries"""
    self._add_summary(sess,
                      self.summary_op,
                      feed_dict,
                      step)


  def add_pld_summary(self,
                      sess,
                      pld_v_pairs,
                      step):
    """Add summaries of placeholders"""
    feed_dict = {}
    sum_op_list = []
    for name,value in pld_v_pairs:
      pld = self.plds[name]
      sum_op = self.pld_sum_ops[name]
      feed_dict[name] = value
      sum_op_list.append(sum_op)
    self._add_summary(sess,
                      sum_op_list,
                      feed_dict,
                      step)


  def _add_summary(self,
                   sess,
                   op_list,
                   feed_dict,
                   step):
    """Add summaries"""
    sum_str = sess.run(op_list,
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
      tf.summary.scalar(l.op.name+'(raw)', l)
      tf.summary.scalar(l.op.name+'(avg)', avg_l)

    return loss_avg_op

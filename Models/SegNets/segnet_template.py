import tensorflow as tf
import numpy as np

from Networks.encoder_decoder import ENCODER_DECODER
from Datasets.dataset import SET
from vizer import SummaryType


class SegNet_Template(ENCODER_DECODER):
  """SegNet template"""
  def __init__(self,
               cfg,
               dataset,
               initer,
               vizer,
               save_dir):
    ENCODER_DECODER.__init__(self,
                             cfg,
                             dataset,
                             initer,
                             vizer,
                             save_dir)
    # output dir
    self.save_dir = save_dir


  def build(self):
    """Total Graph of SegNet"""
    with tf.variable_scope('input') as scope:
      im_shape = [None] + self.dataset.datum_shapes[1]
      la_shape = [None] + self.dataset.datum_shapes[2]
      self.fed.images = tf.placeholder(
          tf.float32, im_shape,
          name='images')
      self.fed.labels = tf.placeholder(
          tf.int32, la_shape,
          name='labels')

    # train op
    self.train_op = None

    # graph output
    self.logits = None
    self.total_loss = None

    # eval summary placeholders
    pld_names = ['mean_acc',
                 'mean_loss',
                 'mean_iu',
                 'mean_fps']
    for name in pld_names:
      self.vizer.add_pld(name)


  def train(self):
    """Perform SegNet training"""
    op_list = [self.logits,
               self.total_loss,
               self.fed.labels,
               self.train_op]

    for step in xrange(self.cfg.max_steps):
      # Get feed_dict
      feed_dict = self._feed_dict(SET.TRAIN)
      print feed_dict.keys()
      print "images shape {}".format(feed_dict[self.fed.images])
      print "labels shape {}".format(feed_dict[self.fed.labels])

      # Optimization
      out_list, t_cost = self._forward(
                             op_list,
                             feed_dict)
      logits, loss, labels, _ = out_list
      self._check_loss(loss)

      # Print train info
      if step % 10 == 0:
        self._print_info(step, loss, t_cost)
        self.eval_hist.update(logits,
                              labels,
                              clean=True)
        self.eval_hist.printout()

      # Validation & Summary
      if step % 100 == 0:
        self.vizer.add_model_summary(
                       self.sess,
                       feed_dict,
                       step)
        print "Start validating ..."
        self.eval(SET.VAL, step)
        print "end validating ..."

      # Save checkpoint
      if step % 1000 == 0 or (step+1) == max_steps:
        self.saver.save(self.sess,
                        self.save_dir,
                        global_step=step)


  def eval(self,
           skey,
           train_step=None):
    """Evaluate on the whole set"""
    op_list = [self.logits,
               self.total_loss,
               self.fed.labels]
    steps, n_examples = \
        self.dataset.epoch_info(skey)
    total_loss = 0.0
    total_time = 0.0
    preds = []
    labels = []

    # evaluate
    for step in xrange(steps):
      feed_dict = self._feed_dict(skey)
      our_list, t_cost = self._forward(
                             fw_op_list,
                             feed_dict)
      logits, loss, labels = our_list
      preds.extend(logits)
      labels.extend(labels)
      total_loss += loss
      total_time += t_cost

    # slice out duplicates
    preds = np.vstack(preds)[:n_examples]
    labels = np.vstack(labels)[:n_examples]

    # print hist
    self.eval_hist.update(preds,
                          labels,
                          clean=True)
    self.eval_hist.printout()

    # add summary
    if skey == SET.VAL:
      assert train_step is not None, \
          "Assign train step when eval val-set"
      values = [self.eval_hist.mean_acc,
                total_loss / n_examples,
                self.eval_hist.mean_iu,
                total_time / n_examples]
      self.vizer.add_pld_summary(
                     self.sess,
                     zip(pld_names, values),
                     train_step)

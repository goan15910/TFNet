import tensorflow as tf
import numpy as np

from Networks.encoder_decoder import ENCODER_DECODER
from Datasets.DS_base.dataset import SET
from Config.model_config import config
from vizer import SummaryType
from initer import InitMethod


class Simple_SegNet(ENCODER_DECODER):
  """Simple SegNet"""
  def __init__(self,
               dataset,
               initer,
               vizer,
               save_dir):
    # customize config here
    cfg = config
    cfg.batch_size = 4
    cfg.max_steps = 20000

    # encoder template
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

    # Normalize input
    with tf.variable_scope('input_norm') as scope:
      norm1 = self._norm(self.fed.images,
                         name='norm1')

    # Encoder
    encoder_out = self._encoder(norm1)

    # Decoder
    decoder_out = self._decoder(encoder_out)

    # Logits
    with tf.variable_scope('classifier') as scope:
      self.logits = self.conv(
                      decoder_out,
                      [1, 1, 64, self.n_classes],
                      init_method=InitMethod.MSRA,
                      act=False,
                      bn=False,
                      name='conv_out')

    # Loss
    # TODO: add loss weights
    onehot_labels = self._onehot_labels(
                        self.fed.labels,
                        self.n_classes)
    xentropy_loss = self._cross_entropy_loss(
                        self.logits,
                        onehot_labels,
                        loss_weights=None)
    self.total_loss = self._total_loss(
                        collections=[self.GKeys.LOSSES],
                        name='total_loss')

    # Train op
    self.train_op = self._train_op(self.total_loss)

    # train_sum_op
    self.train_sum_op = tf.summary.merge_all()

    # eval placeholders
    self.mean_acc = tf.placeholder(
                        tf.float32,
                        name='mean_acc')
    self.mean_loss = tf.placeholder(
                        tf.float32,
                        name='mean_loss')
    self.mean_iu = tf.placeholder(
                        tf.float32,
                        name='mean_iu')
    self.mean_fps = tf.placeholder(
                        tf.float32,
                        name='mean_fps')

    # eval_sum_op
    tf.summary.scalar(
        'mean_acc',
        self.mean_acc,
        collections=[self.GKeys.EVAL_SUMMARIES])
    tf.summary.scalar(
        'mean_loss',
        self.mean_loss,
        collections=[self.GKeys.EVAL_SUMMARIES])
    tf.summary.scalar(
        'mean_iu',
        self.mean_iu,
        collections=[self.GKeys.EVAL_SUMMARIES])
    tf.summary.scalar(
        'mean_fps',
        self.mean_fps,
        collections=[self.GKeys.EVAL_SUMMARIES])
    self.eval_sum_op = \
      tf.summary.merge_all(self.GKeys.EVAL_SUMMARIES)


  def train(self):
    """Perform SegNet training"""
    op_list = [self.logits,
               self.total_loss,
               self.fed.labels,
               self.train_op]

    while self.step < self.max_steps:
      # step 
      step = self.step

      # Set feed_dict
      feed_dict = self._feed_dict(SET.TRAIN)

      # Optimization
      out_list, t_cost = self._forward(
                             op_list,
                             feed_dict)
      logits, loss, labels, _ = out_list
      self._check_loss(loss)

      # Print train info
      if step % 10 == 0:
        self._print_info(step, loss, t_cost)

      # Validation & Summary
      if step % 100 == 0:
        self.vizer.add_summary(
                       self.sess,
                       self.train_sum_op,
                       feed_dict,
                       step)
        print "Start validating ..."
        self.eval(SET.VAL, step)
        print "end validating ..."

      # Save checkpoint
      if step % 1000 == 0 or (step+1) == self.max_steps:
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
    steps = self.dataset.epoch_steps(skey)
    n_examples = self.dataset.n_examples(skey) 
    total_loss = 0.0
    total_time = 0.0
    all_preds = []
    all_labels = []

    # evaluate
    for step in xrange(steps):
      feed_dict = self._feed_dict(skey)
      our_list, t_cost = self._forward(
                             op_list,
                             feed_dict)
      logits, loss, labels = our_list
      all_preds.append(logits)
      all_labels.append(labels)
      total_loss += loss
      total_time += t_cost

    # slice out duplicates
    all_preds = np.vstack(all_preds)[:n_examples]
    all_labels = np.vstack(all_labels)[:n_examples]

    # print hist
    self.eval_hist.update(all_preds,
                          all_labels,
                          clean=True)
    self.eval_hist.printout()

    # add summary
    feed_keys = [self.mean_acc,
                 self.mean_loss,
                 self.mean_iu,
                 self.mean_fps]
    values = [self.eval_hist.acc,
              total_loss / n_examples,
              self.eval_hist.mean_iu,
              total_time / n_examples]
    feed_dict = dict(zip(feed_keys, values))
    self.vizer.add_summary(
                   self.sess,
                   self.eval_sum_op,
                   feed_dict,
                   train_step)


  def _encoder(self, inputT):
    with tf.variable_scope('encoder') as scope:
      conv1 = self.conv(inputT, [7, 7, 3, 64], name="conv1")
      pool1 = self.maxpool(conv1, name='pool1')

      conv2 = self.conv(pool1, [7, 7, 64, 64], name="conv2")
      pool2 = self.maxpool(conv2, name='pool2')

      conv3 = self.conv(pool2, [7, 7, 64, 64], name="conv3")
      pool3 = self.maxpool(conv3, name='pool3')

      conv4 = self.conv(pool3, [7, 7, 64, 64], name="conv4")
      pool4 = self.maxpool(conv4, name='pool4')

      return pool4


  def _decoder(self, inputT):
    with tf.variable_scope('decoder') as scope:
      up4 = self.deconv(inputT, [self.batch_size, 45, 60, 64], name="up4")
      conv_up4 = self.conv(up4, [7, 7, 64, 64], act=False, name="conv_up4")

      up3 = self.deconv(conv_up4, [self.batch_size, 90, 120, 64], name="up3")
      conv_up3 = self.conv(up3, [7, 7, 64, 64], act=False, name="conv_up3")

      up2 = self.deconv(conv_up3, [self.batch_size, 180, 240, 64], name="up2")
      conv_up2 = self.conv(up2, [7, 7, 64, 64], act=False, name="conv_up2")

      up1 = self.deconv(conv_up2, [self.batch_size, 360, 480, 64], name="up1")
      conv_up1 = self.conv(up1, [7, 7, 64, 64], act=False, name="conv_up1")

      return conv_up1

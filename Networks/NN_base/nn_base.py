import tensorflow as tf
from tf import GraphKeys as GKeys

import os, sys
import numpy as np
import math
from math import ceil
from easydict import EasyDict as edict

from utils import join_key_mapping


ACT = edict()
ACT.SOFTMAX = 'softmax'
ACT.SIGMOID = 'sigmoid'


class NN_BASE:
  """Base components of general NN structure"""
  def __init__(self,
               config,
               dataset,
               initer,
               vizer,
               save_dir):
    # Default graph components
    self.phase_train = tf.placeholder(tf.bool, name='phase_train')
    self.global_step = tf.Variable(0, trainable=False)

    # Collection keys
    self.GKeys = edict()
    self.GKeys.update(GKeys)
    self.GKeys.BATCH_NORM = 'batch_norm'

    # Config
    self.config = config

    # Session
    sess_config = tf.ConfigProto()
    sess_config.gpu_option.allow_soft_placement = True
    sess_config.gpu_option.allow_growth = True
    self.sess = tf.Session(config=sess_config)

    # Saver
    self.saver = tf.train.Saver(tf.global_variables())

    # Initer
    self.initer = initer

    # Vizer
    self.vizer = vizer

    # Dataset
    self.dataset = dataset

    # Eval-hist
    self.eval_hist = \
        Eval_hist(self.dataset.num_classes)

    # Fed data dict
    self.fed = edict()


  @property
  def input_nodes(self):
    return input_dict.values()


  @property
  def input_keys(self):
    return input_dict.keys()


  @property
  def conv2d_init(self):
    return self.initer.conv2d_init


  @property
  def bn_init(self):
    return self.initer.bn_init


  def restore(self):
    # TODO
    pass


  def done(self):
    self.dataset.done()
    self.sess.close()


  def _feed_dict(self, skey):
    """Feed batch dict into class dict fed"""
    batch_dict = self.dataset.batch(skey)
    return join_keys_mapping(self.fed,
                             batch_dict)


  def _forward(self,
               op_list,
               feed_dict):
    """One forward pass"""
    tic = time.time()
    out = self.sess.run(op_list, feed_dict=feed_dict)
    toc = time.time() - tic
    return out, toc


  def _check_loss(self, loss):
    """Check wether loss is NaN"""
    assert not np.isnan(loss), "Model diverged with loss = NaN"


  def _print_info(self,
                  step,
                  loss,
                  t_cost):
    """Print the graph output info"""
    format_str = ('{0}: step {1}, loss = {2:.2f}' \
                  '({3:.3f} sec/batch)')
    print format_str.format(datetime.now(), \
                            step, loss, t_cost)


  def _var(self,
           name,
           shape,
           collections=None,
           initializer=None,
           trainable=True,
           device='/cpu:0',
           dtype=tf.float32):
    """Variable declaration for NN-base"""
    collections = set(collections)
    collections.add(GKeys.GLOBAL_VARIABLES)
    var = tf.contrib.framework.variable(
              name=name,
              shape=shape,
              collections=list(collections),
              initializer=initializer,
              trainable=trainable,
              device=device,
              dtype=dtype
          )
    if GKeys.TRAIN_OP not in collections:
      tf.contrib.framework.add_model_variable(var)
    return var


  def _add_weight_decay(self, var, wd):
    """Add weight decay to the variable"""
    wd_loss = tf.multiply(tf.nn.l2_loss(var),
                          wd,
                          name='weight_loss')
    tf.add_to_collection(GKeys.LOSSES, wd_loss)


  def _train_op(self,
                loss,
                opt=None):
    """train-op"""
    # Default optimizer
    if opt is None:
      opt = tf.train.AdamOptimizer(self.config.lr)

    loss_avg_op = self.vizer._sum_losses(loss)

    # Compute gradients.
    with tf.control_dependencies([loss_avg_op]):
      grads = opt.compute_gradients(loss)
      apply_grads_op = opt.apply_gradients(
          grads, global_step=self.global_step)

    # Add trainable variables to vizer
    for var in tf.trainable_variables():
      self.vizer._sum_tensor(var)

    # Add grads to vizer
    for grad, var in grads:
      if grad is not None:
        sum_name = var.op.name + '/grads'
        self.vizer._sum_tensor(grad, sum_name)

    with tf.control_dependencies([apply_grads_op]):
      train_op = tf.no_op(name='train')

    return train_op


  def _total_loss(self,
                  collections=None,
                  name=None):
    """Compute total loss value in the collections"""
    if collections is None:
      collections = [GKeys.LOSSES]
    loss_vars = []
    for key in collections:
      loss_vars.extend(tf.get_collection(GKeys.LOSSES))
    total_loss = tf.add_n(loss_vars, name=name)


  def _cross_entropy_loss(self,
                          logits,
                          labels,
                          loss_weights=None,
                          act=ACT.SOFTMAX,
                          collections=None):
    """
    Mean cross entropy
    Args:
      logits: logits tensor
      labels: shape [..., n_classes]
      loss_weights: shape [n_classes] / None
      act: activation function
    Return:
      mean cross-entropy loss
    """
    with tf.variable_scope('loss'):
      n_cls = labels.get_shape().as_list()[-1]
      epsilon = tf.constant(value=1e-10)

      # preprocess logits
      logits = tf.reshape(logits, (-1, n_cls))
      logits = logits + epsilon
      assert act in ACT.keys(), \
          "Invalid activation type {}".format(act)
      if act == ACT.SOFTMAX:
        logits = tf.nn.softmax(logits) + epsilon
      elif act == ACT.SIGMOID:
        logits = tf.nn.sigmoid(logits) + epsilon

      # preprocess labels
      labels = tf.cast(labels, tf.int32)

      # compute cross entropy mean
      if loss_weights is None:
        ce = -1 * tf.reduce_sum(
                 labels * tf.log(logits),
                 reduction_indices=[1],
                 name='cross_entropy')
      else:
        ce = -1 * tf.reduce_sum(
                 labels * tf.log(logits) * loss_weights,
                 reduction_indices=[1],
                 name='cross_entropy')
      ce_mean = tf.reduce_mean(ce, name='cross_entropy_mean')

      # manage collections
      if collections is None:
        collections = [GKeys.LOSSES]
      elif GKeys.LOSSES not in collections:
        collections.extend(GKeys.LOSSES)

      for key in collections:
        tf.add_to_collection(key, ce_mean)

      return ce_mean


  def _onehot_labels(self,
                     labels,
                     n_classes,
                     axis=-1):
    """
    Create an onehot tensor
    Args:
      inputT: shape of [batch_size] / [batch_size, 1]
      n_classes: the size of final vectors
    Return:
      An onehot [batch_size, size] tensor
      with dtype of int32
    """
    onehot_labels = tf.one_hot(labels,
                               depth=n_classes,
                               axis=axis)
    return onehot_labels


  def _flatten(self, inputT, size):
    """Flatten tensor to shape [-1, size]"""
    return tf.reshape(inputT, (-1, size))


  def _conv2d(self,
              inputT,
              shape,
              stride,
              padding,
              bias,
              init,
              wd=None):
    assert len(shape) == 4, \
        ("conv2d requires shape of format " \
         "(k1, k2, in_c, out_c)")
    k_size, _, in_c, out_c = shape
    with tf.variable_scope('conv2d') as scope:
      kernel = self._var('weights',
                         shape=shape,
                         collections=[GKeys.WEIGHTS],
                         initializer=init.weights)
      conv = tf.nn.conv2d(inputT,
                          kernel,
                          [1, stride, stride, 1],
                          padding=padding)

      if wd is not None:
        self._add_weight_decay(kernel, wd)

      if bias:
        biases = self._var('biases',
                           shape=[out_c],
                           collections=[GKeys.BIASES],
                           initializer=init.biases)
        conv = tf.nn.bias_add(conv, biases)

    return conv_out


  def _dilated_conv2d(self,
                      inputT,
                      shape,
                      dilation,
                      padding,
                      bias,
                      init,
                      wd=None):
    assert len(shape) == 4, \
        ("dilated_conv2d requires shape" \
         " of format (k, k, in_c, out_c)")
    k_size, _, in_c, out_c = shape
    with tf.variable_scope('dilated_conv2d') as scope:
      kernel = self._var('weights',
                         shape=shape,
                         collections=[GKeys.WEIGHTS],
                         initializer=init.weights)
      conv = tf.nn.atrous_conv2d(inputT,
                                 kernel,
                                 rate=dilation,
                                 padding=padding)

      if wd is not None:
        self._add_weight_decay(kernel, wd)

      if bias:
        biases = self._var('biases',
                           shape=[out_c],
                           collections=[GKeys.BIASES],
                           initializer=init.biases)
        conv = tf.nn.bias_add(conv, biases)

    return conv_out


  def _max_pool(self,
                inputT,
                ksize,
                stride,
                padding,
                name):
    k_h, k_w = ksize
    if stride is None:
      strides = None
    else:
      strides = [1, stride, stride, 1]
    return tf.nn.max_pool(inputT,
                      ksize=[1, k_h, k_w, 1],
                      strides=strides,
                      padding=padding,
                      name=name)


  def _avg_pool(self,
                inputT,
                ksize,
                stride,
                padding,
                name):
    k_h, k_w = ksize
    if stride is None:
      strides = None
    else:
      strides = [1, stride, stride, 1]
    return tf.nn.avg_pool(inputT,
                      ksize=[1, k_h, k_w, 1],
                      strides=strides,
                      padding=padding,
                      name=name)


  def _global_avg_pool(self, inputT, name):
    _, H, W, _ = inputT.get_shape().as_list()
    return self._avg_pool(inputT,
                          ksize=(H, W),
                          stride=None,
                          padding='VALID',
                          name=name)


  def _relu(self, inputT):
    return tf.nn.relu(inputT)


  def _batch_norm(self,
                  inputT,
                  is_training,
                  center=True,
                  scale=True,
                  bn_init=None):
    return tf.cond(is_training,
        lambda: tf.contrib.layers.batch_norm(inputT,
                    is_training=True,
                    center=True,
                    scale=True,
                    param_initializers=bn_init,
                    updates_collections=None,
                    variables_collections=[ \
                      self.GKeys.BATCH_NORM],
                    scope="batch_norm",
                    reuse=None), \
        lambda: tf.contrib.layers.batch_norm(inputT,
                    is_training=False,
                    center=True,
                    scale=True,
                    updates_collections=None,
                    scope="batch_norm",
                    reuse=True))


  def _resize(self, inputT, shape, method=None):
    """
    Resize 4D-tensors into new shape.
    Args:
      inputT: 4-D tensors of shape (N, H, W, C)
      shape: 1-D tensor of value (new_H, new_W)
      method: interpolation method
    """
    if method is None:
      method = tf.image.ResizeMethod.BILINEAR
    return tf.image.resize_images(inputT,
                                  shape,
                                  method=method)


  def _deconv2d(self,
                inputT,
                weights,
                output_shape,
                stride,
                padding):
    """2D Deconvolution"""
    strides = [1, stride, stride, 1]
    with tf.variable_scope('deconv2d') as scope:
      deconv = tf.nn.conv2d_transpose(inputT,
                                      weights,
                                      output_shape,
                                      strides=strides,
                                      padding=padding)
    return deconv


  def _bilinear_weights(self,
                        ksize,
                        out_c):
    """
    Bilinear upsampled weights
    Ref: https://github.com/MarvinTeichmann/tensorflow-fcn
    """
    f = ceil(k_size / 2.)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros((ksize, ksize))
    for x in xrange(ksize):
        for y in xrange(ksize):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights_shape = [ksize, ksize, out_c, out_c]
    weights = np.zeros(weights_shape)
    for i in xrange(out_c):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)

    return tf.get_variable(name="bilinear_weights",
                           initializer=init,
                           shape=weights_shape)


  def _concat(self, T_list, axis=3, name=None):
    with tf.variable_scope(name) as scope:
      return tf.concat(T_list, axis)


  def _norm(self,
            inputT,
            depth_radius=5,
            bias=1.0,
            alpha=0.0001,
            beta=0.75,
            name=None):
    return tf.nn.lrn(self.input_dict.images,
                     depth_radius=depth_radius,
                     bias=bias,
                     alpha=alpha,
                     beta=beta,
                     name=name)


  def _bottleneck(self,
                  inputT,
                  vanilla_conv,
                  conv_params,
                  bn,
                  name):
    """
    Bottleneck architecture in Resnet.
    Appears in "Deep Residual Learning for Image Recognition",
    https://arxiv.org/abs/1512.03385
    Args:
    """
    with tf.variable_scope(name) as scope:
      in_c = inputT.get_shape().as_list()[-1]

      # Reduce-block
      reduce_out = vanilla_conv(
                       inputT,
                       name=name+'_reduce',
                       **conv_params.reduce)

      # Center-block
      center_out = vanilla_conv(
                       inputT,
                       name=name+'_center',
                       **conv_params.center)

      # Increase-block
      increase_out = vanilla_conv(
                         inputT,
                         name=name+'_increase',
                         **conv_params.increase)

      # Proj-block
      if inc != h_dim:
        skip_out = vanilla_conv(
                       inputT,
                       name=name+'_proj',
                       **conv_params.proj)
      else:
        skip_out = inputT

      # Merge-block
      out = increase_out + skip_out
      if bn:
        out = self._batch_norm(out, self.phase_train)
      out = self._relu(out)

      return out

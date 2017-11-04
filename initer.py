import tensorflow as tf

import os, sys
import numpy as np
from easydict import EasyDict as edict


# Initialization method keys
InitMethod = edict()
InitMethod.XAVIER = 'xavier'
InitMethod.MSRA = 'msra'
InitMethod.ORTHOGONAL = 'orthogonal'


class Initer:
  """
  Initializer for graph.
  """
  def __init__(self,
               npy_file=None):
    self._npy_init = None
    self._ckpt_fname = None
    self.init_method = None

    # try to load npy_file
    if npy_file is not None:
      self.load_npy(npy_file)


  def set_default_method(self, method):
    self._check_init_method(method)
    self.init_method = method


  def conv2d_init(self,
                  shape,
                  name,
                  method=None):
    """Initializer for conv2d / dilated_conv2d"""
    assert len(shape) == 4, \
        ("conv2d requires shape of format " \
         "(k1, k2, in_c, out_c)")

    init = self._layer_init(name)
    if init is not None:
      return init
    else:
      init = edict()
      init.biases = tf.constant_initializer(0.0)
      init.weights = self._var_init(shape, method)
      return init


  def bn_init(self, name):
    """Initializer for batch norm"""
    caffe_init = self._layer_init(name)
    if caffe_init is not None:
      tf_init = edict()
      tf_init.moving_mean = caffe_init['mean']
      tf_init.moving_variance = caffe_init['variance']
      tf_init.gamma = caffe_init['scale']
      tf_init.beta = caffe_init['offset']
      return tf_init
    else:
      return None


  def bottleneck_init(self,
                      conv_params,
                      bn,
                      name,
                      method=None):
    # TODO
    pass


  def load_npy(self, npy_file):
    """Load initializer from npy file"""
    weight_npy = np.load(npy_file)
    w_dict = weight_npy.item()
    self._npy_init = {}
    for layer,params in w_dict.iteritems():
      init = edict()
      init.weights = tf.constant_initializer(
                         params['weights'],
                         dtype=tf.float32)
      init.biases = tf.constant_initializer(
                         params['biases'],
                         dtype=tf.float32)
      self._npy_init[layer] = init


  def _var_init(self,
                shape,
                method):
    """Get init for var"""
    try:
      self._check_init_method(method)
    except TypeError:
      assert self._set_default_method_already(), \
          "Set default method first!"
      method = self.init_method

    if method == InitMethod.XAVIER:
      return tf.contrib.layers.xavier_initializer()
    elif method == InitMethod.MSRA:
      return self._msra_init(shape)
    elif method == InitMethod.ORTHOGONAL:
      return self._orthogonal_init(shape)


  def _layer_init(self, name):
    """Get init for layer by name"""
    try:
      layer_init = self._npy_init[name]
      print "Initialize {} by pretrained params".format(name)
      return layer_init
    except KeyError:
      print ("New layers {} found. " \
          "Initialize by default method").format(name)
      return None
    except TypeError:
      print "Initialize {} from scratch".format(name)
      return None


  def _convert_resnet_npy_init(self):
    """Convert Resnet npy_init to customize format"""
    # TODO
    pass


  def _msra_init(self,
                 shape):
      """
      MSRA initializer. Reference: he et al., 2015
      https://arxiv.org/abs/1512.03385
      """
      stddev = np.sqrt(2. / np.prod(shape))
      return tf.truncated_normal_initializer(stddev=stddev)


  def _orthogonal_init(self,
                       shape,
                       dtype=tf.float32,
                       partition_info=None,
                       scale=1.1):
      """
      From Lasagne and Keras. Reference: Saxe et al.,
      http://arxiv.org/abs/1312.6120
      """
      def _init(shape,
                dtype=tf.float32,
                partition_info=None):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape) #this needs to be corrected to float32
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
      return _init


  def _set_default_method_already(self):
    if self.init_method is not None:
      return True
    else:
      return False


  def _check_init_method(self, method):
    if method is None:
      raise TypeError
    else:
      assert method in InitMethod.values(), \
        "Invalid initialization method {}".format(method)

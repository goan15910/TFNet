import os, sys
import numpy as np
import math
from math import ceil

# modules
from NN_base.nn_base import NN_BASE
from Initer import InitMethod


class ENCODER_DECODER(NN_BASE):
  """Encoder-Decoder structure"""
  def __init__(self,
               config,
               dataset,
               initer,
               vizer,
               save_dir):
    # Default init
    NN_BASE.__init__(self,
                     config,
                     dataset,
                     initer,
                     vizer,
                     save_dir)

    # Default fed-dict
    self.fed.phase_train = self.phase_train

    # Initer default method
    self.initer.set_default_method(InitMethod.ORTHOGONAL)


  def build(self):
    raise NotImplementedError


  def loss(self, logits, labels):
    raise NotImplementedError


  def conv(self,
           inputT,
           shape,
           stride=1,
           padding='SAME',
           bias=True,
           dilation=None,
           act=True,
           bn=True,
           init_method=None,
           name=None):
    """Convolution"""
    with tf.variable_scope(name) as scope:
      conv_init = self.conv2d_init(
                             shape,
                             name,
                             init_method)
      bn_init = self.bn_init(name)
      wd = self.config.wd

      if dilation is None:
        out = self._conv2d(inputT,
                           shape,
                           stride,
                           padding,
                           bias,
                           conv_init,
                           wd)
      else:
        out = self._dilated_conv2d(inputT,
                                   shape,
                                   dilation,
                                   padding,
                                   bias,
                                   conv_init,
                                   wd)

      if bn:
        out = self._batch_norm(inputT,
                               self.phase_train,
                               bn_init=bn_init)

      if act:
        out = self._relu(out)

      return out


  def deconv(self,
             inputT,
             output_shape,
             ksize=2,
             stride=2,
             padding='SAME',
             name=None):
    """Deconvolution using bilinear upsampled weights"""
    batch_size, _, _, in_c = inputT.get_shape.as_list()
    weights = self._bilinear_weights(ksize, in_c)
    if output_shape[0] is None:
      output_shape[0] = batch_size
    out = self._deconv2d(inputT,
                         weights,
                         output_shape,
                         stride,
                         padding)
    return out


  def maxpool(self,
              inputT,
              ksize=(2, 2),
              stride=2,
              padding='SAME',
              name):
    return self._max_pool(inputT,
                          ksize,
                          stride,
                          padding,
                          name)


  def avgpool(self,
              inputT,
              ksize=(2, 2),
              stride=2,
              padding='SAME',
              name):
    return self._avg_pool(inputT,
                          ksize,
                          stride,
                          padding,
                          name)


  def interp(self,
             inputT,
             new_shape=None,
             factor=None,
             resize_type=None,
             name=None):
    """
    Interpolation with bilinear method. One could
    either specify new shape or the resize factor.
      Args:
        inputT: input tensors
        new_shape: the new shape to resize
        factor: the resize factor
        resize_type: either 'up' / 'down'
    """
    # Some input args check
    assert new_shape is None or len(new_shape) == 2, \
        "shape must be (H, W) / None"
    assert resize_type in ['up', 'down', None], \
        "resize_type is either up / down / None"
    assert (new_shape is None) or (factor is None), \
        "specify either shape or factor, not both"

    with tf.variable_scope(name) as scope:
      shape = inputT.get_shape().as_list()
      assert len(shape) == 4, \
          "Input tensors must be of format NHWC"
      N, H, W, C = input_shape
      if new_shape is not None:
        new_H, new_W = new_shape
      elif factor is not None:
        if resize_type == 'down':
          new_H = int(ceil(H / factor))
          new_W = int(ceil(W / factor))
        else:
          new_H = int(H * factor)
          new_W = int(W * factor)
      new_shape = (new_H, new_W)
      return self._resize(inputT, new_shape)


  # TODO: check weights init from original paper
  def bottleneck(self,
                 inputT,
                 l_dim,
                 h_dim,
                 stride=1,
                 dilation=None,
                 bn=False,
                 bias=False,
                 wd=None,
                 init_method=None,
                 name=None):
    """
    Bottleneck.

    Args:
      l_dim: reduce dimension in bottleneck
      h_dim: increase dimension in bottleneck
      stride: stride of conv in reduce-block
      dilation: dilation of conv in center-block
      bn: wether to use batch norm in bottleneck
      init_method: initialize method
      wd: weight decay factor of each convs
    """
    with tf.variable_scope(name) as scope:
      in_c = inputT.get_shape().as_list()[-1]

      # conv params
      conv_params = edict()

      # Reduce-block
      conv_params.reduce = edict()
      conv_params.reduce.shape = (1, 1, in_c, l_dim)
      conv_params.reduce.stride = stride
      conv_params.reduce.bn = bn
      conv_params.reduce.bias = bias
      conv_params.reduce.act = True
      conv_params.reduce.init_method = init_method
      conv_params.reduce.wd = wd

      # Center-block
      conv_params.center = edict()
      conv_params.center.shape = (3, 3, l_dim, l_dim)
      conv_params.center.dilation = dilation
      conv_params.center.bn = bn
      conv_params.center.bias = bias
      conv_params.center.act = True
      conv_params.center.init_method = init_method
      conv_params.center.wd = wd

      # Increase-block
      conv_params.increase = edict()
      conv_params.increase.shape = (1, 1, l_dim, h_dim)
      conv_params.increase.bn = bn
      conv_params.increase.bias = bias
      conv_params.increase.act = True
      conv_params.increase.init_method = init_method
      conv_params.increase.wd = wd

      # Proj-block
      conv_params.proj = edict()
      conv_params.proj.shape = (1, 1, inc, h_dim)
      conv_params.proj.stride = stride
      conv_params.proj.bn = bn
      conv_params.proj.bias = bias
      conv_params.proj.act = False
      conv_params.proj.init_method = init_method
      conv_params.proj.wd = wd

      # Bottleneck template
      out = self._bottleneck(inputT,
                             self.conv,
                             conv_params,
                             bn=bn,
                             name=name)

      return out


  def _pyramid_pool(self,
                    inputT,
                    ksize,
                    bn,
                    wd,
                    init_method,
                    name):
    """Single pyramid pooling module"""
    with tf.variable_scope(name) as scope:
      _, H, W, in_c = inputT.get_shape().as_list()
      out_c = inc / 4
      pool = self._avg_pool(inputT,
                            ksize,
                            ksize,
                            padding='VALID',
                            name='pool')
      conv = self.conv(inputT,
                       (1, 1, in_c, out_c),
                       stride=1,
                       bn=bn,
                       bias=False,
                       act=True,
                       init_method=init_method,
                       wd=wd,
                       name='conv')
      interp = self.interp(inputT,
                           new_shape=(H, W),
                           name='interp')
      return interp


  def spp(self,
          inputT,
          bn=False,
          wd=None,
          init_method=None,
          name=None):
    """
    Spatial-Pyramid-Pooling in "Pyramid Scene Parsing Network",
    https://arxiv.org/abs/1612.01105
    """
    # Compute the 4 pooling module parameters
    _, H, W, in_c = inputT.get_shape().as_list()
    assert H % 6 == 0, \
        "Feature map must be factor of 6"
    assert C % 4 == 0, \
        "Channel must be factor of 4"

    with tf.variable_scope(name) as scope:
      # pool1
      pyramid1 = self._pyramid_pool(
                     inputT,
                     ksize=H,
                     bn=bn,
                     wd=wd,
                     init_method=init_method,
                     name='pyramid1')

      # pool2
      pyramid2 = self._pyramid_pool(
                     inputT,
                     ksize=H/2,
                     bn=bn,
                     wd=wd,
                     init_method=init_method,
                     name='pyramid2')

      # pool3
      pyramid3 = self._pyramid_pool(
                     inputT,
                     ksize=H/3,
                     bn=bn,
                     wd=wd,
                     init_method=init_method,
                     name='pyramid3')

      # pool6
      pyramid4 = self._pyramid_pool(
                     inputT,
                     ksize=H/6,
                     bn=bn,
                     wd=wd,
                     init_method=init_method,
                     name='pyramid4')

      # concat
      T_list = [inputT, pyramid1, pyramid2, \
                pyramid3, pyramid4]
      concat = self._concat(T_list, name='concat')

      return concat


  def cff(self,
          lowT,
          highT,
          out_c,
          wd=None,
          name=None):
    """
    Cascaded-Feature-Fusion unit. Proposed in
    "ICNet for Real-Time Semantic Segmentation on High-Resolution Images".
    Ref: https://arxiv.org/abs/1704.08545
    Args:
      lowT: low resolution input
      highT: high resolution input
    """
    # Input dim
    _, l_h, l_w, low_c = lowT.get_shape().as_list()
    _, h_h, h_w, high_c = highT.get_shape().as_list()
    assert (h_h == 2 * l_h) and (h_w == 2 * l_w), \
        "Low-res must be half the size of high-res"

    with tf.variable_scope(name) as scope:
      # Upsample low-res input
      up_low = self.interp(lowT,
                           factor=2,
                           resize_type='up',
                           name='interp')
      conv_low = self.conv(up_low,
                           (1, 1, low_c, out_c),
                           dilation=2,
                           bn=False,
                           act=False,
                           wd=wd,
                           name='conv')

      # Project high-res input
      proj_high = self.conv(highT,
                            (1, 1, highT, out_c),
                            bn=False,
                            act=False,
                            wd=wd,
                            name='proj')

      # Sum module
      out = proj_high + conv_low
      out = self._relu(out)

      return out

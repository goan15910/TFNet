import tensorflow as tf
from segnet_template import SegNet_Template


class SegNet(SegNet_Template):
  """Original SegNet"""
  def __init__(self,
               config,
               dataset,
               initer,
               vizer,
               save_dir):
    SegNet_Template.__init__(self,
                             config,
                             dataset,
                             initer,
                             vizer,
                             save_dir)


  def build(self):
    SegNet_Template.build(self)

    # Normalize input
    with tf.variable_scope('input_norm') as scope:
      norm1 = self._norm(self.fed.images,
                         name='norm1')

    # Encoder
    encoder_out = self._encoder(norm1)

    # Decoder
    self.logits = self._decoder(encoder_out)

    # Loss
    xentropy_loss = self._cross_entropy_loss(
                        self.logits,
                        self.fed.labels,
                        loss_weights)
    self.total_loss = self._total_loss(
                        collections=[self.GKeys.LOSSES],
                        name='total_loss')

    # Train op
    self.train_op = self._train_op(self.total_loss)


  def _encoder(self, inputT):
    with tf.variable_scope('encoder') as scope:
      conv1_1 = self.conv(inputT, [3, 3, 3, 64], name="conv1_1")
      conv1_2 = self.conv(conv1_1, [3, 3, 64, 64], name="conv1_2")
      pool1 = self.maxpool(conv1_2, name='pool1')

      conv2_1 = self.conv(pool1, [3, 3, 64, 128], name="conv2_1")
      conv2_2 = self.conv(conv2_1, [3, 3, 128, 128], name="conv2_2")
      pool2 = self.maxpool(conv2_2, name='pool2')

      conv3_1 = self._conv_layer(pool2, [3, 3, 128, 256], name="conv3_1")
      conv3_2 = self._conv_layer(conv3_1, [3, 3, 256, 256], name="conv3_2")
      conv3_3 = self._conv_layer(conv3_2, [3, 3, 256, 256], name="conv3_3")
      pool3 = self.maxpool(conv3_3, name='pool3')

      conv4_1 = self._conv_layer(pool3, [3, 3, 256, 512], name="conv4_1")
      conv4_2 = self._conv_layer(conv4_1, [3, 3, 512, 512], name="conv4_2")
      conv4_3 = self._conv_layer(conv4_2, [3, 3, 512, 512], name="conv4_3")
      pool4 = self.maxpool(conv4_3, name='pool4')

      conv5_1 = self._conv_layer(pool4, [3, 3, 512, 512], name="conv5_1")
      conv5_2 = self._conv_layer(conv5_1, [3, 3, 512, 512], name="conv5_2")
      conv5_3 = self._conv_layer(conv5_2, [3, 3, 512, 512], name="conv5_3")
      pool5 = self.maxpool(conv5_3, name='pool5')

      return pool5


  def _decoder(self, inputT):
    with tf.variable_scope('decoder') as scope:
      up5 = self.deconv(inputT, [None, 23, 30, 512], name="up5")
      conv_decode5_3 = self.conv(up5, [3, 3, 512, 512], act=False, name="conv_decode5_3")
      conv_decode5_2 = self.conv(conv_decode5_3, [3, 3, 512, 512], act=False, name="conv_decode5_2")
      conv_decode5_1 = self.conv(conv_decode5_2, [3, 3, 512, 512], act=False, name="conv_decode5_1")

      up4 = self.deconv(conv_decode5_1, [None, 45, 60, 512], name="up4")
      conv_decode4_3 = self.conv(up4, [3, 3, 512, 512], act=False, name="conv_decode4_3")
      conv_decode4_2 = self.conv(conv_decode4_3, [3, 3, 512, 512], act=False, name="conv_decode4_2")
      conv_decode4_1 = self.conv(conv_decode4_2, [3, 3, 512, 256], act=False, name="conv_decode4_1")

      up3 = self.deconv(conv_decode4_1, [None, 90, 120, 256], name="up3")
      conv_decode3_3 = self.conv(up3, [3, 3, 256, 256], act=False, name="conv_decode3_3")
      conv_decode3_2 = self.conv(conv_decode3_3, [3, 3, 256, 256], act=False, name="conv_decode3_2")
      conv_decode3_1 = self.conv(conv_decode3_2, [3, 3, 256, 128], act=False, name="conv_decode3_1")

      up2 = self.deconv(conv_decode3_1, [None, 180, 240, 128], name="up2")
      conv_decode2_2 = self.conv(up2, [3, 3, 128, 128], act=False, name="conv_decode2_2")
      conv_decode2_1 = self.conv(conv_decode2_2, [3, 3, 128, 64], act=False, name="conv_decode2_1")

      up1 = self.deconv(conv_decode2_1, [None, 360, 480, 64], name="up1")
      conv_decode1_2 = self.conv(up1, [3, 3, 64, 64], act=False, name="conv_decode1_2")
      conv_decode1_1 = self.conv(conv_decode1_2, [3, 3, 64, 64], act=False, name="conv_decode1_1")

      return conv_decode1_1

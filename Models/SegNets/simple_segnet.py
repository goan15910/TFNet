import tensorflow as tf
from segnet_template import SegNet_Template


class Simple_SegNet(SegNet_Template):
  """Simple SegNet"""
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
                             initer,
                             save_dir)


  def build(self):
    SegNet_Template.build()

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
      up4 = self.deconv(inputT, [None, 45, 60, 64], name="up4")
      conv_up4 = self.conv(up5, [7, 7, 64, 64], act=False, name="conv_up4")

      up3 = self.deconv(conv_up4, [None, 90, 120, 64], name="up3")
      conv_up3 = self.conv(up3, [7, 7, 64, 64], act=False, name="conv_up3")

      up2 = self.deconv(conv_up3, [None, 180, 240, 64], name="up2")
      conv_up2 = self.conv(up2, [7, 7, 64, 64], act=False, name="conv_up2")

      up1 = self.deconv(conv_up2, [None, 360, 480, 64], name="up1")
      conv_up1 = self.conv(up1, [7, 7, 64, 64], act=False, name="conv_up1")

      return conv_up1

import tensorflow as tf
from Networks.encoder_decoder import ENCODER_DECODER


class PSPNet(ENCODER_DECODER):
  """PSPNet"""
  def __init__(self, config, sess, vizer, initer):
    ENCODER_DECODER.__init__(self, config, sess, initer)


  def build(self):
    """Total Graph of PSPNet"""
    # TODO: set image/label shape
    self.input_dict.images = tf.placeholder(
        tf.float32, [None, H, W, 3],
        name='images'
    )
    self.input_dict.labels = tf.placeholder(
        tf.int64, [None, H, W, 1],
        name='labels'
    )

    # Normalize input
    with tf.variable_scope('input_norm') as scope:
      norm1 = tf.nn.lrn(self.input_dict.images, \
                        depth_radius=5, \
                        bias=1.0, \
                        alpha=0.0001, \
                        beta=0.75, \
                        name='norm1')

    # Resnet-like Encoder
    bnk5_3 = self._encoder(norm1)

    # Spatial Pyrimid pooling
    # TODO

    # Decoder
    # TODO
    self.logits = 

    # Loss
    # TODO: weight decay, main loss


  def train(self, train_feed_dict, val_feed_dict):
    """Perform SegNet training"""
    opt_op_list = [self.logits, self.total_loss, \
                   self.input_dict.labels, \
                   self.train_op]
    # TODO: define maxsteps
    for step in xrange(max_steps):
      # Optimization
      out_list, t_cost = self._forward( \
                             opt_op_list, \
                             train_feed_dict)
      logits, loss, labels, _ = out_list
      self._check_loss(loss)
      
      # Print train info
      if step % 10 == 0:
        self._print_info(step, loss, t_cost)

      # Validation & Summary
      if step % 100 == 0:
        # TODO: summarize train graph
        print "Start validating ..."
        # TODO: set val_steps
        self.eval(val_feed_dict, val_steps)
        print "end validating ..."

      # Save checkpoint
      if step % 1000 == 0 or (step+1) == max_steps:
        # TODO


  def eval(self, feed_dict, max_steps):
    """Evaluate on the whole batch"""
    fw_op_list = [self.logits, self.total_loss, \
                  self.input_dict.labels]
    total_loss = 0.0
    total_time = 0.0
    # TODO: define n_classes
    hist = np.zeros((n_classes, n_classes))
    all_preds = []
    all_labels = []
    for step in xrange(max_steps):
      out_list, t_cost = self._forward( \
                             fw_op_list, \
                             feed_dict)
      # TODO: unpack out_list
      total_loss += # TODO: get loss from out_list
      total_time += t_cost
  
  
  def _encoder(self, inputT):
    with tf.variable_scope('resnet-encoder') as scope:
      conv1_1 = self.conv(inputT, [3, 3, 3, 64], wd=self.config.wd, name="conv1_1")
      conv1_2 = self.conv(conv1_1, [3, 3, 64, 64], wd=self.config.wd, name="conv1_2")
      pool1 = self.maxpool(conv1_2, name='pool1') 

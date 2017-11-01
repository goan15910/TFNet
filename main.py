import tensorflow as tf
from master import Master

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('mode', None, """ Either train / test """)
tf.app.flags.DEFINE_string('dataset', None, """ Dataset name """)
tf.app.flags.DEFINE_string('model', None, """ Model name """)
tf.app.flags.DEFINE_string('dataset_dir', None, """ Dataset root directory """)
tf.app.flags.DEFINE_string('pretrained', None, """ Pretrained npy file """)
tf.app.flags.DEFINE_integer('threads', 1, """ Number of threads to be used when loading dataset """)
tf.app.flags.DEFINE_string('save_dir', None, """ Save directory path """)


if __name__ == '__main__':

  master = Master(FLAGS)
  master.run()

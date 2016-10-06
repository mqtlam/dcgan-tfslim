import tensorflow as tf
import tensorflow.contrib.slim as slim

def lrelu(x, leak=0.2):
    """Leaky ReLU activation.

    Args:
        x: input
        leak: leak parameter [0.2]

    Returns:
        activation
    """
    return tf.maximum(x, leak*x)

class Discriminator():
    """Discriminator model.
    """
    def __init__(self, FLAGS):
        """Initialization.

        Args:
            FLAGS: flags object
        """
        self.f = FLAGS

    def __call__(self, image, reuse=False):
        """Discriminator function call for training.

        Args:
            image: input image (batch size, width, height, 3)
            reuse: reuse variables [False]

        Returns:
            probability of real image
                values in range of [0, 1]
                dimensionality is (batch size, 1)
        """
        with tf.name_scope('d/h0') as scope:
            reuse_scope = scope if reuse else None
            h0 = slim.conv2d(image, self.f.df_dim, [5, 5],
                stride=2,
                activation_fn=lrelu,
                reuse=reuse_scope,
                scope='d/h0')

        with tf.name_scope('d/h1') as scope:
            reuse_scope = scope if reuse else None
            h1 = slim.conv2d(h0, self.f.df_dim*2, [5, 5],
                stride=2,
                normalizer_fn=slim.batch_norm,
                activation_fn=lrelu,
                reuse=reuse_scope,
                scope='d/h1')

        with tf.name_scope('d/h2') as scope:
            reuse_scope = scope if reuse else None
            h2 = slim.conv2d(h1, self.f.df_dim*4, [5, 5],
                stride=2,
                normalizer_fn=slim.batch_norm,
                activation_fn=lrelu,
                reuse=reuse_scope,
                scope='d/h2')

        with tf.name_scope('d/h3') as scope:
            reuse_scope = scope if reuse else None
            h3 = slim.conv2d(h2, self.f.df_dim*8, [5, 5],
                stride=2,
                normalizer_fn=slim.batch_norm,
                activation_fn=lrelu,
                reuse=reuse_scope,
                scope='d/h3')

        with tf.name_scope('d/h4') as scope:
            reuse_scope = scope if reuse else None
            h3_shape = h3.get_shape()
            extra_dim = h3_shape[1].value*h3_shape[2].value*h3_shape[3].value
            h4 = tf.reshape(h3, [self.f.batch_size, extra_dim])
            h4 = slim.fully_connected(h4, 1,
                activation_fn=None,
                reuse=reuse_scope,
                scope='d/h4')

        return tf.nn.sigmoid(h4), h4

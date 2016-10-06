import tensorflow as tf
import tensorflow.contrib.slim as slim

class Generator():
    """Generator model.
    """
    def __init__(self, FLAGS):
        """Initialization.

        Args:
            FLAGS: flags object
        """
        self.f = FLAGS

    def __call__(self, z, reuse=False):
        """Generator function call for training.

        Args:
            z: input noise (batch size, z dim)
            reuse: reuse variables [False]

        Returns:
            generated image
                parameterized in range of [-1, 1]
                dimensionality is (batch size, width, height, 3)
        """
        # constants for convenience
        s = self.f.output_size
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
        gf = self.f.gf_dim
        gf2, gf4, gf8 = gf*2, gf*4, gf*8

        # project z and reshape
        with tf.name_scope('g/h0') as scope:
            reuse_scope = scope if reuse else None
            z_ = slim.fully_connected(z, s16*s16*gf8,
                activation_fn=None,
                reuse=reuse_scope,
                scope='g/h0')
            h0 = tf.reshape(z_, [-1, s16, s16, gf8])
            h0 = tf.nn.relu(slim.batch_norm(h0))

        # deconvolutions
        with tf.name_scope('g/h1') as scope:
            reuse_scope = scope if reuse else None
            h1 = slim.conv2d_transpose(h0, gf4, [5, 5],
                stride=2,
                normalizer_fn=slim.batch_norm,
                reuse=reuse_scope,
                scope='g/h1')

        with tf.name_scope('g/h2') as scope:
            reuse_scope = scope if reuse else None
            h2 = slim.conv2d_transpose(h1, gf2, [5, 5],
                stride=2,
                normalizer_fn=slim.batch_norm,
                reuse=reuse_scope,
                scope='g/h2')

        with tf.name_scope('g/h3') as scope:
            reuse_scope = scope if reuse else None
            h3 = slim.conv2d_transpose(h2, gf, [5, 5],
                stride=2,
                normalizer_fn=slim.batch_norm,
                reuse=reuse_scope,
                scope='g/h3')

        with tf.name_scope('g/h4') as scope:
            reuse_scope = scope if reuse else None
            h4 = slim.conv2d_transpose(h3, self.f.c_dim, [5, 5],
                stride=2,
                activation_fn=None,
                reuse=reuse_scope,
                scope='g/h4')

        return tf.nn.tanh(h4)

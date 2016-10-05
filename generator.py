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

    def __call__(self, z):
        """Generator function call for training.

        Args:
            z: input noise (batch size, z dim)

        Returns:
            generated image
                parameterized in range of [-1, 1]
                dimensionality is (batch size, width, height, 3)
        """
        s = self.f.output_size
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
        gf = self.f.gf_dim
        gf2, gf4, gf8 = gf*2, gf*4, gf*8

        # project z and reshape
        with tf.name_scope('g/h0') as h0_scope:
            z_ = slim.fully_connected(z, s16*s16*gf8,
                activation_fn=None,
                scope=h0_scope)
            h0 = tf.reshape(z_, [-1, s16, s16, gf8])
            h0 = tf.nn.relu(slim.batch_norm(h0))

        with tf.name_scope('g/h1') as h1_scope:
            h1 = slim.conv2d_transpose(h0, gf4, [5, 5],
                stride=2,
                normalizer_fn=slim.batch_norm,
                scope=h1_scope)

        with tf.name_scope('g/h2') as h2_scope:
            h2 = slim.conv2d_transpose(h1, gf2, [5, 5],
                stride=2,
                normalizer_fn=slim.batch_norm,
                scope=h2_scope)

        with tf.name_scope('g/h3') as h3_scope:
            h3 = slim.conv2d_transpose(h2, gf, [5, 5],
                stride=2,
                normalizer_fn=slim.batch_norm,
                scope=h3_scope)

        with tf.name_scope('g/h4') as h4_scope:
            h4 = slim.conv2d_transpose(h3, self.f.c_dim, [5, 5],
                stride=2,
                activation_fn=None,
                scope=h4_scope)

        return tf.nn.tanh(h4)

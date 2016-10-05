import tensorflow as tf

from ops import *

class Generator():
    """Generator model.
    """
    def __init__(self, FLAGS):
        """Initialization.

        Args:
            FLAGS: flags object
        """
        self.f = FLAGS
        # batch normalization layers
        self.g_bn0 = batch_norm(name='g/bn0')
        self.g_bn1 = batch_norm(name='g/bn1')
        self.g_bn2 = batch_norm(name='g/bn2')
        self.g_bn3 = batch_norm(name='g/bn3')

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
        z_, h0_w, h0_b = linear(z, s16*s16*gf8, 'g/h0/lin', with_w=True)

        h0 = tf.reshape(z_, [-1, s16, s16, gf8])
        h0 = tf.nn.relu(self.g_bn0(h0))

        h1, h1_w, h1_b = deconv2d(h0,
            [self.f.batch_size, s8, s8, gf4], name='g/h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(h1))

        h2, h2_w, h2_b = deconv2d(h1,
            [self.f.batch_size, s4, s4, gf2], name='g/h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, h3_w, h3_b = deconv2d(h2,
            [self.f.batch_size, s2, s2, gf], name='g/h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, h4_w, h4_b = deconv2d(h3,
            [self.f.batch_size, s, s, self.f.c_dim], name='g/h4', with_w=True)

        return tf.nn.tanh(h4)

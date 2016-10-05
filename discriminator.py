import tensorflow as tf

from ops import *

class Discriminator():
    """Discriminator model.
    """
    def __init__(self, FLAGS):
        """Initialization.

        Args:
            FLAGS: flags object
        """
        self.f = FLAGS
        # batch normalization layers
        self.d_bn1 = batch_norm(name='d/bn1')
        self.d_bn2 = batch_norm(name='d/bn2')
        self.d_bn3 = batch_norm(name='d/bn3')

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
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = lrelu(conv2d(image, self.f.df_dim, name='d/h0/conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.f.df_dim*2, name='d/h1/conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.f.df_dim*4, name='d/h2/conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.f.df_dim*8, name='d/h3/conv')))
        h4 = linear(tf.reshape(h3, [self.f.batch_size, -1]), 1, 'd/h3/lin')

        return tf.nn.sigmoid(h4), h4

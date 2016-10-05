import os
import tensorflow as tf

from generator import Generator
from discriminator import Discriminator

class DCGAN():
    """DCGAN model.
    """
    def __init__(self, sess, FLAGS):
        """Initialization.

        Args:
            sess: TensorFlow session
            FLAGS: flags object
        """
        # initialize variables
        self.sess = sess
        self.f = FLAGS

        # inputs: real (training) images
        images_shape = [self.f.output_size, self.f.output_size, self.f.c_dim]
        self.real_images = tf.placeholder(tf.float32, [self.f.batch_size] + images_shape, name="real_images")

        # inputs: z (noise)
        self.z = tf.placeholder(tf.float32, [None, self.f.z_dim], name='z')

        # initialize models
        generator = Generator(FLAGS)
        discriminator = Discriminator(FLAGS)

        # generator network
        self.G = generator(self.z)
        # discriminator network for real images
        self.D, self.D_logits = discriminator(self.real_images)
        # discriminator network for fake images
        self.D_, self.D_logits_ = discriminator(self.G, reuse=True)

        # losses
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        # create summaries
        self.__create_summaries()

        # organize variables
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if "d/" in var.name]
        self.g_vars = [var for var in t_vars if "g/" in var.name]

        # saver
        self.saver = tf.train.Saver()

    def save(self, step):
        """Save model.

        Postconditions:
            checkpoint directory is created if not found
            checkpoint directory is updated with new saved model

        Args:
            step: step of training to save
        """
        model_name = "DCGAN.model"
        model_dir = self.__get_model_dir()
        checkpoint_dir = os.path.join(self.f.checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        model_file_prefix = model_dir
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_file_prefix),
                        global_step=step)

    def checkpoint_exists(self):
        """Check if any checkpoints exist.

        Returns:
            True if any checkpoints exist
        """
        model_dir = self.__get_model_dir()
        checkpoint_dir = os.path.join(self.f.checkpoint_dir, model_dir)
        return os.path.exists(checkpoint_dir)

    def load(self):
        """Load model.

        Returns:
            True if model is loaded successfully
        """
        model_dir = self.__get_model_dir()
        checkpoint_dir = os.path.join(self.f.checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # load model
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def __create_summaries(self):
        """Helper function to create summaries.
        """
        # histogram summaries
        self.z_sum = tf.histogram_summary("z", self.z)
        self.d_sum = tf.histogram_summary("d/output/real", self.D)
        self.d__sum = tf.histogram_summary("d/output/fake", self.D_)

        # image summaries
        self.g_sum = tf.image_summary("G", self.G, max_images=9)

        # scalar summaries
        self.d_loss_real_sum = tf.scalar_summary("d/loss/real", self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary("d/loss/fake", self.d_loss_fake)
        self.d_loss_sum = tf.scalar_summary("d/loss/combined", self.d_loss)
        self.g_loss_sum = tf.scalar_summary("g/loss/combined", self.g_loss)

    def __get_model_dir(self):
        """Helper function to get the model directory.

        Returns:
            string of model directory
        """
        return "{0}_{1}_{2}_{3}".format(self.f.experiment_name,
                                        self.f.dataset,
                                        self.f.batch_size,
                                        self.f.output_size)

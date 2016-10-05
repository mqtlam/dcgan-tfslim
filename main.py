import os
import pprint
pp = pprint.PrettyPrinter()

import tensorflow as tf

from dcgan import DCGAN
from train import train

flags = tf.app.flags
# training params
flags.DEFINE_integer("epoch", 25, "Number of epochs to train. [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate for Adam optimizer [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of Adam optimizer [0.5]")
flags.DEFINE_integer("batch_size", 64, "Number of images in batch [64]")
# model params
flags.DEFINE_integer("output_size", 64, "Size of the output images to produce [64]")
flags.DEFINE_integer("z_dim", 100, "Dimension of input noise vector. [100]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("gf_dim", 64, "Dimension of generator filters in first convolution layer. [64]")
flags.DEFINE_integer("df_dim", 64, "Dimension of discriminator filters in first convolution layer. [64]")
# dataset params
flags.DEFINE_string("data_dir", "data", "Path to datasets directory [data]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA]")
# flags for running
flags.DEFINE_string("experiment_name", "experiment", "Name of experiment for current run [experiment]")
flags.DEFINE_boolean("train", False, "Train if True, otherwise test [False]")
flags.DEFINE_integer("sample_size", 64, "Number of images to sample [64]")
# directory params
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Path to save the checkpoint data [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Path to save the image samples [samples]")
flags.DEFINE_string("log_dir", "logs", "Path to log for TensorBoard [logs]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(FLAGS.__flags)

    # path checks
    if not os.path.exists(os.path.join(FLAGS.checkpoint_dir, FLAGS.experiment_name)):
        os.path.join(FLAGS.checkpoint_dir, FLAGS.experiment_name)
    if not os.path.exists(os.path.join(FLAGS.sample_dir, FLAGS.experiment_name)):
        os.makedirs(os.path.join(FLAGS.sample_dir, FLAGS.experiment_name))

    # training/inference
    with tf.Session() as sess:
        dcgan = DCGAN(sess, FLAGS)

        # load checkpoint if found
        if dcgan.checkpoint_exists():
            print("Loading checkpoints...")
            if dcgan.load():
                print "success!"
            else:
                raise IOError("Could not read checkpoints from {0}!".format(
                    FLAGS.checkpoint_dir))
        else:
            print "No checkpoints found. Training from scratch."
            dcgan.load()

        # train DCGAN
        if FLAGS.train:
            train(dcgan)
        else:
            dcgan.load()

        # inference code can go here

if __name__ == '__main__':
    tf.app.run()

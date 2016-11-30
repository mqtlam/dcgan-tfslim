import os
import numpy as np
from time import gmtime, strftime

from image_ops import save_images

def sample_images(dcgan, min_range=-1, max_range=1):
    """Sample images.

    Postconditions:
        saves to image file

    Args:
        dcgan: DCGAN
        min_range: minimum range (must be [-1, 1] and < max_range) [-1]
        max_range: maximum range (must be [-1, 1] and > min_range) [1]
    """
    FLAGS = dcgan.f
    sample_z = np.random.uniform(min_range, max_range,
                                 size=(FLAGS.sample_size, FLAGS.z_dim)).astype(np.float32)
    samples = dcgan.sess.run(dcgan.G, feed_dict={dcgan.z: sample_z})
    sample_path = os.path.join('./', FLAGS.sample_dir,
                               dcgan.get_model_dir(),
                               'test_{0}.png'.format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    save_images(samples, [8, 8], sample_path)

def visualize_z(dcgan, min_range=-1, max_range=1):
    """Visualize z dimension.
    For each dimension, vary it and fix the rest to 0.

    Postconditions:
        saves to image files

    Args:
        dcgan: DCGAN
        min_range: minimum range (must be [-1, 1] and < max_range) [-1]
        max_range: maximum range (must be [-1, 1] and > min_range) [1]
    """
    FLAGS = dcgan.f
    range_values = np.arange(min_range, max_range, 1./FLAGS.batch_size)
    for z in xrange(FLAGS.z_dim):
        sample_z = np.zeros([FLAGS.batch_size, FLAGS.z_dim])
        for i, z_vector in enumerate(sample_z):
            z_vector[z] = range_values[i]

        samples = dcgan.sess.run(dcgan.G, feed_dict={dcgan.z: sample_z})
        sample_path = os.path.join('./', FLAGS.sample_dir,
                                   dcgan.get_model_dir(),
                                   'visualize_z_{0}.png'.format(z))
        save_images(samples, [8, 8], sample_path)

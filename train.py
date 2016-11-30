import os
import time
import fnmatch
from random import shuffle
import numpy as np
import tensorflow as tf

from image_ops import get_image, save_images

def generate_z(sample_size, z_dim):
    """Helper function to generate noise vector.
    Can replace this with a different noise function.

    Args:
        sample_size: sample/batch size
        z_dim: dimensionality of z noise

    Returns:
        random noise, dimensionality is (sample_size, z_dim)
    """
    return np.random.uniform(-1, 1,
        size=(sample_size, z_dim)).astype(np.float32)

def train(dcgan):
    """Train DCGAN.

    Preconditions:
        checkpoint, data, logs directories exist

    Postconditions:
        checkpoints are saved
        logs are written

    Args:
        dcgan: DCGAN object
    """
    sess = dcgan.sess
    FLAGS = dcgan.f

    # load dataset
    list_file = os.path.join(FLAGS.data_dir, '{0}.txt'.format(FLAGS.dataset))
    if os.path.exists(list_file):
        # load from file when found
        print "Using training list: {0}".format(list_file)
        with open(list_file, 'r') as f:
            data = [os.path.join(FLAGS.data_dir,
                                 FLAGS.dataset, l.strip()) for l in f]
    else:
        # recursively walk dataset directory to get images
        data = []
        dataset_dir = os.path.join(FLAGS.data_dir, FLAGS.dataset)
        for root, dirnames, filenames in os.walk(dataset_dir):
            for filename in fnmatch.filter(filenames, '*.{0}'.format(FLAGS.image_ext)):
                data.append(os.path.join(root, filename))
        shuffle(data)

        # save to file for next time
        with open(list_file, 'w') as f:
            for l in data:
                line = l.replace(dataset_dir + os.sep, '')
                f.write('{0}\n'.format(line))

    assert len(data) > 0, "found 0 training data"
    print "Found {0} training images.".format(len(data))

    # set up Adam optimizers
    d_optim = tf.train.AdamOptimizer(
        FLAGS.learning_rate,
        beta1=FLAGS.beta1
        ).minimize(dcgan.d_loss, var_list=dcgan.d_vars)
    g_optim = tf.train.AdamOptimizer(
        FLAGS.learning_rate,
        beta1=FLAGS.beta1
        ).minimize(dcgan.g_loss, var_list=dcgan.g_vars)
    tf.initialize_all_variables().run()

    # summaries
    g_sum = tf.merge_summary([dcgan.z_sum, dcgan.d_fake_sum,
        dcgan.g_sum, dcgan.d_loss_fake_sum, dcgan.g_loss_sum])
    d_sum = tf.merge_summary([dcgan.z_sum, dcgan.d_real_sum,
        dcgan.real_sum, dcgan.d_loss_real_sum, dcgan.d_loss_sum])
    writer = tf.train.SummaryWriter(os.path.join(FLAGS.log_dir,
        dcgan.get_model_dir()), sess.graph)

    # training images for sampling
    sample_files = data[0:FLAGS.sample_size]
    sample = [get_image(sample_file,
                        FLAGS.output_size) for sample_file in sample_files]
    sample_images = np.array(sample).astype(np.float32)
    sample_path = os.path.join('./', FLAGS.sample_dir,
                               dcgan.get_model_dir(),
                               'real_samples.png')
    save_images(sample_images, sample_path)

    # z for sampling
    sample_z = generate_z(FLAGS.sample_size, FLAGS.z_dim)

    # run for number of epochs
    counter = 1
    start_time = time.time()
    for epoch in xrange(FLAGS.epoch):
        num_batches = int(len(data) / FLAGS.batch_size)
        # training iterations
        for batch_index in xrange(0, num_batches):
            # get batch of images for training
            batch_start = batch_index*FLAGS.batch_size
            batch_end = (batch_index+1)*FLAGS.batch_size
            batch_files = data[batch_start:batch_end]
            batch_images = [get_image(batch_file,
                               FLAGS.output_size) for batch_file in batch_files]

            # create batch of random z vectors for training
            batch_z = generate_z(FLAGS.batch_size, FLAGS.z_dim)

            # update D network
            _, summary_str = sess.run([d_optim, d_sum],
                feed_dict={dcgan.real_images: batch_images, dcgan.z: batch_z})
            writer.add_summary(summary_str, counter)

            # update G network
            _, summary_str = sess.run([g_optim, g_sum],
                feed_dict={dcgan.z: batch_z})
            writer.add_summary(summary_str, counter)

            # update G network again for stability
            _, summary_str = sess.run([g_optim, g_sum],
                feed_dict={dcgan.z: batch_z})
            writer.add_summary(summary_str, counter)

            # compute errors
            errD_fake = dcgan.d_loss_fake.eval({dcgan.z: batch_z})
            errD_real = dcgan.d_loss_real.eval({dcgan.real_images: batch_images})
            errG = dcgan.g_loss.eval({dcgan.z: batch_z})

            # increment global counter (for saving models)
            counter += 1

            # print stats
            print "[train] epoch: {0}, iter: {1}/{2}, time: {3}, d_loss: {4}, g_loss: {5}".format(
                epoch, batch_index, num_batches, time.time() - start_time, errD_fake+errD_real, errG)

            # sample every 100 iterations
            if np.mod(counter, 100) == 1:
                samples, d_loss, g_loss = dcgan.sess.run(
                    [dcgan.G, dcgan.d_loss, dcgan.g_loss],
                    feed_dict={dcgan.z: sample_z,
                               dcgan.real_images: sample_images})
                print "[sample] time: {0}, d_loss: {1}, g_loss: {2}".format(
                    time.time() - start_time, d_loss, g_loss)
                # save samples for visualization
                sample_path = os.path.join('./', FLAGS.sample_dir,
                                           dcgan.get_model_dir(),
                                           'train_{0:02d}_{1:04d}.png'.format(epoch, batch_index))
                save_images(samples, sample_path)

            # save model every 500 iterations
            if np.mod(counter, 500) == 2:
                dcgan.save(counter)
                print "[checkpoint] saved: {0}".format(time.time() - start_time)

    # save final model
    dcgan.save(counter)
    print "[checkpoint] saved: {0}".format(time.time() - start_time)


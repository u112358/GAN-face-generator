data_dir = './data'

# FloydHub - Use with data ID "R5KrjnANiKVhLWAkpXhNBe"
#data_dir = '/input'


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper


import os
from glob import glob
import tensorflow as tf
import numpy as np
import datetime as datetime


def model_inputs(image_width, image_height, image_channels, z_dim):
    """
    Create the model inputs
    :param image_width: The input image width
    :param image_height: The input image height
    :param image_channels: The number of image channels
    :param z_dim: The dimension of Z
    :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
    """
    inputs_real = tf.placeholder(tf.float32, shape=(None, image_width, image_height, image_channels), name='input_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    return inputs_real, inputs_z, learning_rate




def discriminator(images, reuse=False):
    """
    Create the discriminator network
    :param image: Tensor of input image(s)
    :param reuse: Boolean if the weights should be reused
    :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
    """
    alpha = 0.2

    with tf.variable_scope('discriminator', reuse=reuse):
        # using 4 layer network as in DCGAN Paper

        # Conv 1
        conv1 = tf.layers.conv2d(images, 32, 5, 2, 'SAME')
        lrelu1 = tf.maximum(alpha * conv1, conv1)

        # Conv 2
        conv2 = tf.layers.conv2d(lrelu1, 64, 5, 2, 'SAME')
        batch_norm2 = tf.layers.batch_normalization(conv2, training=True)
        lrelu2 = tf.maximum(alpha * batch_norm2, batch_norm2)

        # Conv 3
        conv3 = tf.layers.conv2d(lrelu2, 128, 5, 1, 'SAME')
        batch_norm3 = tf.layers.batch_normalization(conv3, training=True)
        lrelu3 = tf.maximum(alpha * batch_norm3, batch_norm3)

        # Conv 4
        conv4 = tf.layers.conv2d(lrelu3, 256, 5, 1, 'SAME')
        batch_norm4 = tf.layers.batch_normalization(conv4, training=True)
        lrelu4 = tf.maximum(alpha * batch_norm4, batch_norm4)

        conv5 = tf.layers.conv2d(lrelu4, 512, 5, 1, 'SAME')
        batch_norm5 = tf.layers.batch_normalization(conv5, training=True)
        lrelu5 = tf.maximum(alpha * batch_norm5, batch_norm5)
        # Flatten
        flat = tf.reshape(lrelu5, (-1, 8 * 8 * 512))

        # Logits
        logits = tf.layers.dense(flat, 1)

        # Output
        out = tf.sigmoid(logits)

        return out, logits




def generator(z, out_channel_dim, is_train=True):
    """
    Create the generator network
    :param z: Input z
    :param out_channel_dim: The number of channels in the output image
    :param is_train: Boolean if generator is being used for training
    :return: The tensor output of the generator
    """
    alpha = 0.2

    with tf.variable_scope('generator', reuse=False if is_train == True else True):
        # Fully connected
        fc1 = tf.layers.dense(z, 8 * 8 * 512)
        fc1 = tf.reshape(fc1, (-1, 8, 8, 512))
        fc1 = tf.maximum(alpha * fc1, fc1)

        # Starting Conv Transpose Stack
        deconv2 = tf.layers.conv2d_transpose(fc1, 256, 3, 1, 'SAME')
        batch_norm2 = tf.layers.batch_normalization(deconv2, training=is_train)
        lrelu2 = tf.maximum(alpha * batch_norm2, batch_norm2)

        deconv3 = tf.layers.conv2d_transpose(lrelu2, 128, 3, 1, 'SAME')
        batch_norm3 = tf.layers.batch_normalization(deconv3, training=is_train)
        lrelu3 = tf.maximum(alpha * batch_norm3, batch_norm3)

        deconv4 = tf.layers.conv2d_transpose(lrelu3, 64, 3, 2, 'SAME')
        batch_norm4 = tf.layers.batch_normalization(deconv4, training=is_train)
        lrelu4 = tf.maximum(alpha * batch_norm4, batch_norm4)

        deconv5 = tf.layers.conv2d_transpose(lrelu4,32,3,2,'SAME')
        batch_norm5 = tf.layers.batch_normalization(deconv5, training=is_train)
        lrelu5 = tf.maximum(alpha * batch_norm5, batch_norm5)

        # Logits
        logits = tf.layers.conv2d_transpose(lrelu5, out_channel_dim, 3, 2, 'SAME')

        # Output
        out = tf.tanh(logits)
        tf.summary.image('out',out,20)

        return out





def model_loss(input_real, input_z, out_channel_dim):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """

    g_model = generator(input_z, out_channel_dim)
    d_model_real, d_logits_real = discriminator(input_real)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                labels=tf.ones_like(d_model_real) * 0.9))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                labels=tf.zeros_like(d_model_fake)))
    d_loss = d_loss_real + d_loss_fake

    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                labels=tf.ones_like(d_model_fake)))
    tf.summary.scalar('d_loss',d_loss)
    tf.summary.scalar('g_loss',g_loss)
    return d_loss, g_loss





def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    # Optimize
    d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
    g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt




def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode):
    """
    Train the GAN
    :param epoch_count: Number of epochs
    :param batch_size: Batch Size
    :param z_dim: Z dimension
    :param learning_rate: Learning Rate
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :param get_batches: Function to get batches
    :param data_shape: Shape of the data
    :param data_image_mode: The image mode to use for images ("RGB" or "L")
    """

    gpu_config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=gpu_config)
    saver = tf.train.Saver(tf.trainable_variables())
    input_real, input_z, _ = model_inputs(data_shape[1], data_shape[2], data_shape[3], z_dim)
    d_loss, g_loss = model_loss(input_real, input_z, data_shape[3])
    d_opt, g_opt = model_opt(d_loss, g_loss, learning_rate, beta1)
    sum_op = tf.summary.merge_all()
    steps = 0

    sess.run(tf.global_variables_initializer())
    logdir=os.path.join('./log/',datetime.datetime.now().isoformat())
    sumWriter = tf.summary.FileWriter(logdir,sess.graph)
    for epoch_i in range(epoch_count):
        for batch_images in get_batches(batch_size):
            batch_images = batch_images * 2
            steps += 1

            batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))

            summ,gloss,dloss,_ = sess.run([sum_op,g_loss,d_loss,d_opt], feed_dict={input_real: batch_images, input_z: batch_z})
            _ = sess.run(g_opt, feed_dict={input_z: batch_z})
            print 'epoch[%d] step[%d] gloss[%lf] dloss[%lf]' %(epoch_i, steps,gloss,dloss)
            sumWriter.add_summary(summ,steps)
        saver.save(sess,'GAN',global_step=steps)
batch_size = 40
z_dim = 100
learning_rate = 0.0003
beta1 = 0.5

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
epochs = 10

celeba_dataset = helper.Dataset('celeba', glob(os.path.join(data_dir, 'img_align_celeba/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, celeba_dataset.get_batches,
          celeba_dataset.shape, celeba_dataset.image_mode)
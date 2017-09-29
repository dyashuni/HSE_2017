import numpy
import random
import tensorflow as tf

from utils import rel_error
from utils import count_params
from utils import get_session
from utils import show_images
from utils import load_dataset
from utils import deprocess_img
from utils import preprocess_img


def leaky_relu(x, alpha=0.01):
    """Compute the leaky ReLU activation function.

    Inputs:
    - x: TensorFlow Tensor with arbitrary shape
    - alpha: leak parameter for leaky ReLU

    Returns:
    TensorFlow Tensor with the same shape as x
    """
    # TODO: implement leaky ReLU
    with tf.variable_scope('leaky_relu'):
        pass


def sample_noise(batch_size, dim):
    """Generate random uniform noise from -1 to 1.

    Inputs:
    - batch_size: integer giving the batch size of noise to generate
    - dim: integer giving the dimension of the the noise to generate

    Returns:
    TensorFlow Tensor containing uniform noise in [-1, 1] with shape [batch_size, dim]
    """
    # TODO: sample and return noise
    with tf.variable_scope('sample_noise'):
        pass


def discriminator_fc(x):
    """Compute discriminator score for a batch of input images.

    Inputs:
    - x: TensorFlow Tensor of images, shape [batch_size, 32, 32, channels]

    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score
    for an image being real for each input image.
    """
    with tf.variable_scope("discriminator"):
        # TODO: implement architecture
        """
        Use tf.layers to build the model.
        All fully connected layers should include bias terms.
        Architecture:
            Flatten
            Fully connected layer, 256 outputs
            LeakyReLU with alpha 0.01
            Fully connected layer from 256 to 256
            LeakyReLU with alpha 0.01
            Fully connected layer from 256 to 1
        """
        pass
        return logits


def generator_fc(z, channels):
    """Generate images from a random noise vector.

    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]

    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 32, 32, channels].
    """

    with tf.variable_scope("generator"):
        # TODO: implement architecture
        """
        Use tf.layers to build the model.
        All fully connected layers should include bias terms.
        Architecture:
            Flatten
            Fully connected layer, 1024 outputs
            ReLU
            Fully connected layer from 1024 to 1024
            ReLU
            Fully connected layer from 1024 to 32*32*channels
            TanH (To restrict the output to be [-1,1])
            Reshape to size [batch_size, 32, 32, channels]
        """
        pass
        return img


def gan_loss(logits_real, logits_fake):
    """Compute the GAN loss.

    Inputs:
    - logits_real: Tensor, shape [batch_size, 1], output of discriminator
                   Log probability that the image is real for each real image
    - logits_fake: Tensor, shape[batch_size, 1], output of discriminator
                   Log probability that the image is real for each fake image

    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
    """
    with tf.variable_scope('gan_loss'):
        # TODO: compute D_loss and G_loss
        """
        Use tf.ones_like and tf.zeros_like to generate labels for your discriminator.
        Use tf.nn.sigmoid_cross_entropy_with_logits loss to compute your loss function.
        Instead of computing the expectation, we will be averaging over elements of the minibatch, so make sure to combine the loss by averaging instead of summing.
        """
        D_loss = None
        G_loss = None
        pass
        return D_loss, G_loss


def get_solvers(learning_rate=1e-3, beta1=0.5):
    """Create solvers for GAN training.

    Inputs:
    - learning_rate: learning rate to use for both solvers
    - beta1: beta1 parameter for both solvers (first moment decay)

    Returns:
    - D_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
    - G_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
    """
    with tf.variable_scope('get_solvers'):
        # TODO: create an AdamOptimizer for D_solver and G_solver
        D_solver = None
        G_solver = None
        pass
        return D_solver, G_solver


def discriminator_dc(x):
    """Compute discriminator score for a batch of input images.

    Inputs:
    - x: TensorFlow Tensor of images, shape [batch_size, 32, 32, channels]

    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score
    for an image being real for each input image.
    """
    with tf.variable_scope("discriminator"):
        # TODO: implement architecture
        """
        Use tf.layers.conv2d for convolutional layers
        Architecture:
            Convolutional layer, 32 Filters, 5x5, Stride 1, Leaky ReLU(alpha=0.01)
            Max Pool 2x2, Stride 2
            Convolutional layer, 64 Filters, 5x5, Stride 1, Leaky ReLU(alpha=0.01)
            Max Pool 2x2, Stride 2
            Flatten
            Fully Connected size 5 x 5 x 64, Leaky ReLU(alpha=0.01)
            Fully Connected size 1
        """
        pass
        return logits


def generator_dc(z, channels):
    """Generate images from a random noise vector.

    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]

    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 32, 32, channels].
    """
    with tf.variable_scope("generator"):
        # TODO: implement architecture
        """
        Use tf.nn.conv2d_transpose for transposed conv layers
        Architecture:
            Fully connected of size 4*4*512, ReLU
            BatchNorm
            Reshape into Image Tensor of size [batch_size, 4, 4, 512]
            256 filters, conv2d^T (transpose) filters of 4x4, padding SAME, stride 2, ReLU
            BatchNorm
            128 filters, conv2d^T (transpose) filters of 4x4, padding SAME, stride 2, ReLU
            BatchNorm
            'channels' filters,  conv2d^T (transpose) filter of 4x4, padding SAME, stride 2, TanH
        """
        pass
        return img


def train_gan(sess, G_train_step, G_loss, G_sample, D_train_step, D_loss, x, training_data,
              show_every=250, print_every=50, batch_size=128, num_epoch=10):
    """Train a GAN for a certain number of epochs.

    Inputs:
    - sess: A tf.Session that we want to use to run our data
    - G_train_step: A training step for the Generator
    - G_loss: Generator loss
    - G_sample: Generator image
    - D_train_step: A training step for the Generator
    - D_loss: Discriminator loss
    - x: placeholder for input data
    - training_data: Dataset, list of images
    Returns:
        Nothing
    """
    iter = 0
    num_samples = len(training_data)
    for i in range(num_epoch):
        random.shuffle(training_data)
        # create mini batches for each epoch
        mini_batches = [
            training_data[k:k + batch_size] for k in range(0, num_samples, batch_size)
        ]
        # iterate over mini batches
        for mini_batch in mini_batches:
            # last batch may be smaller than the previous ones, so skip it
            if len(mini_batch) < batch_size:
                continue
            # show a sample result
            if iter % show_every == 0:
                samples = sess.run(G_sample)
                samples_deproc = deprocess_img(samples[:25])
                show_images(samples_deproc)
                print()
            # run a batch of data through the network
            mini_batch_array = numpy.stack(mini_batch).astype(dtype=numpy.float32)
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: mini_batch_array})
            _, G_loss_curr = sess.run([G_train_step, G_loss])

            # We want to make sure D_loss doesn't go to 0
            if iter % print_every == 0:
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter, D_loss_curr, G_loss_curr))

            iter += 1
    samples = sess.run(G_sample)
    samples_deproc = deprocess_img(samples[:25])

    print('Final images')
    show_images(samples_deproc, is_wait=True)


def run_gan(dataset, discriminator, generator, num_epoch=10):
    """Helper function for training GANs"""
    tf.reset_default_graph()

    # number of images for each batch
    batch_size = 128
    # noise dimension
    noise_dim = 96

    # shape of train images
    img_shape = list(dataset[0].shape)
    height   = img_shape[0]
    width    = img_shape[1]
    channels = img_shape[2]

    # check image shape
    assert height == 32, 'Error: image height should be 32'
    assert width  == 32, 'Error: image width  should be 32'

    # placeholder for images from the training dataset
    placeholder_size = [None] + img_shape
    x = tf.placeholder(tf.float32, placeholder_size)
    # random noise fed into our generator
    z = sample_noise(batch_size, noise_dim)
    # generated images
    G_sample = generator(z, channels)

    with tf.variable_scope('') as scope:
        img_preproc = preprocess_img(x)
        logits_real = discriminator(img_preproc)
        # Re-use discriminator weights on new inputs
        scope.reuse_variables()
        logits_fake = discriminator(G_sample)

    # get solvers
    D_solver, G_solver = get_solvers()

    # get discriminator and generator loss
    D_loss, G_loss = gan_loss(logits_real, logits_fake)

    # Get the list of variables for the discriminator and generator
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

    # setup training steps
    D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
    G_train_step = G_solver.minimize(G_loss, var_list=G_vars)

    with get_session() as sess:
        sess.run(tf.global_variables_initializer())
        train_gan(sess, G_train_step, G_loss, G_sample, D_train_step, D_loss, x, dataset,
                  batch_size=batch_size, num_epoch=num_epoch)


# test functions
def test_leaky_relu(x, y_true):
    tf.reset_default_graph()
    with get_session() as sess:
        y_tf = leaky_relu(tf.constant(x))
        y = sess.run(y_tf)
        error = rel_error(y_true, y)
        print('Maximum error: %g'%error)
        if error < 1e-8:
            print('Leaky ReLu test passed!')
        else:
            print('Leaky ReLu test failed!')
            exit(1)


def test_sample_noise():
    batch_size = 3
    dim = 4
    tf.reset_default_graph()
    with get_session() as sess:
        z = sample_noise(batch_size, dim)
        # Check z has the correct shape
        assert z.get_shape().as_list() == [batch_size, dim]
        # Make sure z is a Tensor and not a numpy array
        assert isinstance(z, tf.Tensor)
        # Check that we get different noise for different evaluations
        z1 = sess.run(z)
        z2 = sess.run(z)
        assert not numpy.array_equal(z1, z2)
        # Check that we get the correct range
        assert numpy.all(z1 >= -1.0) and numpy.all(z1 <= 1.0)
        print("Sample Noise test passed!")


def test_discriminator(discriminator, true_count):
    tf.reset_default_graph()
    with get_session() as sess:
        y = discriminator(tf.ones((1, 32, 32, 3)))
        cur_count = count_params()
        if cur_count != true_count:
            print('Incorrect number of parameters in discriminator. {0} instead of {1}. Check your architecture.'.format(cur_count,true_count))
        else:
            print('Correct number of parameters in discriminator.')


def test_generator(generator, true_count):
    tf.reset_default_graph()
    with get_session() as sess:
        y = generator(tf.ones((1, 96)), 3)
        cur_count = count_params()
        if cur_count != true_count:
            print('Incorrect number of parameters in generator. {0} instead of {1}. Check your achitecture.'.format(cur_count, true_count))
        else:
            print('Correct number of parameters in generator.')


def test_gan_loss(logits_real, logits_fake, d_loss_true, g_loss_true):
    tf.reset_default_graph()
    with get_session() as sess:
        d_loss, g_loss = sess.run(gan_loss(tf.constant(logits_real), tf.constant(logits_fake)))

    d_loss_error = rel_error(d_loss_true, d_loss)
    g_loss_error = rel_error(g_loss_true, g_loss)
    print("Maximum error in d_loss: %g"%d_loss_error)
    print("Maximum error in g_loss: %g"%g_loss_error)
    if d_loss_error < 1e-8 and g_loss_error < 1e-8:
        print("GAN Loss test passed!")
    else:
        print("GAN Loss test failed!")


if __name__ == '__main__':
    answers = numpy.load('gan_check.npz')

    # test implementation of functions
    test_leaky_relu(answers['lrelu_x'], answers['lrelu_y'])
    test_sample_noise()
    test_discriminator(discriminator_fc, true_count=852737)
    test_generator(generator_fc, true_count=4297728)
    test_gan_loss(answers['logits_real'], answers['logits_fake'], answers['d_loss_true'], answers['g_loss_true'])

    ########
    # MNIST
    ########
    mnist_path        = 'D:/git/HSE_2017/Datasets/mnist/train/'  # path to MNIST train
    mnist_num_samples = 60000  # for testing you may want to use less samples
    ### load mnist dataset
    mnist = load_dataset(mnist_path, max_num_samples=mnist_num_samples, is_color=False)
    ### show a random batch
    show_images(random.sample(mnist, 25))
    ### run training of fully connected GAN on MNIST
    run_gan(mnist, discriminator_fc, generator_fc, num_epoch=10)

    # test implementation of functions
    test_discriminator(discriminator_dc, true_count=2616897)
    test_generator(generator_dc, true_count=3456899)

    ### run training of deep convolutional GAN on MNIST
    ### This may take a while
    run_gan(mnist, discriminator_dc, generator_dc, num_epoch=5)

    ##########
    # CIFAR10
    ##########
    cifar10_path        = 'D:/git/HSE_2017/Datasets/cifar10/train/'  # path to cifar10 train
    cifar10_num_samples = 50000  # for testing you may want to use less samples
    ### load cifar dataset
    cifar10 = load_dataset(cifar10_path, max_num_samples=cifar10_num_samples, is_color=True)
    ### show a random batch
    show_images(random.sample(cifar10, 25))
    ### run training of deep convolutional GAN on cifar10
    ### This may take a while
    run_gan(cifar10, discriminator_dc, generator_dc, num_epoch=10)

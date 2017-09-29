import os
import cv2
import numpy
import random

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from multiprocessing import Process

# List of image extensions
img_ext = ['.jpg', '.jpeg', '.png', '.bmp']

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'


def preprocess_img(x):
    """Scale images [0, 255] to [-1, 1]"""
    return x / 128.0 - 1.0


def deprocess_img(x):
    """Scale images [-1, 1] to [0, 1]"""
    return (x + 1.0) / 2.0


def rel_error(x,y):
    """Relative error between two numpy arrays"""
    return numpy.max(numpy.abs(x - y) / (numpy.maximum(1e-8, numpy.abs(x) + numpy.abs(y))))


def count_params():
    """Count the number of parameters in the current TensorFlow graph """
    param_count = numpy.sum([numpy.prod(x.get_shape().as_list()) for x in tf.global_variables()])
    return param_count


def show_images(images, is_wait=False):
    """Show images"""
    is_gray = False
    img_shape = images[0].shape
    if img_shape[2] == 1:
        is_gray = True

    sqrtn = int(numpy.ceil(numpy.sqrt(len(images))))

    plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if is_gray:
            # remove redundant dimension
            img = img[:, :, 0]
            plt.set_cmap('gray')
        else:
            # swap channels
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(img)

    plt.pause(0.5)
    if is_wait:
        print('Click on last figure to continue ...')
        plt.waitforbuttonpress(0)


def get_session():
    """Return default tf session"""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def list_files(dirs=list(), ext=list()):
    """Search files with provided extensions"""
    files = list()
    # check var types
    if isinstance(dirs, str):
        dirs = [dirs]
    if isinstance(ext, str):
        ext = [ext]

    for dir in dirs:
        for root, dir_names, file_names in os.walk(dir):
            for filename in file_names:
                files.append(root + '/' + filename)
    if len(ext) > 0:
        ext_lowercase = [extI.lower() for extI in ext]
        files_ext = [f for f in files if f.lower().endswith(tuple(ext_lowercase))]
        return files_ext
    else:
        return files


def load_dataset(dataset_path, target_shape=(32, 32), is_color=True, max_num_samples=None):
    """Load images and resize them to specified shape"""
    print('Load dataset {0}'.format(dataset_path))
    img_files = list_files(dataset_path, ext=img_ext)
    random.shuffle(img_files)

    if max_num_samples is not None:
        img_files = img_files[:max_num_samples]

    imgs = []
    for img_file in img_files:
        if is_color:
            # read image as color
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            img_resized = cv2.resize(img, target_shape, interpolation=cv2.INTER_AREA)
        else:
            # read image as gray
            img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, target_shape, interpolation=cv2.INTER_AREA)
            # add channel axis to image
            img_resized = img_resized[..., numpy.newaxis]
        imgs.append(img_resized)

    return imgs

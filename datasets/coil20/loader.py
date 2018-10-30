import os
import glob
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import sys

NUM_CLASS = 20
NUM_DATA_PER_CLASS = 72


def make_probability(dim=3, var=0.1):
    """ Gaussian Mixture Model, Normal Distribution

    :param dim:
    :param var:
    :return:
    """
    assert dim >= 1, "lack of dimension"
    pi = np.random.binomial(1, 0.5)
    noise_0 = pi * np.random.normal(-1, var) + (1.0 - pi) * np.random.normal(+1, var)
    if dim == 1:
        return np.array([noise_0])

    noise = np.random.normal(0, 1, dim-1)
    noise = np.concatenate(([noise_0], noise), axis=0)
    return noise


def get_image(
        cls,
        index,
        path='coil-20-proc',
        img_size=128):

    filename = 'obj{}__{}.png'.format(cls, index)
    img = Image.open(os.path.join(path, filename))
    x = np.asarray(img)

    if x.shape[0] != img_size:
        img = img.resize((img_size, img_size))
        x = np.asarray(img)

    return x.reshape((1, img_size, img_size))


def get_train_index():
    """
    This method is used for creating training iterator
    (NUM_CLASS - 1) means that 20th class is used for confirming generalization and cannot use training

    :return:
    """
    data_size = (NUM_CLASS - 1) * NUM_DATA_PER_CLASS
    return np.array([i for i in range(0, data_size)])


def get_train_index_rand(size=4):
    data_size = (NUM_CLASS - 1) * NUM_DATA_PER_CLASS
    return np.random.randint(0, data_size, size)


def get_ref_and_real_data(indexes, size_per_class=72,
                          classes=20,
                          dim_noise=3,
                          img_size=128,
                          path='coil-20-proc',
                          dif_index=1):
    """

    [1] https://github.com/yvan/cLPR
    :return: data
    """

    refs = []
    reals = []
    noises = []

    for index in indexes:

        cls = math.floor(index / size_per_class) + 1 # no exist class 0
        num = index % size_per_class

        refs.append(get_image(cls=cls, index=num, path=path, img_size=img_size))

        noise = make_probability(dim=dim_noise, var=0.1)
        dif = dif_index if noise[0] >= 0 else -dif_index
        next_index = (num + dif + size_per_class) % size_per_class

        reals.append(get_image(cls=cls, index=next_index, path=path, img_size=img_size))
        noises.append(noise)

    return np.array(refs), np.array(reals), np.array(noises)


if __name__ == '__main__':
    path = sys.argv[1]
    for i in range(72*20):
        indexes = [i]
        x_ref, x_real, noise = get_ref_and_real_data(indexes=indexes, size_per_class=72, classes=20, dim_noise=3, img_size=128, path=path, dif_index=1)

        # 40, 1, 32, 32
        x = np.concatenate((x_ref, x_real), axis=0)
        x = np.asarray(x, dtype=np.uint8)
        _, _, H, W = x.shape


        x = np.reshape(x, (2, 1, H, W))
        # col, row, ch, H, W -> col, H, row, W, ch
        x = x.transpose((0, 2, 1, 3))
        x = x.reshape((2 * H, 1 * W))

        img = Image.fromarray(x)
        plt.imshow(img)
        plt.pause(.01)



import os
import glob
from PIL import Image
import numpy as np


def load(data_size=9260,
        path='cLPR/imgs_jpg'):
    """

    [1] https://github.com/yvan/cLPR
    :return: data
    """

    images = []

    for i in range(data_size):
        filename = 'cube1_index_{}.jpg'.format(i)
        filename = os.path.join(path, filename)
        img = np.asarray(Image.open(filename))
        images.append(img)

    return np.array(images)


def get_ref_and_real_data(indexes, size=9261,
        path='cLPR/imgs_jpg'):
    """

    [1] https://github.com/yvan/cLPR
    :return: data
    """

    refs = []
    reals = []
    noises = []
    for index in indexes:
        noise = np.random.normal(size=3)
        noises.append(noise)

        filename = 'cube1_index_{}.jpg'.format(index)
        filename = os.path.join(path, filename)
        x_ref = np.asarray(Image.open(filename)).transpose(2, 0, 1)
        refs.append(x_ref)

        real_noise = np.random.choice([-1, 1])  # L data is generated. Each element in the data is +1 or -1.
        j = (index + real_noise + size) % size
        filename = 'cube1_index_{}.jpg'.format(j)
        filename = os.path.join(path, filename)
        x_real = np.asarray(Image.open(filename)).transpose(2,0,1)
        reals.append(x_real)

    return np.array(refs), np.array(reals), np.array(noises)


if __name__ == '__main__':
    x_ref, x_real, noise = get_ref_and_real_data(1)
    print(x_ref.shape) # (3, 256, 256)
    print(x_real.shape) # (3, 256, 256)
    print(noise) # [x, y, z]

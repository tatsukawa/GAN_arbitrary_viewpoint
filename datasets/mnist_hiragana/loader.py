import os
import re
import glob
import numpy as np
from PIL import Image


def read(size=5):
    filenames = [filename for filename in glob.glob('./datasets/mnist_hiragana/hiragana_images/*.jpg')]
    filenames = np.random.choice(filenames, size)
    data = []
    print(filenames)
    for filename in filenames:
        img = Image.open(filename)
        img = img.resize(size=(28, 28))
        img = np.asarray(img)
        print(img.shape)
        data.append(np.asarray(img))

    data = np.array(data)
    data = data.reshape((size, 1, 28, 28))
    return data


if __name__ == '__main__':
    a = read(5)
    print(a)

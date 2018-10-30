import os
import glob
import numpy as np
from PIL import Image


def read(path, n=5):
    filenames = [filename for filename in glob.glob(os.path.join(path,'hiragana_images/*.jpg'))]
    filenames = np.random.choice(filenames, n)
    data = []
    print(filenames)
    for filename in filenames:
        img = Image.open(filename)
        img = img.resize(size=(28, 28))
        img = np.asarray(img)
        print(img.shape)
        data.append(np.asarray(img))

    data = np.array(data)
    data = data.reshape((n, 1, 28, 28))
    return data


if __name__ == '__main__':
    path = './mnist_hiragana'
    a = read(path, 5)
    print(a)

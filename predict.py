# -*- coding: utf-8 -*-
# !/usr/bin/env python

from __future__ import print_function
import argparse

import os
import chainer
import matplotlib
matplotlib.use('Agg')
from chainer import Variable
from chainer import serializers
from PIL import Image
import numpy as np
import yaml

from utils.config import Config

class AttributeDict(object):
    def __init__(self, obj):
        self.obj = obj

    def __getstate__(self):
        return self.obj.items()

    def __setstate__(self, items):
        if not hasattr(self, 'obj'):
            self.obj = {}
        for key, val in items:
            self.obj[key] = val

    def __getattr__(self, name):
        if name in self.obj:
            return self.obj.get(name)
        else:
            return None

    def fields(self):
        return self.obj

    def keys(self):
        return self.obj.keys()


def main():
    parser = argparse.ArgumentParser(description='3d pose generator')
    parser.add_argument('--config_path', '-cp', type=str, default='')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gen_snapshot', '-gs', type=str, default="", help='To restore snapshot')
    parser.add_argument('--dis_snapshot', '-ds', type=str, default="", help='To restore snapshot')
    parser.add_argument('--visualize_size', '-vs', type=int, default=10, help='Interval of visualizing')

    args = parser.parse_args()

    with open(os.path.join(args.config_path, 'config.yml'), 'r+') as f:
        config = AttributeDict(yaml.load(f))

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(config.batch_size))
    print('# iteration: {}'.format(config.iteration))
    print('')

    if config.dataset == 'mnist':
        from models.net_MNIST_no_encode import Generator
        from models.net_MNIST import Discriminator
        from datasets.mnist.loader import get_ref_and_real_data
        train, _ = chainer.datasets.get_mnist(withlabel=False, ndim=3, scale=255.)  # ndim=3 : (ch,width,height)
        ref_images = train[:args.visualize_size]
        data = get_ref_and_real_data(ref_images, 0.4, 20)
        init_ch = 1
    elif config.dataset == 'coil20':
        from dis_models.discriminator import Discriminator
        from gen_models.generator import Generator
        from datasets.coil20.loader import get_ref_and_real_data, get_train_index, get_train_index_rand
        train = get_train_index()
        indexes = get_train_index_rand(size=args.visualize_size)
        ref_index = train[indexes]
        data = get_ref_and_real_data(ref_index, img_size=config.img_size, dim_noise=config.dim_noise, dif_index=config.dif_index)
        init_ch = 1
    elif args.dataset == 'coil100':
        from dis_models.discriminator import Discriminator
        from gen_models.generator import Generator
        from datasets.coil100.loader import get_ref_and_real_data, get_train_index, get_train_index_rand
        train = get_train_index()
        indexes = get_train_index_rand(size=args.visualize_size)
        ref_index = train[indexes]
        data = get_ref_and_real_data(ref_index, img_size=config.img_size, dim_noise=config.dim_noise, dif_index=config.dif_index)
        init_ch = 3

    gen = Generator(init_ch=init_ch, dim_z=config.dim_noise, bottom_size=config.img_size)
    dis = Discriminator(init_ch=init_ch, dim_z=config.dim_noise, bottom_size=config.img_size)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()

    serializers.load_npz(args.dis_snapshot, dis)
    serializers.load_npz(args.gen_snapshot, gen)


    z = [1.0]
    for i in range(config.dim_noise - 1):
        z.append(0.0)
    x_ref, x_rot, eps = data

    converter = chainer.dataset.concat_examples
    device = args.gpu

    x_real = Variable(converter(x_rot, device))
    x_ref = Variable(converter(x_ref, device))

    x_real = Variable(x_real.data.astype(np.float32)) / 255.0
    x_ref = Variable(x_ref.data.astype(np.float32)) / 255.0

    eps = np.array([z for i in range(config.visualize_size)])
    eps = Variable(converter(eps, device))
    eps = Variable(eps.data.astype(np.float32))

    batch = x_ref.shape[0]
    width = x_ref.shape[-1]
    height = x_ref.shape[-2]
    channel = x_ref.shape[-3]
    # (10, 1, 28, 28)

    # TODO: fix hard coding
    image_size = batch

    for i in range(100):
        with chainer.using_config('train', False):
            x = gen(x_ref, eps)
            gen.cleargrads()

        x_ref = x

        x_gen = chainer.cuda.to_cpu(x.data)
        x_gen = x_gen.reshape((5, int(args.visualize_size / 5), channel, height, width))

        x_gen = x_gen*255
        x_gen = x_gen.clip(0.0, 255.0)
        # gen_output_activation_func is sigmoid
        x_gen = np.asarray(x_gen, dtype=np.uint8)
        # gen output_activation_func is tanh
        # x = np.asarray(np.clip((x+1) * 0.5 * 255, 0.0, 255.0), dtype=np.uint8)
        _, _, _, H, W = x_gen.shape
        #x = x.reshape((n_images, 3, 1, H, W))
        # col, row, ch, H, W -> col, H, row, W, ch
        x_gen = x_gen.transpose(0, 3, 1, 4, 2)
        x_gen = x_gen.reshape((5 * H, int(args.visualize_size / 5) * W, 3))
        preview_dir = '{}/preview'.format(args.config_path)
        preview_path = preview_dir + \
                       '/image{:0>6}.png'.format(i)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x_gen).save(preview_path)


if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-
# !/usr/bin/env python

import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable
import chainer.functions as F
from datasets.mnist.loader import get_ref_and_real_data


def out_generated_image(gen, data, dst, device):
    @chainer.training.make_extension()
    def make_image(trainer):
        x_ref, x_rot, eps = data
        batch = x_ref.shape[0]
        width = x_ref.shape[-1]
        height = x_ref.shape[-2]
        channel = x_ref.shape[-3]
        xp = gen.xp

        image_size = batch

        converter = chainer.dataset.concat_examples
        x_real = Variable(converter(x_rot, device))
        x_ref = Variable(converter(x_ref, device))
        eps = Variable(converter(eps, device))

        x_real = Variable(x_real.data.astype(np.float32)) / 255.0
        x_ref = Variable(x_ref.data.astype(np.float32)) / 255.0
        eps = Variable(eps.data.astype(np.float32))

        with chainer.using_config('train', False):
            x = gen(x_ref, eps)

        x_ref = chainer.cuda.to_cpu(x_ref.data)
        x_real = chainer.cuda.to_cpu(x_real.data)
        x_gen = chainer.cuda.to_cpu(x.data)

        x_ref = x_ref.reshape((1, image_size, channel, height, width))
        x_real = x_real.reshape((1, image_size, channel, height, width))
        x_gen = x_gen.reshape((1, image_size, channel, height, width))

        x = np.concatenate((x_ref, x_real, x_gen), axis=0)
        x = x*255
        x = x.clip(0.0, 255.0)
        # gen_output_activation_func is sigmoid
        x = np.asarray(x, dtype=np.uint8)
        # gen output_activation_func is tanh
        # x = np.asarray(np.clip((x+1) * 0.5 * 255, 0.0, 255.0), dtype=np.uint8)
        _, _, _, H, W = x.shape
        #x = x.reshape((n_images, 3, 1, H, W))
        # col, row, ch, H, W -> col, H, row, W, ch
        x = x.transpose(0, 3, 1, 4, 2)
        if channel == 3:
            x = x.reshape((3 * H, image_size * W, 3))
        elif channel == 1:
            x = x.reshape((3 * H, image_size * W))

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir + \
                       '/image{:0>6}.png'.format(trainer.updater.iteration)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x).save(preview_path)

#        dif = np.concatenate((y_1, y_2), axis=0)
#        preview_dir = '{}/preview'.format(dst)
#        preview_path = preview_dir + '/dif_image{:0>6}.png'.format(trainer.updater.iteration)
#        Image.fromarray(dif).save(preview_path)

    return make_image

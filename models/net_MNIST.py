"""
This code is from
"""

# -*- coding: utf-8 -*-
# !/usr/bin/env python

from __future__ import print_function

#import cupy
import numpy

import chainer
from chainer import initializers
import chainer.functions as F
import chainer.links as L


def to_onehot(label, class_num):
    return numpy.eye(class_num)[label]


class Generator(chainer.Chain):
    def __init__(self, init_ch=1, dim_z=1, ch=64, bottom_size=32):
        initializer = initializers.HeNormal()
        self.bottom_size = bottom_size
        self.ch = ch
        super(Generator, self).__init__(
            c1=L.Convolution2D(init_ch+dim_z, ch, ksize=4, stride=2, pad=1),
            c2=L.Convolution2D(ch, 2*ch, ksize=4, stride=2, pad=1),
            l0z=L.Linear(2*ch*(bottom_size//4)*(bottom_size//4), 2*ch*(bottom_size//4)*(bottom_size//4), initialW=initializer),
            dc1=L.Deconvolution2D(2*ch, ch, 4, stride=2, pad=1, initialW=initializer),
            dc2=L.Deconvolution2D(ch, 1, 4, stride=2, pad=1, initialW=initializer),
            bn0=L.BatchNormalization(ch),
            bn1=L.BatchNormalization(ch*2),
            bn2=L.BatchNormalization(ch*2*(bottom_size//4*(bottom_size//4))),
            bn3=L.BatchNormalization(ch),
        )


    def __call__(self, x, z):
        H, W = x.shape[2], x.shape[3]
        z = F.broadcast_to(
            F.reshape(z, (z.shape[0], z.shape[1], 1, 1)),
            (z.shape[0], z.shape[1], H, W)
        )
        h = F.concat((x, z), axis=1)
        h = F.relu(self.bn0(self.c1(h)))
        h = F.relu(self.bn1(self.c2(h)))

        h = F.relu(self.bn2(self.l0z(h)))
        h = F.reshape(h, (-1, 2*self.ch, (H//4), (W//4)))
        h = F.relu(self.bn3(self.dc1(h)))
        x_rec = F.sigmoid((self.dc2(h)))
        return x_rec

class Discriminator(chainer.Chain):
    def __init__(self, init_ch=1, dim_z=4, ch=6, bottom_size=32):
        initializer = initializers.HeNormal()
        super(Discriminator, self).__init__(
            c0=L.Convolution2D(init_ch+dim_z, ch, ksize=4, stride=2, pad=1, initialW=initializer),
            c1=L.Convolution2D(ch, 2*ch, ksize=4, stride=2, pad=1, initialW=initializer),
            l2=L.Linear(2*ch * (bottom_size // 4) * (bottom_size // 4), 1, initialW=initializer),
            bn0=L.BatchNormalization(64),
            bn1=L.BatchNormalization(128),
        )

    def __call__(self, x, z):
        # x's shape is the same z's shape.
        h = F.concat((x, z), axis=1)
        h = F.leaky_relu(self.bn0(self.c0(h)))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(h)
        l = self.l2(h)
        return l
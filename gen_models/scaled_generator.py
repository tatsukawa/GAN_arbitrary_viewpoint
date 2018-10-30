import math
import chainer
import chainer.links as L
from chainer import functions as F

from gen_models.resblocks import Block


class Generator(chainer.Chain):

    def __init__(self, init_ch=6, ch=8, activation=F.relu, distribution="normal", batch_size=64, dim_noise=3):
        super(Generator, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.distribution = distribution
        self.batch_size = batch_size
        self.dim_noise = dim_noise
        with self.init_scope():
            # Encoder
            self.block1 = Block(init_ch+dim_noise, ch, activation=activation, upsample=True)
            self.block2 = Block(ch, ch*4, activation=activation, upsample=True)
            self.block3 = Block(ch*4, ch*8, activation=activation, upsample=True)
            self.conv4 = L.Convolution2D(ch*8, ch*4, ksize=3, pad=1, stride=2, initialW=initializer)
            self.conv5 = L.Convolution2D(ch*4, ch*2, ksize=3, pad=1, stride=2, initialW=initializer)
            self.conv6 = L.Convolution2D(ch*2, ch, ksize=3, pad=1, stride=2, initialW=initializer)
            self.conv7 = L.Convolution2D(ch, 1, ksize=1, pad=0, stride=1, initialW=initializer)
            self.b4 = L.BatchNormalization(ch*8)
            self.b5 = L.BatchNormalization(ch*4)
            self.b6 = L.BatchNormalization(ch*2)
            self.b7 = L.BatchNormalization(ch)


    def __call__(self, x, z, **kwargs):
        h = x
        _b, _c, _h, _w = h.shape

        _z = F.broadcast_to(
            F.reshape(z, (z.shape[0], z.shape[1], 1, 1)),
            (z.shape[0], z.shape[1], _h, _w)
        )
        h = F.concat((h, _z), axis=1)

        h = self.block1(h, **kwargs)
        h = self.block2(h, **kwargs)
        h = self.block3(h, **kwargs)
        h = self.b4(h)
        h = self.conv4(h)
        h = self.activation(h)
        h = self.b5(h)
        h = self.conv5(h)
        h = self.activation(h)
        h = self.b6(h)
        h = self.conv6(h)
        h = self.activation(h)
        h = self.b7(h)
        h = self.conv7(h)
        h = F.tanh(h)
        return h
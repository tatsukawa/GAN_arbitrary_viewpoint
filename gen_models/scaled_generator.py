import math
import chainer
import chainer.links as L
from chainer import functions as F


#class Block(chainer.Chain):
#    def __init__(self,
#                 in_channels,
#                 out_channels,
#                 hidden_channels=None,
#                 ksize=3,
#                 pad=1,
#                 activation=F.relu,
#                 batch_size=64,
#                 is_shortcut=True,
#                 dim_noise=3
#                 ):
#        super(Block, self).__init__()
#        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
#        initializer_sc = chainer.initializers.GlorotUniform()
#        self.activation = activation
#        self.batch_size = batch_size
#        hidden_channels = out_channels if hidden_channels is None else hidden_channels
#        self.is_shortcut = is_shortcut
#        self.z_dim = dim_noise
#        with self.init_scope():
#            # But now, this is hard-coding and I have to improve these codes
#            # in_channels+1 means that concatenating the noise term to input shape
#            self.c1 = L.Convolution2D(in_channels+self.z_dim, hidden_channels, ksize=ksize, pad=pad, initialW=initializer)
#            self.c2 = L.Convolution2D(hidden_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
#            # in_channels+1 means the same reason above
#            self.b1 = L.BatchNormalization(in_channels+self.z_dim)
#            self.b2 = L.BatchNormalization(hidden_channels)
#
#            if is_shortcut:
#                self.c_sc = L.Convolution2D(in_channels+self.z_dim, out_channels, ksize=1, pad=0, initialW=initializer_sc)
#
#    def residual(self, x, z, **kwargs):
#        h = x
#        # zは(batch_size, dim)になっている
#        # xは(batch_size, channel, height, width)になっている
#        _b, _c, _h, _w = h.shape
#
#        # これが良いのかという問題がある. とりあえず試してみるだけ
#        _z = F.broadcast_to(z, (_h*_w, z.shape[0], z.shape[1]))
#        _z = F.transpose(_z, (1, 2, 0))
#        _z = F.reshape(_z, (_b, z.shape[1], _h, _w))
#
#        h = F.concat((h, _z), axis=1)
#        h = self.b1(h, **kwargs)
#
#        h = self.activation(h)
#        h = self.c1(h)
#        h = self.b2(h, **kwargs)
#        h = self.activation(h)
#        h = self.c2(h)
#        return h
#
#    def shortcut(self, x, z):
#        h = x
#        _b, _c, _h, _w = h.shape
#
#        # これが良いのかという問題がある. とりあえず試してみるだけ
#        _z = F.broadcast_to(z, (_h*_w, z.shape[0], z.shape[1]))
#        _z = F.transpose(_z, (1, 2, 0))
#        _z = F.reshape(_z, (_b, z.shape[1], _h, _w))

#        h = F.concat((h, _z), axis=1)
#
#        return self.c_sc(h)

#    def __call__(self, x, z, **kwargs):
#        if self.is_shortcut:
#            return self.residual(x, z, **kwargs) + self.shortcut(x, z)
#        else:
#            return self.residual(x, z, **kwargs)
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
            # TODO: :thinking_face:
            self.block1 = Block(init_ch+dim_noise, ch, activation=activation, upsample=True) # 32x32
            self.block2 = Block(     ch,  ch * 4, activation=activation, upsample=True) # 64x64
            self.block3 = Block( ch * 4,  ch * 8, activation=activation, upsample=True) # 128x1
            self.conv4 = L.Convolution2D(ch * 8, ch * 4, ksize=3, pad=1, stride=2, initialW=initializer)
            self.conv5 = L.Convolution2D(ch * 4, ch * 2, ksize=3, pad=1, stride=2, initialW=initializer)
            self.conv6 = L.Convolution2D(ch * 2,     ch, ksize=3, pad=1, stride=2, initialW=initializer)
            self.conv7 = L.Convolution2D(    ch,      1, ksize=1, pad=0, stride=1, initialW=initializer)
#            self.block5 = Block( ch * 8,  ch * 4, activation=activation)
#            self.block4 = Block( ch * 2,  ch * 1, activation=activation)
            self.b4 = L.BatchNormalization(ch*8)
            self.b5 = L.BatchNormalization(ch*4)
            self.b6 = L.BatchNormalization(ch*2)
            self.b7 = L.BatchNormalization(ch)


    def __call__(self, x, z, **kwargs):
        h = x
        _b, _c, _h, _w = h.shape
        # これが良いのかという問題がある. とりあえず試してみるだけ
        _z = F.broadcast_to(z, (_h*_w, z.shape[0], z.shape[1]))
        _z = F.transpose(_z, (1, 2, 0))
        _z = F.reshape(_z, (_b, z.shape[1], _h, _w))

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
import math
import chainer
import chainer.links as L
from chainer import functions as F


class Block(chainer.Chain):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 ksize=3,
                 pad=1,
                 activation=F.relu,
                 batch_size=64,
                 is_shortcut=False,
                 dim_z=3
                 ):
        super(Block, self).__init__()
        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.batch_size = batch_size
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.is_shortcut = is_shortcut
        self.z_dim = dim_z
        with self.init_scope():
            # But now, this is hard-coding and I have to improve these codes
            # in_channels+1 means that concatenating the noise term to input shape
            self.c1 = L.Convolution2D(in_channels+self.z_dim, hidden_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = L.Convolution2D(hidden_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            # in_channels+1 means the same reason above
            self.b1 = L.BatchNormalization(in_channels+self.z_dim)
            self.b2 = L.BatchNormalization(hidden_channels)

            if is_shortcut:
                self.c_sc = L.Convolution2D(in_channels+self.z_dim, out_channels, ksize=1, pad=0, initialW=initializer_sc)

    def residual(self, x, z, **kwargs):
        h = x
        # zは(batch_size, dim)になっている
        # xは(batch_size, channel, height, width)になっている
        _b, _c, _h, _w = h.shape

        # これが良いのかという問題がある. とりあえず試してみるだけ
        _z = F.broadcast_to(
            F.reshape(z, (z.shape[0], z.shape[1], 1, 1)),
            (z.shape[0], z.shape[1], _h, _w)
        )
        h = F.concat((h, _z), axis=1)
        h = self.b1(h, **kwargs)

        h = self.activation(h)
        h = self.c1(h)
        h = self.b2(h, **kwargs)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x, z):
        h = x
        _b, _c, _h, _w = h.shape

        _z = F.broadcast_to(
            F.reshape(z, (z.shape[0], z.shape[1], 1, 1)),
            (z.shape[0], z.shape[1], _h, _w)
        )
        h = F.concat((h, _z), axis=1)

        return self.c_sc(h)

    def __call__(self, x, z, **kwargs):
        if self.is_shortcut:
            return self.residual(x, z, **kwargs) + self.shortcut(x, z)
        else:
            return self.residual(x, z, **kwargs)


class Generator(chainer.Chain):

    def __init__(self, init_ch=6, ch=16, activation=F.relu, distribution="normal", batch_size=64, dim_z=3, bottom_size=32):
        super(Generator, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.distribution = distribution
        self.batch_size = batch_size
        self.dim_z = dim_z
        with self.init_scope():
            # Encoder
            # TODO: :thinking_face:
            self.block1 = Block(init_ch,      ch, activation=activation, batch_size=batch_size, is_shortcut=True , dim_z=dim_z)
            self.block2 = Block(     ch,  ch * 2, activation=activation, batch_size=batch_size, is_shortcut=True , dim_z=dim_z)
            self.block3 = Block( ch * 2,  ch * 2, activation=activation, batch_size=batch_size, is_shortcut=True , dim_z=dim_z)
            self.block4 = Block( ch * 4,  ch * 4, activation=activation, batch_size=batch_size, is_shortcut=True , dim_z=dim_z)
            self.linear = L.Linear(ch * 4 * (bottom_size * bottom_size), ch * 4 * (bottom_size * bottom_size))
            self.b4 = L.BatchNormalization(ch * 4 * (bottom_size * bottom_size))
            self.block5 = Block( ch * 4,  ch * 2, activation=activation, batch_size=batch_size, is_shortcut=False, dim_z=dim_z)
            self.block6 = Block( ch * 2,  ch * 2, activation=activation, batch_size=batch_size, is_shortcut=False, dim_z=dim_z)
            self.block7 = Block( ch * 2,  ch * 1, activation=activation, batch_size=batch_size, is_shortcut=False, dim_z=dim_z)
            self.b8 = L.BatchNormalization(ch)
            self.l8 = L.Convolution2D(ch, 1, ksize=3, stride=1, pad=1, initialW=initializer)


    def __call__(self, x, z, **kwargs):
        h = x
        h = self.block1(h, z, **kwargs)
        h = self.block2(h, z, **kwargs)
        h = self.block3(h, z, **kwargs)
        h = self.block4(h, z, **kwargs)
        h = self.activation(self.linear(self.linear(h)))
        h = self.block5(h, z, **kwargs)
        h = self.block6(h, z, **kwargs)
        h = self.block7(h, z, **kwargs)
        h = self.b8(h)
        h = self.activation(h)
        h = F.tanh(self.l8(h))
        return h

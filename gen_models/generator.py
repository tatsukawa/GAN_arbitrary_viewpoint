import math
import chainer
import chainer.links as L
from chainer import functions as F
from links.svd_linear import SVDLinear


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

        _b, _c, _h, _w = h.shape
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

    def __init__(self, init_ch=6, ch=8, out_ch=3, activation=F.relu, distribution="normal", batch_size=64, dim_z=3, bottom_size=32):
        super(Generator, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        #initializer_u = chainer.initializers.Uniform(scale=1)
        #initializer_v = chainer.initializers.Uniform(scale=1)
        self.activation = activation
        self.distribution = distribution
        self.batch_size = batch_size
        self.dim_z = dim_z
        self.ch = ch
        with self.init_scope():
            # Encoder
            self.enc1 = Block(init_ch,      ch, activation=activation, batch_size=batch_size, is_shortcut=True , dim_z=dim_z)
            self.enc2 = Block(     ch,  ch*2, activation=activation, batch_size=batch_size, is_shortcut=True , dim_z=dim_z)
            self.enc3 = Block( ch*2,  ch*2, activation=activation, batch_size=batch_size, is_shortcut=True , dim_z=dim_z)
            self.linear = L.Linear(ch * 2 * (bottom_size * bottom_size), ch * 2 * (bottom_size * bottom_size))
            # WIP: I have not finished implemented this.
            # This code means reduction of dimension.
            # self.linear = SVDLinear(ch * 4 * (bottom_size * bottom_size), (ch * 4 * (bottom_size * bottom_size)), k=(bottom_size * bottom_size * ch * 4), initialU=initializer_u, initialV=initializer_v)
            self.b4 = L.BatchNormalization(ch * 2 * (bottom_size * bottom_size))
            self.dec1 = Block(ch * 2, ch * 2, activation=activation, batch_size=batch_size, is_shortcut=False, dim_z=dim_z)
            self.dec2 = Block(ch * 2, ch, activation=activation, batch_size=batch_size, is_shortcut=False, dim_z=dim_z)
            self.b8 = L.BatchNormalization(ch)
            self.l8 = L.Convolution2D(ch, out_ch, ksize=3, stride=1, pad=1, initialW=initializer)


    def __call__(self, x, z, **kwargs):
        h = x
        H, W = h.shape[2], h.shape[3]
        h = self.enc1(h, z, **kwargs)
        h = self.enc2(h, z, **kwargs)
        h = self.enc3(h, z, **kwargs)
        h = F.reshape(h, (-1, H*W*2*self.ch))
        h = self.activation(self.b4(self.linear(h)))
        h = F.reshape(h, (-1, 2*self.ch, H, W))
        h = self.dec1(h, z, **kwargs)
        h = self.dec2(h, z, **kwargs)
        h = self.b8(h)
        h = self.activation(h)
        h = F.tanh(self.l8(h))
        return h

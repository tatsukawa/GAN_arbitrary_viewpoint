import chainer
from chainer import functions as F
import chainer.links as L
from links.sn_embed_id import SNEmbedID
from links.sn_linear import SNLinear
from dis_models.resblocks import Block, OptimizedBlock
import math


class Block1(chainer.Chain):
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
        super(Block1, self).__init__()
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
            return self.residual(x, **kwargs)


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return F.average_pooling_2d(x, 2)


class Discriminator(chainer.Chain):
    def __init__(self, init_ch=1, ch=6, activation=F.relu, dim_z=4, bottom_size=32):
        super(Discriminator, self).__init__()
        self.activation = activation
        initializer = chainer.initializers.GlorotUniform()
        with self.init_scope():

            # 1x32x32
            self.phi_block1 = OptimizedBlock(init_ch, ch)
            self.phi_block2 = Block(ch, ch * 2, activation=activation, downsample=True)
            self.phi_block3 = Block(ch * 2, ch * 2, activation=activation, downsample=True)
            # ch*2x8x8 => basically 6*2x8*8 = 768

            self.enc1 = Block1(init_ch, ch, activation=activation, is_shortcut=True, dim_z=dim_z)
            self.enc2 = Block1(ch, ch*2, activation=activation, is_shortcut=True, dim_z=dim_z)
            self.enc3 = Block1(ch*2, ch*2, activation=activation, is_shortcut=True, dim_z=dim_z)
            self.linear = L.Linear(ch * 2 * (bottom_size * bottom_size), ch * 2 * (bottom_size//8 * bottom_size//8))

            self.l8 = SNLinear(ch*2*(bottom_size//8)*(bottom_size//8),
                               1,
                               initialW=initializer)


    def _embedding(self, x_ref, z):
        h = x_ref
        h = self.enc1(h, z)
        h = self.enc2(h, z)
        h = self.enc3(h, z)

        h = F.reshape(h, (h.shape[0], -1))
        h = self.activation(h)
        h = self.linear(h)

        return h

    def __call__(self, x, x_ref, z):
        h = x
#        h = F.concat((h, z), axis=1)
        h = self.phi_block1(h)
        h = self.phi_block2(h)
        h = self.phi_block3(h)
        h = self.activation(h)
#        h = F.sum(h, axis=(2, 3))
        C, H, W = h.shape[1], h.shape[2], h.shape[3]
        h = F.reshape(h, (-1, C*H*W))
        output = self.l8(h)
        embed = self._embedding(x_ref, z)
        output += F.sum(embed * h, axis=1, keepdims=True)
        return output

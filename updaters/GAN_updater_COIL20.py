# -*- coding: utf-8 -*-
# !/usr/bin/env python

from __future__ import print_function

import chainer
import chainer.functions as F
from chainer import Variable
import numpy as np
import cupy as cp

def loss_dcgan_dis(dis_fake, dis_real):
    L1 = F.mean(F.softplus(-dis_real))
    L2 = F.mean(F.softplus(dis_fake))
    loss = L1 + L2
    return loss

def loss_dcgan_gen(dis_fake):
    loss = F.mean(F.softplus(-dis_fake))
    return loss

def loss_hinge_dis(dis_fake, dis_real):
    loss = F.mean(F.relu(1. - dis_real))
    loss += F.mean(F.relu(1. + dis_fake))
    return loss

def loss_hinge_gen(dis_fake):
    loss = -F.mean(dis_fake)
    return loss

def loss_mse_gen(x_gen, x_real):
    loss = F.mean_squared_error(x_gen, x_real)
    return loss


class Updater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self.img_size = kwargs.pop('img_size')
        self.loss_type = kwargs.pop('loss_type')
        self.dim_noise = kwargs.pop('dim_noise')
        self.dif_index = kwargs.pop('dif_index')
        self.n_dis = kwargs.pop('n_dis')
        self.dataset = kwargs.pop('dataset')

        if self.loss_type == 'dcgan':
            self.loss_gen = loss_dcgan_gen
            self.loss_dis = loss_dcgan_dis
        elif self.loss_type == 'hinge':
            self.loss_gen = loss_hinge_gen
            self.loss_dis = loss_hinge_dis
        elif self.loss_type == 'mse':
            self.loss_gen = loss_mse_gen

        if self.dataset == 'mnist':
            from datasets.mnist.loader import get_ref_and_real_data
            self.get_data = get_ref_and_real_data
        elif self.dataset == 'coil20':
            from datasets.coil20.loader import get_ref_and_real_data
            self.get_data = get_ref_and_real_data
        elif self.dataset == 'coil100':
            from datasets.coil100.loader import get_ref_and_real_data
            self.get_data = get_ref_and_real_data
        else:
            print('please select dataset')
            exit(1)

        super(Updater, self).__init__(*args, **kwargs)


    def get_batch(self):
        batch = self.get_iterator('main').next()
        batch_size = len(batch)

        if self.dataset == 'mnist':
            ref_images, rot_images, eps = self.get_data(
                batch,
                0.4,
                20
            )
        elif self.dataset == 'coil20':
            ref_images, rot_images, eps = self.get_data(
                batch,
                img_size=self.img_size,
                dim_noise=self.dim_noise,
                dif_index=self.dif_index
            )
        elif self.dataset == 'coil100':
             ref_images, rot_images, eps = self.get_data(
                batch,
                img_size=self.img_size,
                dim_noise=self.dim_noise,
                dif_index=self.dif_index
            )
        else:
            exit(0)

        x_real = Variable(self.converter(rot_images, self.device))
        x_ref = Variable(self.converter(ref_images, self.device))
        eps = Variable(self.converter(eps, self.device))

        x_real = Variable(x_real.data.astype(np.float32)) / 255.
        x_ref= Variable(x_ref.data.astype(np.float32)) / 255.
        eps = Variable(eps.data.astype(np.float32))

        return x_ref, x_real, eps

    def calc_acc(self, dis_out, label=0, plot_name='acc'):
        prob = F.sigmoid(dis_out)

        label = cp.array([[label] for i in range(len(dis_out))])
        label = Variable(self.converter(label, self.device))
        label = Variable(label.data.astype(int))
        acc_real = F.binary_accuracy(prob - 0.5, label)
        chainer.report({plot_name: acc_real}, self.dis)

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')
        gen, dis = self.gen, self.dis

        if self.loss_type == 'mse':
             x_ref, x_real, eps = self.get_batch()
             H, W = x_ref.shape[2], x_ref.shape[3]
             _z = F.broadcast_to(
                F.reshape(eps, (eps.shape[0], eps.shape[1], 1, 1)),
                (eps.shape[0], eps.shape[1], H, W)
             )

             x_gen = gen(x=x_ref, z=eps)
             loss_gen = self.loss_gen(x_gen, x_real)
             gen.cleargrads()
             loss_gen.backward()
             gen_optimizer.update()
             chainer.report({'loss': loss_gen}, gen)
        else:
            for i in range(self.n_dis):
                x_ref, x_real, eps = self.get_batch()
                H, W = x_ref.shape[2], x_ref.shape[3]

                _z = F.broadcast_to(
                    F.reshape(eps, (eps.shape[0], eps.shape[1], 1, 1)),
                    (eps.shape[0], eps.shape[1], H, W)
                )

                if i == 0:
                    x_gen = gen(x=x_ref, z=eps)
                    dis_fake = dis(x_gen, x_ref, eps)
                    loss_gen = self.loss_gen(dis_fake=dis_fake)
                    gen.cleargrads()
                    loss_gen.backward()
                    gen_optimizer.update()
                    chainer.report({'loss': loss_gen}, gen)

                dis_real= dis(x_real, x_ref, eps)
                self.calc_acc(dis_real, label=1, plot_name='acc_real')

                x_gen = gen(x=x_ref, z=eps)
                dis_fake = dis(x_gen, x_ref, eps)

                self.calc_acc(dis_fake, label=0, plot_name='acc_fake')
                x_gen.unchain_backward()

                loss_dis = self.loss_dis(dis_fake=dis_fake, dis_real=dis_real)
                dis.cleargrads()
                loss_dis.backward()
                dis_optimizer.update()
                chainer.report({'loss': loss_dis}, dis)

# sigmoid_cross_entropy(x,1) = softplus(-x)
# sigmoid_cross_entropy(x,0) = softplus(x)

        # TODO: fix this code
        #is_independent_noise = False
        #if is_independent_noise:
        #    eps = np.random.binomial(1, 0.5, len(batch))


#        if self.add_noise_to_dis:
#            _eps = F.broadcast_to(eps, (_h*_w, _b, eps.shape[-1]))
#            _eps = F.transpose(_eps, (1, 2, 0))
#            _eps = F.reshape(_eps, (_b, eps.shape[-1], _h, _w))
#            in_dis_x = F.concat((x_real, x_ref, _eps))
#         if self.add_noise_to_dis:
#            _eps = F.broadcast_to(eps, (_h*_w, _b, eps.shape[-1]))
#            _eps = F.transpose(_eps, (1, 2, 0))
#            _eps = F.reshape(_eps, (_b, eps.shape[-1], _h, _w))
#            in_dis_x = F.concat((x_gen, x_ref, _eps))
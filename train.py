# -*- coding: utf-8 -*-
# !/usr/bin/env python

from __future__ import print_function
import argparse

import os
import chainer
import numpy as np
import uuid
from chainer import training
from chainer.training import extensions
import matplotlib
matplotlib.use('Agg')
from utils.config import Config

from visualize import out_generated_image


def make_optimizer(model, alpha=0.001, beta1=0.0, beta2=0.9):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    optimizer.setup(model)
    #optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
    return optimizer



def main():
    parser = argparse.ArgumentParser(description='3d pose generator')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='Number of images in each mini-batch')
    parser.add_argument('--img_size', '-is', type=int, default=128)
    parser.add_argument('--iteration', '-i', type=int, default=100, help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='', help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
    parser.add_argument('--snapshot_interval', type=int, default=50000, help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=10, help='Interval of displaying log to console')
    parser.add_argument('--visualize_interval', type=int, default=20, help='Interval of visualizing')
    parser.add_argument('--visualize_size', type=int, default=10, help='Interval of visualizing')
    parser.add_argument('--loss_type', type=str, default='dcgan', choices=['dcgan', 'hinge', 'wgan', 'mse'], help='')
    parser.add_argument('--n_dis', type=int, default=5)
    parser.add_argument('--dim_noise', type=int, default=3)
    parser.add_argument('--dif_index', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='coil20', choices=['mnist', 'coil20', 'coil100'])

    args = parser.parse_args()

    uid = str(uuid.uuid4())[:8]

    if args.out == '':
        config = Config()

        config.add('batch_size', args.batch_size)
        config.add('img_size', args.img_size)
        config.add('iteration', args.iteration)
        config.add('loss_type', args.loss_type)
        config.add('dim_noise', args.dim_noise)
        config.add('display_interval', args.display_interval)
        config.add('snapshot_interval', args.snapshot_interval)
        config.add('visualize_interval', args.visualize_interval)
        config.add('visualize_size', args.visualize_size)
        config.add('dif_index', args.dif_index)
        config.add('n_dis', args.n_dis)
        config.add('dataset', args.dataset)

        args.out = 'result/{}'.format(uid)
        os.makedirs(args.out)

        config.save(os.path.join(args.out, 'config.yml'))

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batch_size))
    print('# iteration: {}'.format(args.iteration))
    print('# out dir: {}'.format(args.out))
    print('')

    from dis_models.discriminator import Discriminator
    from gen_models.generator import Generator
#    from models.net_MNIST_no_encode import Generator
#    from models.net_MNIST import Discriminator

    if args.dataset == 'coil100':
        init_ch = 3
    else:
        init_ch = 1
    gen = Generator(init_ch=init_ch, dim_z=args.dim_noise, bottom_size=args.img_size)
    dis = Discriminator(init_ch=init_ch, dim_z=args.dim_noise, bottom_size=args.img_size)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()

    # Setup an optimizer

    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    if args.dataset == 'mnist':
        train, _ = chainer.datasets.get_mnist(withlabel=False, ndim=3, scale=255.)  # ndim=3 : (ch,width,height)
        ref_images = train[:args.visualize_size]
        from datasets.mnist.loader import get_ref_and_real_data
        data = get_ref_and_real_data(ref_images, 0.4, 20)
    elif args.dataset == 'coil20':
        from datasets.coil20.loader import get_ref_and_real_data, get_train_index, get_train_index_rand
        train = get_train_index()
        indexes = get_train_index_rand(size=args.visualize_size)
        ref_index = train[indexes]
        data = get_ref_and_real_data(ref_index, img_size=args.img_size, dim_noise=args.dim_noise, dif_index=args.dif_index)
    elif args.dataset == 'coil100':
        from datasets.coil100.loader import get_ref_and_real_data, get_train_index, get_train_index_rand
        train = get_train_index()
        indexes = get_train_index_rand(size=args.visualize_size)
        ref_index = train[indexes]
        data = get_ref_and_real_data(ref_index, img_size=args.img_size, dim_noise=args.dim_noise, dif_index=args.dif_index)

    train_iter = chainer.iterators.SerialIterator(train, args.batch_size)

    from updaters.GAN_updater_COIL20 import Updater


    # TODO: refactoring
    updater = Updater(
        models=(gen, dis),
        img_size=args.img_size,
        loss_type=args.loss_type,
        dim_noise=args.dim_noise,
        dif_index=args.dif_index,
        n_dis=args.n_dis,
        dataset=args.dataset,
        iterator=train_iter,
        optimizer={
            'gen': opt_gen, 'dis': opt_dis
        },
        device=args.gpu)

    trainer = training.Trainer(updater, (args.iteration, 'iteration'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    visualize_interval = (args.visualize_interval, 'iteration')
    trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object( gen, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object( dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))

    trainer.extend(
        extensions.PrintReport(
            ['epoch', 'iteration', 'gen/loss', 'dis/loss', 'dis/acc_real', 'dis/acc_fake']
        ),
        trigger=display_interval
    )

    trainer.extend(extensions.ProgressBar(update_interval=10))

    # Visualizer

    trainer.extend(
        out_generated_image(
            gen,
            data,
            args.out,
            args.gpu
        ),
        trigger=visualize_interval)

    # Loss visualize
    triner.extend(
        extensions.PlotReport(
            ['gen/loss', 'dis/loss'],
            'iteration',
            file_name='loss.png'
        ),
        trigger=display_interval
    )

    # Accuracy visualize
    trainer.extend(
        extensions.PlotReport(
            ['dis/acc_real', 'dis/acc_fake'],
            'iteration',
            file_name='acc.png'
        ),
        trigger=display_interval
    )

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

    chainer.serializers.save_npz(open(os.path.join(args.out, 'gen.npz')), gen)
    chainer.serializers.save_npz(open(os.path.join(args.out, 'dis.npz')), dis)


if __name__ == '__main__':
    main()
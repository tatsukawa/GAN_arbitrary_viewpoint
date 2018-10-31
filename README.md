# Generate Image in Arbitrary Viewpoint using Controllable Noise

Most source codes are referenced from [GANs with spectral normalization and projection discriminator](https://github.com/pfnet-research/sngan_projection)

## How to use
Firstly, you have to download datasets.
So, please see the dataset repository.

After downloading the dataset, you can run the training or prediction scripts.

For example, 
### MNIST
```sh
python3 train.py --epoch 400 --img_size 28 --out result --loss_type dcgan --dataset mnist
```

### COIL20
```sh
python3 train.py --epoch 400 --img_size 32 --out result --loss_type dcgan --dataset coil20
```

### COIL100
```sh
python3 train.py --epoch 400 --img_size 32 --out result --loss_type dcgan --dataset coil100
```

### Prediction
```sh
python3 predict.py --config_path result --gen_snapshot result/gen_snapshot --dis_snapshot result/dis_snapshot
```

## Result

This is a rotation image when coil20 dataset is given once.
![COIL20 result](https://github.com/tatsukawa/GAN_arbitrary_viewpoint/blob/master/images/rot.gif)

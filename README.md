# GAN-IS

## Cifar10
simply check whether gradient flows to cifar10 GAN model
``` bash
  python cifar10_IS_gradient_check.py --imageSize 32 --nsamples 100 --netG 'pretrained/generator/path' --netD 'pretrained/discriminator/path'
```

## MNIST
simply check whether gradient flows to GAN model
``` bash
  python mnist_IS_gradient_check.py --nsamples 100 --cuda
```

simply evaluate the model with FID score
``` bash
  python mnist_calculate_fid.py --nsamples 60000 --cuda
```


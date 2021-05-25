import argparse
import os, shutil
from random import shuffle
import numpy as np

import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# import the models
from cifar10_dcgan.dcgan import Generator

from metrics.fid_score import calculate_fid_given_paths

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='cifar10_dcgan/weights/netG_epoch_199.pth', help="path to netG (to continue training)")
parser.add_argument('--real_data_root', default='data/cifar10', help="path of real dataset")

parser.add_argument('--outdir', default='./samples', help='folder to save images')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nsamples', type=int, default=60000, required=True, help='number of image to evaluate IS')
parser.add_argument('--num-workers', type=int, default=8,help='Number of processes to use for data loading')
parser.add_argument('--device', type=str, default=None,help='Device to use. Like cuda, cuda:0 or cpu')

args = parser.parse_args()
print(args)

num_gpu = 1 if torch.cuda.is_available() else 0

# GAN
netG = Generator(ngpu=1).eval()

# load weights
netG.load_state_dict(torch.load(args.netG))

if torch.cuda.is_available():
    netG = netG.cuda()

latent_size = args.nz
batch_size = args.batchSize
num_images = args.nsamples

data_train = CIFAR10(args.real_data_root,
                    download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]))
                    
data_test = CIFAR10(args.real_data_root,
                    train=False,
                    download=True,
                    transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]))

data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=batch_size, num_workers=8)

real_test_images = torch.zeros((0,3,32,32)) # nsamples * # of channel * H * W
real_train_images = torch.zeros((0,3,32,32)) # nsamples * # of channel * H * W
fake_images = torch.zeros((num_images,3,32,32)) # nsamples * # of channel * H * W

# patch fake images, real images
print("Generating and Saving images")
# generate real test images
for real_batch in data_test_loader:
    real_test_images = torch.cat([real_test_images,torch.tensor(real_batch[0])],dim=0)
# generate real train images
for real_batch in data_train_loader:
    real_train_images = torch.cat([real_train_images,torch.tensor(real_batch[0])],dim=0)
# generate fake images
for s in range(0,num_images,batch_size):
    if s+batch_size>=num_images:
        e=num_images
        cur_batch_size = e-s
    else:
        e=s+batch_size
        cur_batch_size = batch_size
    
    fixed_noise = torch.randn(cur_batch_size, latent_size, 1, 1)
    if torch.cuda.is_available():
        with torch.no_grad():
            fake_images[s:e] = netG(fixed_noise.cuda().detach()).cpu()
        del fixed_noise
        torch.cuda.empty_cache()
    else:
        fake_images[s:e] = netG(fixed_noise).cpu()
    
real_test_path = os.path.join(args.outdir,'real_test')
real_train_path = os.path.join(args.outdir,'real_train')
fake_path = os.path.join(args.outdir,'fake')
image_dir_pathes = [real_test_path, real_train_path, fake_path]

for image_path in image_dir_pathes:
    if os.path.exists(image_path):
        shutil.rmtree(image_path)
    os.makedirs(image_path,exist_ok=True)

# save real images in npy
for i,image in enumerate(real_test_images):
    np.save(os.path.join(real_test_path,str(i)),image)
# save real images in npy
for i,image in enumerate(real_train_images):
    np.save(os.path.join(real_train_path,str(i)),image)
# save fake images in npy
for i,image in enumerate(fake_images):
    np.save(os.path.join(fake_path,str(i)),image)

if args.cuda is True:
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
else:
    device = torch.device('cpu')




fid_value = calculate_fid_given_paths([real_test_path, fake_path],
                                        batch_size,
                                        device,
                                        2048,
                                        lenet=None,
                                        num_workers=args.num_workers)
print('FID on CIFAR10 (test VS GAN)): ', fid_value)

fid_value = calculate_fid_given_paths([real_test_path, real_train_path],
                                        batch_size,
                                        device,
                                        2048,
                                        lenet=None,
                                        num_workers=args.num_workers)                                        
print('FID on CIFAR10 (test VS train): ', fid_value)



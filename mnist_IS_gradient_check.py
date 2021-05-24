import argparse
import torch

# load the models
from mnist_dcgan.dcgan import Discriminator, Generator
from mnist_classifier.lenet import LeNet5

from metrics.inception_score import inception_score

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='mnist_dcgan/weights/netG_epoch_99.pth', help="path to netG (to continue training)")
parser.add_argument('--netD', default='mnist_dcgan/weights/netD_epoch_99.pth', help="path to netD (to continue training)")
parser.add_argument('--netC', default='mnist_classifier/weights/lenet_epoch=28_test_acc=0.991.pth', help="path to netC (to continue training)")

parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nsamples', type=int, default=1000, required=True, help='number of image to evaluate IS')

# check whether there is a gradient flow in the model
def print_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    print("gradient check", ave_grads)

args = parser.parse_args()
print(args)

num_gpu = 1 if torch.cuda.is_available() else 0

# GAN
netD = Discriminator(ngpu=1).train()
netG = Generator(ngpu=1).train()
# classifier
netC = LeNet5().eval()

# load weights
netD.load_state_dict(torch.load(args.netD))
netG.load_state_dict(torch.load(args.netG))
netC.load_state_dict(torch.load(args.netC))

if torch.cuda.is_available():
    netD = netD.cuda()
    netG = netG.cuda()
    netC = netC.cuda()

latent_size = args.nz
batch_size = args.batchSize
num_images = args.nsamples

fake_images = torch.zeros((num_images,1,28,28)) # nsamples * # of channel * H * W

for s in range(0,num_images,batch_size):
    if s+batch_size>=num_images:
        e=num_images
        cur_batch_size = e-s
    else:
        e=s+batch_size
        cur_batch_size = batch_size
    fixed_noise = torch.randn(cur_batch_size, latent_size, 1, 1)
    if torch.cuda.is_available():
        fixed_noise = fixed_noise.cuda()
    fake_images[s:e] = netG(fixed_noise).cpu()

optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

optimizerG.zero_grad()

# IS = (mean(inception_score), std(inception_score))
IS = inception_score(fake_images, cuda=True, batch_size=32, resize=False, splits=10, classifier=netC, log_logit=True)
IS[0].backward()

print_grad_flow(netG.named_parameters())

optimizerG.step()

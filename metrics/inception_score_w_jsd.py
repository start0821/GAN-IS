# https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

from tqdm import tqdm

# FIX: classifier
def inception_score(imgs, score=False, cuda=True, batch_size=33, resize=False, splits=1, classifier=None, log_logit=False, true_dist=None, avg=False, weights = 1, div =0):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Classifier
    splits -- number of splits
    """
    N = len(imgs)


    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor


    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    ### Load pretrained classifier
    if classifier is None:
        # Load inception model
        inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
        inception_model.eval()
        up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
        def get_pred(x):
            if resize:
                x = up(x)
            x = inception_model(x)
            return F.softmax(x).data
    else:
        classifier.eval()
        def get_pred(x):
            x = classifier(x)
            if log_logit:
                return x.exp()
            else:
                return x

    # Get predictions
    output_sample = next(iter(dataloader))
    if cuda:
        output_sample = output_sample.cuda()
    output_shape = get_pred(output_sample).shape
    preds = torch.zeros((N, output_shape[-1]))
    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batch_size_i = batch.size()[0]
        if cuda:
            batch = batch.cuda()
        if score:
            preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batch).cpu()
        else:
            preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batch)

    if true_dist is not None and cuda:
        true_dist = true_dist.cpu()



    # Now compute the mean kl-div
    split_scores = torch.zeros(splits)
    split_reg_term = torch.zeros(splits)
    
    kl_d = torch.nn.KLDivLoss(reduction='sum')

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]

        py = torch.mean(part, axis=0)
        
        scores = torch.zeros(part.shape[0])
        reg_term = torch.zeros(part.shape[0])
        for i in range(part.shape[0]):
            pyx = part[i, :]
            if true_dist is not None and avg == False:
                scores[i] = kl_d(py.log(),pyx) 
                reg_term[i] = kl_d(py.log(), true_dist)
            elif true_dist is not None and avg == True:
                scores[i] = kl_d(py.log(),pyx) 
                reg_term[i] = 0.5 * (kl_d(py.log(), 0.5 * (true_dist + py) ) + kl_d(true_dist.log(), 0.5 * (py + true_dist)))
            else:
                scores[i] = kl_d(py.log(),pyx)
        split_scores[k] = torch.exp(torch.mean(scores))
        split_reg_term[k] = torch.exp(torch.mean(reg_term))
    return torch.mean(split_scores), torch.std(split_scores), torch.mean(split_reg_term), torch.std(split_reg_term)

if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    cifar = dset.CIFAR10(root='data/', download=True,
                             transform=transforms.Compose([
                                 transforms.Scale(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])
    )

    IgnoreLabelDataset(cifar)

    print ("Calculating Inception Score...")
    print (inception_score(IgnoreLabelDataset(cifar), cuda=True, batch_size=200, resize=True, splits=10))

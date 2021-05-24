import torch.nn as nn
from collections import OrderedDict

from mnist_classifier.lenet import LeNet5 as VanillaLeNet5

class LeNet5(nn.Module):

    DEFAULT_BLOCK_INDEX = 1

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        16: 0,   # First max pooling features
        120: 1,  # Second max pooling featurs
        84: 2,  # Pre-aux classifier features
        10: 3  # final logits
    }

    """
    Input - 1x28x28
    C1 - 6@24x24 (5x5 kernel)
    tanh
    S2 - 6@12x12 (2x2 kernel, stride 2) Subsampling
    C3 - 16@8x8 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@4x4 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (4x4 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """   
    def __init__(self,
                lenet: VanillaLeNet5,
                output_blocks=(DEFAULT_BLOCK_INDEX,)):
        super(LeNet5, self).__init__()

        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        
        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'
        
        self.blocks = nn.ModuleList()

        # Block 0
        block0 = [
            lenet.convnet[0],
            lenet.convnet[1],
            lenet.convnet[2],
            lenet.convnet[3],
            lenet.convnet[4],
            lenet.convnet[5],
        ]
        self.blocks.append(nn.Sequential(*block0))
        
        # Block 1
        if self.last_needed_block >= 1:
            block1 = [
                lenet.convnet[6],
                lenet.convnet[7],
            ]
            self.blocks.append(nn.Sequential(*block1))
        
        # Block 2
        if self.last_needed_block >= 2:
            block2 = [
                lenet.fc[0],
                lenet.fc[1],
            ]
            self.blocks.append(nn.Sequential(*block2))
        
        # Block 3
        if self.last_needed_block >= 3:
            block3 = [
                lenet.fc[2],
            ]
            self.blocks.append(nn.Sequential(*block3))


    def forward(self, img):
        outp = []
        x = img

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)
            if idx == 1:
                x = x.view(img.size(0), -1)

            if idx == self.last_needed_block:
                break
        return outp

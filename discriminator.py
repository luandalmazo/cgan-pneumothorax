"""
Using https://arxiv.org/pdf/1711.11585.pdf (Pix2PixHD)
or https://arxiv.org/pdf/1703.10593.pdf (CycleGAN)
appendix notation.
"""

import torch.nn as nn
import torch
import torchvision

class Conv4(nn.Module):
    """Ck"""
    def __init__(self, input_channels, output_channels, kernel_size=4):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=2, padding=1, padding_mode="reflect")
        self.activation = nn.LeakyReLU(0.2)

        # if should_norm:
            # self.pipeline = nn.Sequential(self.conv, self.norm, self.activation)
        # else:
        self.pipeline = nn.Sequential(self.conv, self.activation)


    def forward(self, x):
        return self.pipeline(x)

class Discriminator(nn.Module):
    """
    After the last layer, we apply a convolution to produce a
    1-dimensional output. We do not use InstanceNorm for the
    first C64 layer. We use leaky ReLUs with a slope of 0.2.
    The discriminator architecture is:
    C64-C128-C256-C512
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.c1 = Conv4(1 + self.num_classes, 16)
        self.c2 = Conv4(16, 32)
        self.c3 = Conv4(32, 64)
        self.c4 = Conv4(64, 128)
        self.c5 = Conv4(128, 256)
        self.c6 = Conv4(256, 512)
        self.c7 = Conv4(512, 512)

        self.last_conv = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        # self.linear = nn.Linear()
        # self.nonlinear = nn.Sigmoid()
        self.pipeline = nn.Sequential(self.c1, self.c2,self.c3, self.c4, self.c5, self.c6, self.c7, self.last_conv)

    def forward(self, images, labels):
        n = self.num_classes
        # images -> images[i] has label labels[i]
        onehot = torch.nn.functional.one_hot(labels, n)
        onehot = onehot[:, :, None, None] # no idea what this does, but it works
        images_onehot = onehot.repeat(1, 1, 256, 256) # transform onehot vectors into onehot matrices
        x = torch.concatenate((images, images_onehot), dim=1)
        x = self.pipeline(x)

        x = x.view(len(x), -1) # flatten it?
        return x

if __name__ == "__main__":
    gen = Discriminator(num_classes=2)
    inp = torch.zeros(2, 1, 256, 256)
    print(inp.shape)

    out = gen(inp, torch.tensor([0,1]))
    print(out.shape)
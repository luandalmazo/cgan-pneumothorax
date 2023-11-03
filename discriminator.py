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
    def __init__(self, input_channels, output_channels, should_norm=False):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1, padding_mode="reflect")
        # self.norm = nn.InstanceNorm2d(output_channels)
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

        self.c1 = Conv4(1 + self.num_classes, 64, should_norm=False)
        self.c2 = Conv4(64, 128)
        self.c3 = Conv4(128, 256)
        self.c4 = Conv4(256, 512)
        self.last_conv = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1, padding_mode="reflect")

        self.pipeline = nn.Sequential(self.c1, self.c2,self.c3, self.c4, self.last_conv)

    def forward(self, images, labels):
        n = self.num_classes
        # images -> images[i] has label labels[i]

        onehot = torch.nn.functional.one_hot(labels, n)
        # print("ONEHOT:", onehot)
        # print("ONEHOT SHAPE", onehot.shape)
        onehot = onehot[:, :, None, None] # no idea what this does, but it works
        images_onehot = onehot.repeat(1, 1, 256, 256) # transform onehot vectors into onehot matrices
        # print("ONEHOT IMAGE:", images_onehot)
        # print("ONEHOT IMAGE SHAPE", images_onehot.shape)

        x = torch.concatenate((images, images_onehot), dim=1)
        print("X SHAPE", x.shape) 


        return self.pipeline(x)


if __name__ == "__main__":
    gen = Discriminator(num_classes=2)
    inp = torch.zeros(2, 1, 256, 256)
    print(inp.shape)

    out = gen(inp, torch.tensor([0,1]))
    print(out.shape)
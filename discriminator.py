"""
Using https://arxiv.org/pdf/1711.11585.pdf (Pix2PixHD)
or https://arxiv.org/pdf/1703.10593.pdf (CycleGAN)
appendix notation.
"""

import torch.nn as nn
import torch
import torchvision

# class Conv4(nn.Module):
#     """Ck"""
#     def __init__(self, input_channels, output_channels, kernel_size=4, should_norm=True):
#         super().__init__()

#         self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=2, padding=1, padding_mode="reflect")
#         # self.norm = nn.InstanceNorm2d(output_channels)
#         self.norm = nn.BatchNorm2d(output_channels)
#         self.should_norm = should_norm

        
#         self.activation = nn.LeakyReLU(0.2)

#         # if should_norm:
#             # self.pipeline = nn.Sequential(self.conv, self.norm, self.activation)
#         # else:
#         # self.pipeline = nn.Sequential(self.conv, self.activation)


#     def forward(self, x):
#         x = self.conv(x)
#         if self.should_norm:
#             x = self.norm(x)
#         x = self.activation(x)
#         # print(x.shape)
#         return x

# class Discriminator(nn.Module):
#     """
#     After the last layer, we apply a convolution to produce a
#     1-dimensional output. We do not use InstanceNorm for the
#     first C64 layer. We use leaky ReLUs with a slope of 0.2.
#     The discriminator architecture is:
#     C64-C128-C256-C512
#     """
#     def __init__(self, num_classes):
#         super().__init__()
#         self.num_classes = num_classes

#         # self.c1 = Conv4(1 + self.num_classes, 16)
#         self.c1 = Conv4(1 + self.num_classes, 32)
#         # self.c2 = Conv4(16, 32)
#         # self.c3 = Conv4(32, 64)
#         self.c2 = Conv4(32, 64)
#         # self.c4 = Conv4(64, 128)
#         self.c3 = Conv4(64, 128)
#         self.c4 = Conv4(128, 256)
#         # self.c6 = Conv4(256, 512)
#         # self.c7 = Conv4(512, 512)

#         # self.last_conv = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
#         self.last_conv = nn.Conv2d(256, 1, kernel_size=4, stride=1)
#         # self.linear = nn.Linear()
#         # self.nonlinear = nn.Sigmoid()
#         # self.pipeline = nn.Sequential(self.c1, self.c2,self.c3, self.c4, self.c5, self.c6, self.c7, self.last_conv)
#         self.pipeline = nn.Sequential(self.c1, self.c2,self.c3, self.c4, self.last_conv)

#     def forward(self, images, labels):
#         n = self.num_classes
#         # images -> images[i] has label labels[i]
#         onehot = torch.nn.functional.one_hot(labels, n)
#         onehot = onehot[:, :, None, None] # no idea what this does, but it works

#         input_size = images.shape[-1]
#         images_onehot = onehot.repeat(1, 1, input_size, input_size) # transform onehot vectors into onehot matrices
#         x = torch.concatenate((images, images_onehot), dim=1)
#         x = self.pipeline(x)

#         x = x.view(len(x), -1) # flatten it?
#         return x

class Discriminator(nn.Module):
    def __init__(self, nz=100, ndf=128, nc=1, num_classes=2):
        super(Discriminator, self).__init__()
        self.num_classes = 2
        self.nz = nz
        self.ndf = ndf
        self.nc=nc
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc+num_classes, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, images, labels):
        n = self.num_classes
        onehot = torch.nn.functional.one_hot(labels, n)
        onehot = onehot[:, :, None, None] # no idea what this does, but it works
        input_size = images.shape[-1]
        images_onehot = onehot.repeat(1, 1, input_size, input_size) # transform onehot vectors into onehot matrices
        x = torch.concatenate((images, images_onehot), dim=1)
        x = self.main(x)

        # x = x.view(len(x), -1) # flatten it?
        return x
    












######## 
# WGAN UTILS

def get_gradient(crit, real, fake):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    
    epsilon = torch.rand(len(real), 1, 1, 1, device=real.device, requires_grad=True)
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = crit(mixed_images)
    
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        # Note: You need to take the gradient of outputs with respect to inputs.
        # This documentation may be useful, but it should not be necessary:
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
        #### START CODE HERE ####
        inputs=mixed_images,
        outputs=mixed_scores,
        #### END CODE HERE ####
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient



if __name__ == "__main__":
    gen = Discriminator(num_classes=2)
    inp = torch.zeros(2, 1, 64, 64)
    print(inp.shape)

    out = gen(inp, torch.tensor([0,1]))
    print(out.shape)
"""
Not exactly a Pix2PixHD implementation as in 
https://github.com/Firdavs-coder/Aladdin-Persson-AI-Watermark-Destroy/tree/master ,

and using https://arxiv.org/pdf/1711.11585.pdf (Pix2PixHD)
or https://arxiv.org/pdf/1703.10593.pdf (CycleGAN)
appendix notation.
"""

import torch.nn as nn
import torch
import torchvision


class Down(nn.Module):
    """dk"""
    def __init__(self, input_channels, output_channels):
        super().__init__()

        # self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, bias=False, padding=1, padding_mode="reflect")
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, bias=True, padding=1, padding_mode="reflect")
        # self.norm = nn.InstanceNorm2d(output_channels)
        self.norm = nn.Identity()
        # self.norm = nn.BatchNorm2d(output_channels)
        self.activation = nn.ReLU(0.2)
        self.pipeline = nn.Sequential(self.conv, self.norm, self.activation)

    def forward(self, x):
        return self.pipeline(x)

class Conv7Stride1(nn.Module):
    """c7s1-k"""
    def __init__(self, input_channels, output_channels, activation, padding=3):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=7, stride=1, padding=padding, padding_mode="reflect")
        # self.norm = nn.InstanceNorm2d(output_channels)
        self.norm = nn.Identity()
        # self.norm = nn.BatchNorm2d(output_channels)
        self.activation = activation

        assert isinstance(self.activation, nn.Module)

        self.pipeline = nn.Sequential(self.conv, self.norm, self.activation)

    def forward(self, x):
        return self.pipeline(x)
        # x = self.conv(x)
        # x = self.norm(x)
        # x = self.activation(x)
        # return x

class Residual(nn.Module):
    """rk"""
    def __init__(self, num_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        # self.norm1 = nn.InstanceNorm2d(num_channels)
        self.norm1 = nn.Identity()
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        # self.norm2 = nn.InstanceNorm2d(num_channels)
        self.norm2 = nn.Identity()
        # self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return out + x


class Up(nn.Module):
    """uk, but it is an upsample-dk"""
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, bias=False, padding=1, padding_mode="reflect")
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, bias=True, padding=1, padding_mode="reflect")
        # self.norm = nn.InstanceNorm2d(output_channels)
        self.norm = nn.Identity()
        # self.norm = nn.BatchNorm2d(output_channels)
        self.activation = nn.ReLU(0.2)
        self.pipeline = nn.Sequential(self.upsample, self.conv, self.norm, self.activation)

    def forward(self, x):
        return self.pipeline(x)


class UnetGenerator(nn.Module):
    """
    CycleGAN paper:
    The network with 6 residual blocks consists of:
    c7s1-64,d128,d256,R256,R256,R256,
    R256,R256,R256,u128,u64,c7s1-3
    The network with 9 residual blocks consists of:
    c7s1-64,d128,d256,R256,R256,R256,
    R256,R256,R256,R256,R256,R256,u128
    u64,c7s1-3
    """
    def __init__(self, num_residuals=6, num_channels=1):
        super().__init__()
        self.num_residuals = num_residuals

        self.c7init = Conv7Stride1(num_channels + 1, 64, activation=nn.ReLU(0.2)) # RGB -> 64.
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.residuals = nn.Sequential(
                            *[Residual(256) for _ in range(self.num_residuals)]
                        )

        # remember skip connections
        self.up1 = Up(256*2, 128)
        self.up2 = Up(128*2, 64)
        # self.c7end = Conv7Stride1(64*2, num_channels, activation=nn.Sigmoid()) # 64 -> RGB
        self.c7end = Conv7Stride1(64*2, num_channels, activation=nn.Tanh()) # 64 -> RGB

    def forward(self, image):
        noise_channel = torch.randn_like(image)

        x = torch.concatenate((image, noise_channel), dim=1)

        # 3x256x256
        init = self.c7init(x) # 64x256x256
        d1 = self.down1(init) # 128x128x128 
        d2 = self.down2(d1) # 256x64x64        
        res = self.residuals(d2) # 256x64x64

        # skip d2
        res_d2 = torch.concatenate((res, d2), dim=1) # 512x64x64
        u1 = self.up1(res_d2) # 128x128x128

        #skip d1
        u1_d1 = torch.concatenate((u1, d1), dim=1) # 256x128x128
        u2 = self.up2(u1_d1) # 64x256x256

        # skip init
        u2_init = torch.concatenate((u2, init), dim=1) #128x256x256
        output = self.c7end(u2_init) #3x256x256

        # print("GEN OUTPUT SHAPE", output.shape)
        return output

Generator = UnetGenerator


class Conv4(nn.Module):
    """Ck"""
    def __init__(self, input_channels, output_channels, kernel_size=4, should_norm=False):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=2, padding=1, padding_mode="reflect")
        self.norm = nn.InstanceNorm2d(output_channels)
        # self.norm = nn.BatchNorm2d(output_channels)
        self.should_norm = should_norm

        
        self.activation = nn.LeakyReLU(0.2)

        # if should_norm:
            # self.pipeline = nn.Sequential(self.conv, self.norm, self.activation)
        # else:
        # self.pipeline = nn.Sequential(self.conv, self.activation)


    def forward(self, x):
        x = self.conv(x)
        if self.should_norm:
            x = self.norm(x)
        x = self.activation(x)
        # print(x.shape)
        return x

class PatchDiscriminator(nn.Module):
    """
    After the last layer, we apply a convolution to produce a
    1-dimensional output. We do not use InstanceNorm for the
    first C64 layer. We use leaky ReLUs with a slope of 0.2.
    The discriminator architecture is:
    C64-C128-C256-C512
    """
    def __init__(self, num_channels=1):
        super().__init__()
        self.num_channels = num_channels

        # self.c1 = Conv4(1 + self.num_classes, 16)
        self.c1 = Conv4(num_channels * 2, 64)
        # self.c2 = Conv4(16, 32)
        # self.c3 = Conv4(32, 64)
        self.c2 = Conv4(64, 128)
        # self.c4 = Conv4(64, 128)
        self.c3 = Conv4(128, 256)
        self.c4 = Conv4(256, 512)
        # self.c6 = Conv4(256, 512)
        # self.c7 = Conv4(512, 512)

        # self.last_conv = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        self.last_conv = nn.Conv2d(512, 1, kernel_size=4, stride=1)
        # self.linear = nn.Linear()
        # self.nonlinear = nn.Sigmoid()
        # self.pipeline = nn.Sequential(self.c1, self.c2,self.c3, self.c4, self.c5, self.c6, self.c7, self.last_conv)
        self.pipeline = nn.Sequential(self.c1, self.c2,self.c3, self.c4, self.last_conv)

    def forward(self, input_images, output_images):
        x = torch.concatenate((input_images, output_images), dim=1) 
        return self.pipeline(x)

    # def forward(self, images, labels):
    #     n = self.num_classes
    #     # images -> images[i] has label labels[i]
    #     onehot = torch.nn.functional.one_hot(labels, n)
    #     onehot = onehot[:, :, None, None] # no idea what this does, but it works

    #     input_size = images.shape[-1]
    #     images_onehot = onehot.repeat(1, 1, input_size, input_size) # transform onehot vectors into onehot matrices
    #     x = torch.concatenate((images, images_onehot), dim=1)
    #     x = self.pipeline(x)

    #     x = x.view(len(x), -1) # flatten it?
    #     return x

Discriminator = PatchDiscriminator




if __name__ == "__main__":
    print("GEN")
    gen = Generator()
    inp = torch.zeros(2, 1, 256, 256)
    print(inp.shape)

    out = gen(inp)

    print(out.shape)


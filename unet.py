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

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, bias=False, padding=1, padding_mode="reflect")
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
        self.pipeline = nn.Sequential(self.conv, self.norm, self.activation)

    def forward(self, x):
        return self.pipeline(x)

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
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, bias=False, padding=1, padding_mode="reflect")
        # self.norm = nn.InstanceNorm2d(output_channels)
        self.norm = nn.Identity()
        # self.norm = nn.BatchNorm2d(output_channels)
        self.activation = nn.ReLU(0.2)
        self.pipeline = nn.Sequential(self.upsample, self.conv, self.norm, self.activation)

    def forward(self, x):
        return self.pipeline(x)


class Generator(nn.Module):
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
    def __init__(self, num_residuals=6):
        super().__init__()
        self.num_residuals = num_residuals

        # (TODO CALCULATE inp size and out sizes for all)

        self.c7init = Conv7Stride1(3, 64, activation=nn.ReLU(0.2)) # RGB -> 64.
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.residuals = nn.Sequential(
                            *[Residual(256) for _ in range(self.num_residuals)]
                        )

        # remember skip connections
        self.up1 = Up(256*2, 128)
        self.up2 = Up(128*2, 64)
        self.c7end = Conv7Stride1(64*2, 3, activation=nn.Sigmoid()) # 64 -> RGB
        # self.c7end = Conv7Stride1(64*2, 3, activation=nn.Tanh()) # 64 -> RGB

    def forward(self, x):
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


if __name__ == "__main__":
    gen = Generator()
    inp = torch.zeros(2, 3, 256, 256)
    print(inp.shape)

    out = gen(inp)

    print(out.shape)


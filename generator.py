import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import show_tensor_images

class UpConvBlock(nn.Module):
    '''
    Block for upsample-then-convolution in the Generator.
    Parameters:
        input_channels: how many channels the input feature representation has
        output_channels: how many channels the output feature representation should have
        kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
        stride: the stride of the convolution
        final_layer: a boolean, true if it is the final layer and false otherwise 
                    (affects activation and batchnorm)
    '''
    def __init__(self, input_channels, output_channels, kernel_size=4, stride=1, final_layer=False, batch_norm=True):
        super().__init__()
        if not final_layer:
            self.pipeline = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),  # Upsample
                nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, bias=False, padding=1, padding_mode="reflect"),
                nn.ReLU(inplace=True)  # ReLU activation
            )
        else:
            self.pipeline = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),  # Upsample
                nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, bias=False, padding=1, padding_mode="reflect"),
                # nn.Sigmoid()  # Sigmoid activation for the final layer
                nn.Tanh()  # Sigmoid activation for the final layer
            )

    def forward(self, x):
        return self.pipeline(x)

class Generator(nn.Module):
    '''
    Generator Class with upsample-then-convolution.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, input_dim=398, num_classes=2, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # self.up0 = nn.Upsample(scale_factor=4, mode="nearest")
        self.up1 = UpConvBlock((self.input_dim+self.num_classes)//4, hidden_dim * 8)
        self.up2 = UpConvBlock(hidden_dim * 8, hidden_dim * 4)
        self.up3 = UpConvBlock(hidden_dim * 4, hidden_dim * 4)
        self.up4 = UpConvBlock(hidden_dim * 4, hidden_dim * 2)
        self.up5 = UpConvBlock(hidden_dim * 2, hidden_dim * 2)
        self.up6 = UpConvBlock(hidden_dim * 2, hidden_dim)
        self.up7 = UpConvBlock(hidden_dim, im_chan, final_layer=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, labels):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, input_dim)
        '''
        noise = get_noise(len(labels), self.input_dim).to(self.device)

        # labels = labels.long()

        one_hot_vector = torch.nn.functional.one_hot(labels, self.num_classes)
        x = torch.concatenate((noise, one_hot_vector), dim=1)
        x = x.view(len(x), (self.input_dim+self.num_classes)//4, 2, 2)
        print(x)
        # print("-----")
        print(x.shape)
        # x = self.up0(x)        
        # print(x.shape)
        x = self.up1(x)
        print(x.shape)
        x = self.up2(x)
        print(x.shape)
        x = self.up3(x)
        print(x.shape)
        x = self.up4(x)
        print(x.shape)
        x = self.up5(x)
        print(x.shape)
        x = self.up6(x)
        print(x.shape)
        x = self.up7(x)
        print(x.shape)

        return x

def get_noise(n_samples, input_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, input_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        input_dim: the dimension of the input vector, a scalar
        device: the device type
    '''
    return torch.randn(n_samples, input_dim, device=device)

if __name__ == "__main__":
    gen = Generator().to("cuda:0")
    out = gen(torch.tensor([0]).to("cuda:0").long())
    print(out.shape)

    show_tensor_images(out, show="save", name="NOISESAO")


import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from utils import show_tensor_images

class ConditionalInput(nn.Module):
    """Constant Input param"""
    def __init__(self, channel, size=4, num_classes=2):
        super().__init__()
        self.size = size
        self.num_classes = num_classes
        self.learnable = nn.Parameter(torch.randn(1, channel-num_classes, size, size))

    def forward(self, labels):
        batch_size = len(labels)
        onehot = torch.nn.functional.one_hot(labels, self.num_classes)
        onehot = onehot[:, :, None, None] # no idea what this does, but it works
        images_onehot = onehot.repeat(1, 1, self.size, self.size) # transform onehot vectors into onehot matrices
        constant = self.learnable.repeat(batch_size, 1, 1, 1)
        out = torch.concatenate((constant, images_onehot), dim=1)

        return out

class ModulatedConv2d(nn.Module):
    """As in StyleGAN v2, weight modulation and demodulation applied to the weights of the convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, style_dim=256):
        super().__init__()
        self.eps = 1e-8
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.style_dim = style_dim
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        # Style modulation network
        self.style_mapping = nn.Linear(style_dim, in_channels)
        
    def forward(self, x, style):
        # Style modulation
        style = self.style_mapping(style)
        style = style.view(-1, self.in_channels, 1, 1)
        # Modulate weights
        modulated_weight = self.weight * style
        out = nn.functional.conv2d(x, modulated_weight, bias=self.bias, padding=1)
        # Demodulate
        std = torch.std(out, dim=(2, 3), keepdim=True)
        out = out / std
        return out

class UpStyleBlock(nn.Module):
    '''
    Block for upsample-then-convolution in the Generator.
    '''
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, final_layer=False, style_dim=512):
        super().__init__()
        self.style_dim = style_dim
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')  # Upsample
        self.conv = ModulatedConv2d(input_channels, output_channels, stride=stride,
                                    kernel_size=kernel_size, style_dim=self.style_dim,
                                    padding=1)
        if not final_layer:
            self.nonlinear =  nn.ReLU()  # ReLU activation
        else:
            # self.nonlinear =  nn.Tanh()  # Tanh activation
            self.nonlinear =  nn.Sigmoid()  # Sigma activation

    def forward(self, x, style):
        x = self.up(x)
        x = self.conv(x, style)
        x = self.nonlinear(x)
        return x

class StyleGenerator(nn.Module):
    '''
    Generator Class with upsample-then-convolution.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, style_dim=256, num_classes=2, im_chan=1):
        super().__init__()
        self.style_dim = style_dim
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.input = ConditionalInput(self.style_dim, size=4, num_classes=self.num_classes)

        # self.up0 = nn.Upsample(scale_factor=4, mode="nearest")
        self.up1 = UpStyleBlock(self.style_dim, 256, style_dim=style_dim)
        self.up2 = UpStyleBlock(256, 128, style_dim=style_dim)
        self.up3 = UpStyleBlock(128, 64, style_dim=style_dim)
        self.up4 = UpStyleBlock(64, 32, style_dim=style_dim)
        self.up5 = UpStyleBlock(32, 16, style_dim=style_dim)
        self.up6 = UpStyleBlock(16, im_chan, final_layer=True, style_dim=style_dim)

    def forward(self, labels):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, input_dim)
        '''
        noise = get_noise(1, self.style_dim).to(self.device).squeeze(0)
        # pre_style = self.style_transform(noise)
        # style = pre_style
        style = noise
       
        x = self.input(labels)
        x = self.up1(x, style)
        x = self.up2(x, style)
        x = self.up3(x, style)
        x = self.up4(x, style)
        x = self.up5(x, style)
        x = self.up6(x, style)
        # print(x.shape)
        return x

def get_noise(n_samples, input_dim, device='cpu'):
    return torch.randn(n_samples, input_dim, device=device)


class UpConvBlock(nn.Module):
    '''
    Block for upsample-then-convolution in the Generator.
    '''
    def __init__(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=1)

        if not final_layer:
            self.nonlinear =  nn.ReLU()  # ReLU activation
        else:
            # self.nonlinear =  nn.Tanh()  # Tanh activation
            self.nonlinear =  nn.Sigmoid()  # Sigma activation

    def forward(self, x):
        x = self.upconv(x)
        x = self.nonlinear(x)
        return x

class TransposedGenerator(nn.Module):
    def __init__(self, num_classes=2, im_chan=1):
        super().__init__()
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.size = 4
        
        # self.up0 = nn.Upsample(scale_factor=4, mode="nearest")
        self.up1 = UpConvBlock(512, 256)
        self.up2 = UpConvBlock(256, 128)
        self.up3 = UpConvBlock(128, 64)
        self.up4 = UpConvBlock(64, 32)
        self.up5 = UpConvBlock(32, 16)
        self.up6 = UpConvBlock(16, im_chan, final_layer=True)

    def get_input(self, labels):
        batch_size = len(labels)
        onehot = torch.nn.functional.one_hot(labels, self.num_classes)
        onehot = onehot[:, :, None, None] # no idea what this does, but it works
        images_onehot = onehot.repeat(1, 1, self.size, self.size) # transform onehot vectors into onehot matrices
        
        noise = torch.randn(batch_size, 512-self.num_classes, self.size, self.size).to(self.device)
        inp = torch.concatenate((noise, images_onehot), dim=1)
        return inp

    def forward(self, labels):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, input_dim)
        '''
       
        x = self.get_input(labels)
        # print(x.shape)
        x = self.up1(x)
        # print(x.shape)
        x = self.up2(x)
        # print(x.shape)
        x = self.up3(x)
        # print(x.shape)
        x = self.up4(x)
        # print(x.shape)
        x = self.up5(x)
        # print(x.shape)
        x = self.up6(x)
        # print(x.shape)
        return x

Generator = TransposedGenerator


if __name__ == "__main__":
    gen = Generator().to("cuda:0")
    out = gen(torch.tensor([0]).to("cuda:0").long())
    print(out.shape)

    show_tensor_images(out, show="save", name="EXAMPLE")


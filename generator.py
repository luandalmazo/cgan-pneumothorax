import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from utils import show_tensor_images

# self.channels = {       
# # 4: 512,
# # 8: 512,
# # 16: 512,
# # 32: 512,
# # 64: 256 * 2,
# # 128: 128 * 2,
# # 256: 64 * 2,
# # 512: 32 * 2,
# # 1024: 16 * 2,
# 4: 512,
# 8: 512,
# 16: 256,
# 32: 256,
# 64: 128,
# 128: 64,
# 256: 32
# }

class ConstantInput(nn.Module):
    """Constant Input param"""
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch_size = len(input)
        out = self.input.repeat(batch_size, 1, 1, 1)
        return out

class NoiseInjection(nn.Module):
    """Inject random noise into X"""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        if noise is None:
            batch_size, _, height, width = x.shape
            noise = x.new_empty(batch_size, 1, height, width).normal_()

        return x + self.weight * noise

class ModulatedConv2d(nn.Module):
    """As in StyleGAN v2, weight modulation and demodulation applied to the weights of the convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, style_dim=256):
        super().__init__()
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
        # print(self.weight)
        print(self.weight.shape)
        # print(style.shape)
        modulated_weight = self.weight * style

        # from lucidrains
        # Modulate weights
        # w1 = style[:, None, :, None, None]
        # w2 = self.weight[None, :, :, :, :]
        # modulated_weight = w2 * (w1 + 1)

        print(modulated_weight.shape)


        # Perform convolution
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
        self.noise = NoiseInjection()
        if not final_layer:
            self.nonlinear =  nn.ReLU()  # ReLU activation
        else:
            self.nonlinear =  nn.Tanh()  # Tanh activation

    def forward(self, x, style):
        x = self.up(x)
        x = self.conv(x, style)
        x = self.noise(x)
        x = self.nonlinear(x)
        return x

class Generator(nn.Module):
    '''
    Generator Class with upsample-then-convolution.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, style_dim=512, num_classes=2, im_chan=1):
        super(Generator, self).__init__()
        self.style_dim = style_dim

        self.style_transform = nn.Linear(style_dim-num_classes, style_dim-num_classes)

        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.input = ConstantInput(self.style_dim)

        # self.up0 = nn.Upsample(scale_factor=4, mode="nearest")
        self.up1 = UpStyleBlock(self.style_dim, 512)
        # self.up2 = UpStyleBlock(512, 256)
        self.up2 = UpStyleBlock(512, 256)
        self.up3 = UpStyleBlock(256, 128)
        self.up4 = UpStyleBlock(128, 64)
        self.up5 = UpStyleBlock(64, 32)
        self.up6 = UpStyleBlock(32, im_chan, final_layer=True)

    def forward(self, labels):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, input_dim)
        '''
        # noise = get_noise(len(labels), self.style_dim-2).to(self.device)
        noise = get_noise(1, self.style_dim-2).to(self.device).squeeze(0)

        pre_style = self.style_transform(noise)

        one_hot_vector = torch.nn.functional.one_hot(labels, self.num_classes)
        style = torch.concatenate((pre_style, one_hot_vector), dim=1)
       
        x = self.input(labels)
        print(x.shape)

        x = self.up1(x, style)
        print(x.shape)
        x = self.up2(x, style)
        print(x.shape)
        x = self.up3(x, style)
        print(x.shape)
        x = self.up4(x, style)
        print(x.shape)
        x = self.up5(x, style)
        print(x.shape)
        x = self.up6(x, style)
        print(x.shape)
        # x = self.up7(x, style)
        # print(x.shape)

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

    show_tensor_images(out, show="save", name="EXAMPLE")


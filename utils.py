from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn as nn


def show_tensor_images(image_tensor, num_images=250, size=(1, 28, 28), nrow=5, show="plot", name=None):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)

    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    # plt.imshow(image_grid.squeeze())
    if show == "plot":
        plt.show()
    elif show == "save":
        plt.savefig(f"fig{name}.png")
    

def weightsinit(m):
    # if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    if isinstance(m, nn.Conv2d):
        nn.init.normal(m.weight, 0.0, 0.02)
    # if isinstance(m, nn.InstanceNorm2d):
    #     nn.init.normal(m.weight, 0.0, 0.02)
    #     nn.init.constant(m.bias, 0)
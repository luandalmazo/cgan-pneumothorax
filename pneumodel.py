import torch.nn as nn
from torchvision.models import resnet18


class PneuModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.resnet = resnet18(weights = "IMAGENET1K_V1")

        # Black and white image
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Binary classification
        self.resnet.fc = nn.Linear(in_features=512, out_features=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        # Init these new weights
        nn.init.xavier_uniform_(self.resnet.conv1.weight)
        nn.init.xavier_uniform_(self.resnet.fc.weight)


    def forward(self, x):
        x = self.resnet(x)
        s = self.sigmoid(x)
        return s


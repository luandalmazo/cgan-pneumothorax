import torch
from torch.utils.data import random_split
from dataset import PneumoDataset
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
import argparse
import sys
from torch.nn import BCELoss

dataset = PneumoDataset()
train, validation = random_split(dataset, [0.8, 0.2])

parser = argparse.ArgumentParser(description='Pneumothorax Detection')
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--batch_size', default=16, type=int)
args = parser.parse_args()

epochs, lr, batch_size = args.epochs, args.lr, args.batch_size

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=True)

### HANDCRAFTED 
resnet = resnet18(weights = "IMAGENET1K_V1")
resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = BCELoss().to(device)
optimizer = Adam(resnet.parameters(), lr=lr)

softmax = torch.nn.Softmax().to(device)

resnet = resnet.to(device)

for epoch in range(epochs):
    mean_loss = 0
    resnet.train()
    for _, pair in enumerate(train_loader):
        image, label = pair
        image = image.to(device)
        label = label.to(device)


        optimizer.zero_grad()
        output = resnet(image)

        output = output.squeeze(1).float()
        # print(output)
        output = softmax(output)
        
        label = label.float()
        # print(output)
        # sys.stdin.flush()
        # print("OUTSHAPE", output.shape)
        # print("LABSHAPE", label.shape)

        loss = criterion(output, label)
        mean_loss += loss.item()
        loss.backward()
        optimizer.step()
        

    resnet.eval()
    mean_loss = mean_loss/len(train_loader)
    print(f'Epoch: {epoch}, Train Loss: {mean_loss}')

    with torch.no_grad():
        mean_loss = 0
        for _, pair in enumerate(validation_loader):
            image, label = pair
            image = image.to(device)
            label = label.to(device)
            output = resnet(image)

            output = output.squeeze(1).float()
            output = softmax(output)
            
            label = label.float()

            loss = criterion(output, label)
            mean_loss += loss.item()

    mean_loss = mean_loss/len(validation_loader)
    print(f'Epoch: {epoch}, Validation Loss: {mean_loss}')


torch.save(resnet, './trained_resnet.pkl')
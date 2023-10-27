import torch
from torch.utils.data_random_split import random_split
from dataset import PneumoDataset
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
import argparse
from torch.optim import BCELoss

dataset = PneumoDataset()
train, validation = random_split(dataset, [80, 20])

parser = argparse.ArgumentParser(description='Pneumothorax Detection')
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--batch_size', default=16, type=int)
args = parser.parse_args()

epochs, lr, batch_size = args.epochs, args.lr, args.batch_size

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=True)
resnet = resnet18(weights = "IMAGENET1K_V1")
criterion = BCELoss()
optimizer = Adam(resnet.parameters(), lr=lr)

for epoch in range(epochs):
    mean_loss = 0
    resnet.train()
    for _, pair in enumerate(train_loader):
        image, label = pair
        optimizer.zero_grad()
        output = resnet(image)
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
            output = resnet(image)
            loss = criterion(output, label)
            mean_loss += loss.item()

    mean_loss = mean_loss/len(validation_loader)
    print(f'Epoch: {epoch}, Validation Loss: {mean_loss}')



torch.save(resnet.state_dict(), './resnet.pth')
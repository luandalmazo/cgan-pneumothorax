import torch
from torch.utils.data import random_split
from dataset import PneumoDataset
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
import argparse
import sys
from torch.nn import BCELoss
from torchmetrics.classification import BinaryAccuracy
from datetime import datetime
from pneumodel import PneuModel


dataset = PneumoDataset()
rng = torch.Generator().manual_seed(42)
train, validation = random_split(dataset, [0.8, 0.2], rng)

parser = argparse.ArgumentParser(description='Pneumothorax Detection')
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--batch_size', default=16, type=int)
args = parser.parse_args()

epochs, lr, batch_size = args.epochs, args.lr, args.batch_size

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=rng)
validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=True, generator=rng)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = PneuModel()
resnet = resnet.to(device)

criterion = BCELoss().to(device)
optimizer = Adam(resnet.parameters(), lr=lr)
metric = BinaryAccuracy().to(device)

for epoch in range(epochs):
    print(f"Epoch {epoch};")
    mean_loss = 0
    resnet.train()
    for _, pair in enumerate(train_loader):
        image, label = pair
        image = image.to(device)
        label = label.to(device)


        optimizer.zero_grad()
        output = resnet(image)

        output = output.squeeze(1).float()
        label = label.float()

        loss = criterion(output, label)
        mean_loss += loss.item()
        loss.backward()
        optimizer.step()
        

    mean_loss = mean_loss/len(train_loader)
    print(f'\tTrain Loss: {mean_loss}')

    resnet.eval()

    metric.reset()
    # print("AFTER RESET", metric.compute())

    with torch.no_grad():
        mean_loss = 0
        for _, pair in enumerate(validation_loader):
            image, label = pair
            image = image.to(device)
            label = label.to(device)
            output = resnet(image)

            output = output.squeeze(1).float()
            label = label.float()

            metric(output, label)

    # mean_loss = mean_loss/len(validation_loader)
    acc = metric.compute()
    print(f'\tValidation Accuracy is: {acc}')


torch.save(resnet, f'./resnet-sigma-{datetime.now()}.pkl')
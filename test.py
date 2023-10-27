import torch
from torch.utils.data import random_split, DataLoader
from dataset import PneumoDataset
# from torcheval.metrics import BinaryAccuracy
from torchmetrics.classification import BinaryAccuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("trained_resnet-sigma.pkl")

dataset = PneumoDataset()
train, validation = random_split(dataset, [0.8, 0.2])

batch_size = 16
validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=True)

# softmax = torch.nn.Softmax()
sigma = torch.nn.Sigmoid()


threshold = 0.5

# metric = BinaryAccuracy(threshold=threshold)
metric = BinaryAccuracy().to(device)


model.eval()
with torch.no_grad():
    mean_loss = 0
    for _, pair in enumerate(validation_loader):
        image, label = pair
        image = image.to(device)
        label = label.to(device)
        output = model(image)

        output = output.squeeze(1).float()
        output = sigma(output)
        
        label = label.float()
        print(output)
        print(label)
        
        metric(output, label)


acc = metric.compute()

print(f"Model accuracy for threshold={threshold} is {acc}")
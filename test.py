import torch
from torch.utils.data import random_split, DataLoader
from dataset import PneumoDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("trained_resnet.pkl")

dataset = PneumoDataset()
train, validation = random_split(dataset, [0.8, 0.2])

batch_size = 16
validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=True)

softmax = torch.nn.Softmax()

model.eval()
with torch.no_grad():
    mean_loss = 0
    for _, pair in enumerate(validation_loader):
        image, label = pair
        image = image.to(device)
        label = label.to(device)
        output = model(image)

        output = output.squeeze(1).float()
        output = softmax(output)
        
        label = label.float()
        print(output)
        print(label)
        

        # mean_loss += loss.item()

# mean_loss = mean_loss/len(validation_loader)
# print(f'Epoch: {epoch}, Validation Loss: {mean_loss}')
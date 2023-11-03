import torch
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load("gen-2023-11-03\ 12:02:23.539974.pkl", map_location=device)
if not torch.cuda.is_available():
    model = model.module
model.eval()
model = model.to(device)

labels = torch.tensor((0, 1))

fake_images = model(labels)

utils.show_tensor_images(fake_images, show="save", name="Hilutho")
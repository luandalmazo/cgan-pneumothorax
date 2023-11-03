import torch
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load("gen.pkl", map_location=device)
if not torch.cuda.is_available():
    model = model.module
model.eval()
model = model.to(device)

labels = torch.tensor((0, 1)).to(device)

fake_images = model(labels)

from dataset import PneumoDataset
pneumodataset = PneumoDataset()

one_image, _ = pneumodataset.__getitem__(3)
one_image = one_image.to(device).unsqueeze(0)
# print(one_image)
# utils.show_tensor_grayscale(one_image, show="save", name="Hilutho")

images = torch.concatenate((fake_images, one_image), dim=0)

utils.show_tensor_grayscale(images, show="save", name="Hilutho")
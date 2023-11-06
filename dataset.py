
from torch.utils.data import Dataset, DataLoader
import os
import pydicom
from torchvision import transforms


augment_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=False),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.1),
])

default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR,antialias=False),
    # transforms.CenterCrop(224),
    # transforms.Normalize(mean=[0.5], std=[0.5])
])

small_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR,antialias=False),
    # transforms.CenterCrop(224),
    # transforms.Normalize(mean=[0.5], std=[0.5])
])


class PneumoDataset(Dataset):
    def __init__(self, transform=default_transform):
        self.transform = transform
        self.dir_no_pneumothorax= "./siim_small/train/No Pneumothorax/"
        self.dir_pneumothorax= "./siim_small/train/Pneumothorax/"
        self.list_no_pneumothorax = sorted(os.listdir(self.dir_no_pneumothorax))
        self.list_pneumothorax = sorted(os.listdir(self.dir_pneumothorax))

        self.list_no_pneumothorax = [(0, os.path.join(self.dir_no_pneumothorax, file)) for file in self.list_no_pneumothorax]
        self.list_pneumothorax = [(1, os.path.join(self.dir_pneumothorax, file)) for file in self.list_pneumothorax]
            
        self.dataset = self.list_no_pneumothorax+self.list_pneumothorax


    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        
        label, dicom_file = self.dataset[index]
        image = pydicom.dcmread(dicom_file)
        image_pixel_data = image.pixel_array
        image_pixel_data = self.transform(image_pixel_data)
        return image_pixel_data, float(label)



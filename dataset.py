
from torch.utils.data import Dataset, DataLoader
import os
import pydicom
from torchvision import transforms
import pandas as pd
import glob


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


class SmallPneumoDataset(Dataset):
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


class PneumoDataset(Dataset):
    def __init__(self, transform=default_transform, train=True):
        self.transform = transform

        
        self.dir = "./dataset_full/dicom-images-train/*/*/*.dcm" if train else "./dataset_full/dicom-images-test/*/*/*.dcm"
        # self.dir = "./dataset_full/dicom-images-train/" if train else "./dataset_full/dicom-images-test/"

        self.list_dir = sorted(glob.glob(self.dir))
        # print("LIST DIR TYPE IS", type(self.list_dir))
            
        self.dataset = pd.read_csv('./dataset_full/train-rle.csv', delimiter="," )

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):

        dicom_basename = self.dataset.iloc[index][0]
        # print(dicom_basename)

        label = self.dataset.iloc[index][1]

        matching_files = [file for file in self.list_dir if dicom_basename in file]

        try:
            dicom_file = matching_files[0]
        except IndexError:
            raise Exception("BRUH MOMENTO DATASET IS BUILT DIFFERENT")

        # dicom_file = self.list_dir[index]
        # dicom_basename = os.path.basename(dicom_file)
        # dicom_basename = os.path.splitext(dicom_basename)[0]
        # # search for dicom basename
        # query = self.dataset[self.dataset["ImageId"] == dicom_basename] 
        # # return type is a one-row dataframe. PixelEncondings is column index 1.
        # label = query.iloc[0][1].strip()

        image = pydicom.dcmread(dicom_file)
        image_pixel_data = image.pixel_array
        image_pixel_data = self.transform(image_pixel_data)

        label = 0 if label == '-1' else 1
        
        return image_pixel_data, float(label)
        
if __name__ == '__main__':
    dataset = PneumoDataset()
    print(dataset.__getitem__(4))
    # print(dataset.list_dir)
        
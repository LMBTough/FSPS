import os
from PIL import Image
from torch.utils import data
import pandas as pd
from torchvision import transforms as T


class ImageNet(data.Dataset):
    

    def __init__(self, dir, csv_path, transforms=None, num_images=None):
        self.dir = dir   
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms
        self.num_images = num_images


    def __getitem__(self, index):
        if self.num_images is not None and index >= self.num_images:
            raise StopIteration
        img_obj = self.csv.loc[index]
        ImageID = img_obj['ImageId'] + '.png'
        # ImageID = img_obj['ImageId'] + '.jpg'
        Truelabel = img_obj['TrueLabel'] - 1
        TargetClass = img_obj['TargetClass'] - 1
        img_path = os.path.join(self.dir, ImageID)
        pil_img = Image.open(img_path).convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)
        return data, ImageID, Truelabel


    def __len__(self):
        if self.num_images is None:
            return len(self.csv)
        else:
            return min(len(self.csv), self.num_images)









"""Simple Pytorch Dataloader.
Input image csv file, transforms

"""

# - dataloader.py - #

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomCrop, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.utils.data.dataset import Dataset
import numpy as np
# ------------------------
#  Datasets
# ------------------------
class LGDataSet(Dataset):
    
    def __init__(self, data, transform = None, is_test=False):
        
        self.data = data
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self,idx):
        file_name_input = self.data.iloc[idx]['input_img']
        file_name_target = self.data.iloc[idx]['label_img']
        
        if self.is_test:
            images = np.load(file_name_input)
        else:
            images = np.load(file_name_input)
            targets = np.load(file_name_target)
         
        
        if self.transform:
            transformed = self.transform(image=images, mask=targets)
            images = transformed["image"]/255.
            targets = transformed["mask"]/255.
            
        if self.is_test:
            return images
        else:
            return images, targets

# ------------------------
#  Augmentation
# ------------------------
def get_transforms(*, data, img_size):
    """make transforms
    2 cases : train, valid/test

    """
    if data == 'train':
        return A.Compose([
                    A.HorizontalFlip(p=0.3), 
                    A.VerticalFlip(p=0.3), 

                 A.OneOf([
                     A.Cutout(always_apply=False, p=1.0, num_holes=8, max_h_size=img_size//32, max_w_size=img_size//32)
                    ], p=0.3),

            ToTensorV2(transpose_mask=False)
        ],p=1.)
    elif data == 'valid':
        return A.Compose([

            ToTensorV2(transpose_mask=False)
        ],p=1.)
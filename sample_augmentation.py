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
import pandas as pd
from torch.utils.data import DataLoader
import os, cv2
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
            images2 = transformed["image"]/255.
            
        if self.is_test:
            return images
        else:
            return images, images2

# ------------------------
#  Augmentation
# ------------------------
def get_transforms(*, data, img_size):
    if data == 'train':
        return A.Compose([
                    A.HorizontalFlip(p=0.3), 
                    A.VerticalFlip(p=0.3), 

                 A.OneOf([
                     A.Cutout(always_apply=False, p=1.0, num_holes=8, max_h_size=img_size//32, max_w_size=img_size//32)
                    ], p=1),

            ToTensorV2(transpose_mask=False)
        ],p=1.)
    elif data == 'valid':
        return A.Compose([

            ToTensorV2(transpose_mask=False)
        ],p=1.)


if __name__ == '__main__':
    train_transform = get_transforms(data='train', img_size=768)
    df = pd.read_csv(f'./data/preprocess_train_768.csv')
    train_data = df[df['type_']=='train'].reset_index(drop=True).copy()
    train_data = train_data.sample(5)

    ## dataset ------------------------------------
    train_dataset = LGDataSet(data = train_data, transform = train_transform)
    trainloader = DataLoader(dataset=train_dataset, batch_size=5,
                                num_workers=8, shuffle=True, pin_memory=True)
    
    # make augmented images
    for img,aug_img in trainloader:
        break
    img = (img).cpu().detach().numpy().astype('uint8')
    aug_img = (aug_img*255).permute(0,2,3,1).cpu().detach().numpy().astype('uint8')

    os.makedirs(f'./img/', exist_ok=True)
    for i in range(len(aug_img)):
        cv2.imwrite(f'./img/augmented_img_{i}.png', aug_img[i])
        cv2.imwrite(f'./img/original_img_{i}.png', img[i])


    

    
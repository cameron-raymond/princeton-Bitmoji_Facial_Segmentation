# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Bitmoji Facial Segmentation Model
# 
# [Reference](https://github.com/sam-watts/futoshiki-solver/blob/master/puzzle_segmentation/semantic_seg.ipynb)

# %%
from torchvision import models
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import json
import segmentation_models_pytorch as smp
import torch
import pandas as pd
import numpy as np
import os
import cv2
import shutil
import functools


# %%
class BitmojiDataset(Dataset):
    """Bitmoji dataset."""

    def __init__(self, csv_file, root_dir, resize=None, transform=None, augmentation=None, copies=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.bitmoji_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.resize = resize
        self.transform = transform
        self.augmentation = augmentation
        self.copies = copies

    def __len__(self):
        if self.copies:
            return self.bitmoji_frame.shape[0] * self.copies
        else:
            return self.bitmoji_frame.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if self.copies:
            idx = idx % self.bitmoji_frame.shape[0]

        img_name = os.path.join(self.root_dir,
                                self.bitmoji_frame.iloc[idx, 0])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        shape = image.shape
                
        bitmoji = self.bitmoji_frame.loc[idx, 'region_shape_attributes']
        mask = get_mask(bitmoji, shape[:2])
        mask = mask / 255 # normalise
        # mask = np.stack(masks, axis=-1).astype('float')
        
        if self.resize is not None:
            image, mask = cv2.resize(image, (320, 480)), cv2.resize(mask, (320, 480))
            mask = mask.reshape((*mask.shape, 1))
            
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        return image, mask

def get_mask(row, shape):
    row = json.loads(row)
    mask = np.zeros((*shape, 1))
    try:
        coords = np.array([[x,y] for x, y in zip(row['all_points_x'], row['all_points_y'])])
        cv2.fillPoly(mask, [coords], 255)
        
    except:
        pass
    return mask

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 9))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title(), color='white')
        plt.imshow(image)
    plt.show()


# %%
y_train_fp = '../data/train-bitmoji_annotation.csv'
x_train_fp = '../data/images'
bitmoji_dataset = BitmojiDataset(y_train_fp,
                                 x_train_fp,
                                 True)

# for i in range(5):
#     out = bitmoji_dataset[i+123]
#     visualize(
#         image = out[0],
#         mask = out[1].squeeze()
#     )


# %%
import albumentations as albu


# %%
def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=20, shift_limit=0.2, p=0.8, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.GaussNoise(p=0.1),
        albu.Perspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        )
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(320, 320)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# %%
augmented_dataset = BitmojiDataset(y_train_fp,
                                   x_train_fp,
                                   resize=True,
                                   augmentation=get_training_augmentation())

# same image with different random transforms
# for i in range(3):
#     image, mask = augmented_dataset[i]
#     visualize(image=image, mask=mask.squeeze())


# %%
import gc; gc.collect()


# %%
ENCODER = 'efficientnet-b3'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['face']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = torch.device('cpu')


# %%
# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
).to(DEVICE)


# %%
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


# %%
train_dataset = BitmojiDataset(y_train_fp,
                               x_train_fp,
                               True,
                               get_preprocessing(preprocessing_fn), 
                               get_training_augmentation(), 
                               copies=2)
                                     
print('Number of training samples:', len(train_dataset))
y_valid_fp,x_valid_fp = '../data/test-bitmoji_annotation.csv',x_train_fp
valid_dataset = BitmojiDataset(y_valid_fp, 
                               x_valid_fp, 
                               True,
                               get_preprocessing(preprocessing_fn),
                               get_validation_augmentation(), 
                               copies=1)


# %%
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=0)


# %%
loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0005),
])


# %%
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)


# %%
max_score = 0

for i in range(0, 40):
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, '../models/best_model_3.pth')
        print('Model saved!')
        
    if i == 15:
        new_lr = 5e-5
        optimizer.param_groups[0]['lr'] = new_lr
        print(f'Decrease decoder learning rate to {new_lr}!')


# %%




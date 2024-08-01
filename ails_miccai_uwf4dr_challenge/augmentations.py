from torchvision.transforms import v2
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# use this augmentation pipeline in the case of:
# 1. training
# 2. both datasets are included (therefore: resizing or cropping)
""" rotate_affine_flip_choice = v2.Compose([
    v2.ToPILImage(),
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.3, hue=0.3),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomRotation(degrees=15, expand=True),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    v2.Resize(size=(800, 1016), antialias=True)
])

# use this augmentation pipeline in the case of:
# validation with both datasets included (therefore: resizing)
resize_only = v2.Compose([
    v2.ToPILImage(),
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(size=(512, 512), antialias=True)
]) """

#from bertl
"""rotate_affine_flip_choice = v2.Compose([
    v2.ToPILImage(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    #v2.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.3, hue=0.3),
    v2.RandomHorizontalFlip(),
    #v2.RandomVerticalFlip(),
    v2.RandomRotation(degrees=15, expand=True),
    #v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    v2.Resize(size=(800, 1016), antialias=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])"""

transforms_train = A.Compose([
        A.Resize(800, 1016, p=1),
        MultiplyMask(p=1),
        ResidualGaussBlur(p=cfg.p_gaussblur),
        A.Equalize(p=cfg.p_equalize),
        A.CLAHE(clip_limit=5., p=cfg.p_clahe),
        A.HorizontalFlip(p=cfg.p_horizontalflip),
        A.Affine(rotate=cfg.rotation, rotate_method='ellipse', p=cfg.p_affine),
        A.Normalize(mean=[0.406, 0.485, 0.456], std=[0.225, 0.229, 0.224], p=1),
        #A.Resize(770, 1022, p=1), # comment whenever not using DinoV2
        ToTensorV2(p=1)
    ])
"""resize_only = v2.Compose([
    v2.ToPILImage(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    #v2.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.3, hue=0.3),
    #v2.RandomHorizontalFlip(),
    #v2.RandomVerticalFlip(),
    #v2.RandomRotation(degrees=15, expand=True),
    #v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    v2.Resize(size=(800, 1016), antialias=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])"""

transforms_val = A.Compose([
            A.Resize(800, 1016, p=1),
            MultiplyMask(p=1),
            A.Normalize(mean=[0.406, 0.485, 0.456], std=[0.225, 0.229, 0.224], p=1),
            #A.Resize(770, 1022, p=1), # comment whenever not using DinoV2
            ToTensorV2(p=1)
        ])
from torchvision.transforms import v2
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from ails_miccai_uwf4dr_challenge.preprocess_augmentations import ResidualGaussBlur, MultiplyMask


# use this augmentation pipeline in the case of:
# 1. training
# 2. both datasets are included (therefore: resizing or cropping)
"""transforms_train = v2.Compose([
    v2.ToPILImage(),
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.3, hue=0.3),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomRotation(degrees=15, expand=True),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    v2.Resize(size=(800, 1016), antialias=True)
])"""
"""
# use this augmentation pipeline in the case of:
# validation with both datasets included (therefore: resizing)
resize_only = v2.Compose([
    v2.ToPILImage(),
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(size=(512, 512), antialias=True)
]) """

"""#from bertl
transforms_train = v2.Compose([
    v2.ToPILImage(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    #v2.Grayscale(num_output_channels=3),
    #v2.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.3, hue=0.3),
    v2.RandomHorizontalFlip(),
    #v2.RandomVerticalFlip(),
    v2.RandomRotation(degrees=15, expand=True),
    #v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    v2.Resize(size=(800, 1016), antialias=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])"""

transforms_train = A.Compose([
        A.Resize(800, 1016),
        #MultiplyMask(),
        #ResidualGaussBlur(),
        A.Equalize(),
        #A.ToGray(),
        #A.CLAHE(clip_limit=5.),
        A.HorizontalFlip(),
        A.Affine(rotate_method='ellipse'),
        A.Normalize(mean=[0.406, 0.485, 0.456], std=[0.225, 0.229, 0.224]),
        #A.Resize(770, 1022, p=1), # comment whenever not using DinoV2
        ToTensorV2()
    ])


"""transforms_val = v2.Compose([
    v2.ToPILImage(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    #v2.Grayscale(num_output_channels=3),
    #v2.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.3, hue=0.3),
    #v2.RandomHorizontalFlip(),
    #v2.RandomVerticalFlip(),
    #v2.RandomRotation(degrees=15, expand=True),
    #v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    v2.Resize(size=(800, 1016), antialias=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])"""

transforms_val = A.Compose([
            A.Resize(800, 1016),
            MultiplyMask(),
            #A.ToGray(),
            A.Normalize(mean=[0.406, 0.485, 0.456], std=[0.225, 0.229, 0.224]),
            #A.Resize(770, 1022, p=1), # comment whenever not using DinoV2
            ToTensorV2()])
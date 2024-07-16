import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from ails_miccai_uwf4dr_challenge.config import PROJ_ROOT
from pathlib import Path
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2
import matplotlib.pyplot as plt

ellipse = cv2.ellipse(np.zeros((800, 1016), dtype=np.uint8), (525, 400), (480, 380), 0, 0, 360, 1, -1) # ellipse mask made from hand-drawn mask
MASK = np.array([ellipse, ellipse, ellipse], dtype=np.uint8).transpose(1, 2, 0)



class ResidualGaussBlur(ImageOnlyTransform):
    """
    """
    def __init__(self, p=1) -> None:
        super(ResidualGaussBlur, self).__init__()
        self.p = p

    def apply(self, img, **params):

        if np.random.uniform(0, 1) > self.p:
            return img
        
        if img.shape[2] != 3:
            img = img.transpose(1, 2, 0)

        img = img + A.GaussNoise(p=1)(image=img)["image"]
        
        return img


class MultiplyMask(ImageOnlyTransform):
    """
    Masks out the portion of UWF images that is !!ON AVERAGE!! covered by the device (and thus is noise).
    Research has shown that this can improve the performance of the model.

    https://www.sciencedirect.com/science/article/pii/S0010482523002159?via%3Dihub#fig2
    &
    https://ieeexplore.ieee.org/document/9305690

    """
    def __init__(self, p=1) -> None:
        super(MultiplyMask, self).__init__()
        self.p = p

        self.mask = MASK
        self.cropper = lambda img: img[400-380:400+380, 525-480:525+480]
        
    def apply(self, img, **params):

        if np.random.uniform(0, 1) > self.p:
            return img
        
        if img.shape[2] != 3:
            img = img.transpose(1, 2, 0)

        img = img * self.mask
        img = self.cropper(img)
        
        return img


preprocessing = A.Compose([
        A.Resize(800, 1016, p=1),
        ResidualGaussBlur(p=1),
        MultiplyMask(p=1),
        A.Resize(800, 1016, p=1),
        #A.Equalize(p=0.1),
        A.CLAHE(clip_limit=5., p=0.3)
    ])

augment_train = A.Compose([
        A.VerticalFlip(p=0.5),
        #A.HorizontalFlip(p=0.5),
        #A.Affine(rotate=15, rotate_method='ellipse', p=0.5),
        ToTensorV2(p=1)
    ])

augment_val = A.Compose([
        ToTensorV2(p=1)
    ])


transforms_train = A.Compose([
    preprocessing,
    augment_train
])

transforms_val = A.Compose([
    preprocessing,
    augment_val
])

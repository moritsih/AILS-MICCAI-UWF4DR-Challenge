import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from ails_miccai_uwf4dr_challenge.config import PROJ_ROOT
from pathlib import Path
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2

# Load mask
MASK = np.load(PROJ_ROOT / "preprocessing" / "mask.npy")


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
    def __init__(self, safe_db_lists=[], p=1) -> None:
        super(MultiplyMask, self).__init__()
        self.safe_db_lists = safe_db_lists
        self.p = p

        self.mask = MASK

    def apply(self, img, **params):

        if np.random.uniform(0, 1) > self.p:
            return img
        
        if img.shape[2] != 3:
            img = img.transpose(1, 2, 0)

        img = img * self.mask
        
        return img
    

class CropBlackBorders(ImageOnlyTransform):
    """
    """
    def __init__(self, p=1) -> None:
        super(CropBlackBorders, self).__init__()
        self.p = p

    def apply(self, img, **params):

        if np.random.uniform(0, 1) > self.p:
            return img
        
        if img.shape[2] != 3:
            img = img.transpose(1, 2, 0)

        img = self.cropper(img)
        
        return img
    
    def cropper(self, img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        crop = img[y:y+h,x:x+w]
        
        return crop


preprocessing = A.Compose([
        A.Resize(800, 1016, p=1),
        ResidualGaussBlur(p=1),
        MultiplyMask(p=1),
        CropBlackBorders(p=1),
        A.Resize(800, 1016, p=1),
        A.Equalize(p=1),
        A.CLAHE(clip_limit=5., p=1)
    ])

augment_train = A.Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
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

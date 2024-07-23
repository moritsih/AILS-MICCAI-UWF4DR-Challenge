import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from ails_miccai_uwf4dr_challenge.config import PROJ_ROOT
from pathlib import Path
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2
import matplotlib.pyplot as plt
from skimage import restoration
import torch


class GreenChannelEnhancement(ImageOnlyTransform):

    def __init__(self, p=1) -> None:
        super(GreenChannelEnhancement, self).__init__()
        self.p = p
        
    def apply(self, img, **params):

        if np.random.uniform(0, 1) > self.p:
            return img
        
        # Convert to numpy array if it's a tensor
        if isinstance(img, torch.Tensor):
            img = img.numpy().transpose((1, 2, 0))

        # Ensure the image is in the correct format
        img = img.astype(np.float32)

        # Separate the channels
        r, g, b = cv2.split(img)

        # Apply Wiener filter to the green channel
        psf = np.ones((5, 5)) / 25
        g_filtered = restoration.wiener(g, psf, balance=0.1)

        # Apply CLAHE to the filtered green channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g_enhanced = clahe.apply((g_filtered * 255).astype(np.uint8))
        g_enhanced = g_enhanced / 255.0  # Normalize back to range [0, 1]

        # Ensure all channels are the same type
        r = r.astype(np.float32)
        g_enhanced = g_enhanced.astype(np.float32)
        b = b.astype(np.float32)

        # Merge the enhanced green channel back with the original red and blue channels
        enhanced_img = cv2.merge((r, g_enhanced, b))

        # Convert back to tensor
        enhanced_img = torch.from_numpy(enhanced_img.transpose((2, 0, 1)))
        return enhanced_img
        


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


ellipse = cv2.ellipse(np.zeros((800, 1016), dtype=np.uint8), (525, 400), (480, 380), 0, 0, 360, 1, -1) # ellipse mask made from hand-drawn mask
MASK = np.array([ellipse, ellipse, ellipse], dtype=np.uint8).transpose(1, 2, 0)


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


transforms_train = A.Compose([
    A.Resize(800, 1016, p=1),
    MultiplyMask(p=1),
    ResidualGaussBlur(p=3),
    A.Equalize(p=.1),
    A.CLAHE(clip_limit=5., p=.3),
    A.HorizontalFlip(p=.3),
    A.Affine(rotate=15, rotate_method='ellipse', p=.3),
    A.Normalize(mean=[0.406, 0.485, 0.456], std=[0.225, 0.229, 0.224], p=1),
    ToTensorV2(p=1)
])

transforms_val = A.Compose([
        A.Resize(800, 1016, p=1),
        MultiplyMask(p=1),
        A.Normalize(mean=[0.406, 0.485, 0.456], std=[0.225, 0.229, 0.224], p=1),
        ToTensorV2(p=1)
    ])

def main():
    from ails_miccai_uwf4dr_challenge.dataset_strategy import CustomDataset, DatasetStrategy, CombinedDatasetStrategy, \
    Task1Strategy, Task2Strategy, Task3Strategy, TrainValSplitStrategy, RandomOverSamplingStrategy, DatasetBuilder

    dataset_strategy = CombinedDatasetStrategy()
    task_strategy = Task2Strategy()
    
    split_strategy = TrainValSplitStrategy(split_ratio=0.8)
    resampling_strategy = RandomOverSamplingStrategy()

    builder = DatasetBuilder(dataset_strategy, task_strategy, split_strategy, resampling_strategy)
    train_data, val_data = builder.build()

    train_dataset = CustomDataset(train_data, transform=transforms_train)
    val_dataset = CustomDataset(val_data, transform=transforms_val)

    img, label = train_dataset[0]

    plt.imshow(img.permute(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    main()

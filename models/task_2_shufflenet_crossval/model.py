import os
import torch
from torch import nn
from pathlib import Path
import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def remove_prefix(state_dict, prefix):
    """
    Remove the prefix from state_dict keys.
    """
    return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}


class model:
    def __init__(self):
        
        # The model is evaluated using CPU, please do not change to GPU to avoid error reporting.
        self.device = torch.device("cpu")
        self.model = None

    def init(self):
        pass  # nothing to do here

    def load(self, dir_path):
        """
        load the model and weights.
        dir_path is a string for internal use only - do not remove it.
        all other paths should only contain the file name, these paths must be
        concatenated with dir_path, for example: os.path.join(dir_path, filename).
        :param dir_path: path to the submission directory (for internal use only).
        :return:
        """
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True)

        net_fl = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1)
        )

        self.model.fc = net_fl

        # join paths
        checkpoints = list(Path(dir_path).glob('*.pth'))
        checkpoint_paths = [os.path.join(dir_path, self.checkpoint) for self.checkpoint in checkpoints]

        state_dicts = [torch.load(checkpoint_path, map_location=self.device) for checkpoint_path in checkpoint_paths] 
        state_dicts = [remove_prefix(state_dict, 'model.') for state_dict in state_dicts] # we need to remove the prefix as on training EfficientNet was wrapped

        def make_model(state_dict):
            model = self.model
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            return model

        self.models = [make_model(state_dict) for state_dict in state_dicts]


    def predict(self, input_image):
        """
        perform the prediction given an image.
        input_image is a ndarray read using cv2.imread(path_to_image, 1).

        !! Note that the order of the three channels of the input_image read by cv2.imread is BGR. This is the way we use to read the image.
        !! If you use Image.open() from PIL in your training process, the order of the three channels will be RGB. 
        Please pay attention to this difference.

        :param input_image: the input image to the model.
        :return: a float value indicating the probability of class 1.
        """
        # apply the same transformations as during validation
        transform = A.Compose([
            A.Resize(800, 1016, p=1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1),
            #MultiplyMask(p=1),
            ToTensorV2(p=1)
        ])
        
        image = transform(image=input_image)['image']
        image = image.unsqueeze(0)  # Add batch dimension
        image = image.to(self.device)

        with torch.no_grad():
            output = torch.stack([torch.sigmoid(model(image)).squeeze(0) for model in self.models]).mean()

        class_1_prob = output.item()  # Convert to float

        return float(class_1_prob)



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
        self.cropper = lambda img: img[400 - 380:400 + 380, 525 - 480:525 + 480]

    def apply(self, img, **params):

        if np.random.uniform(0, 1) > self.p:
            return img

        if img.shape[2] != 3:
            img = img.transpose(1, 2, 0)

        img = img * self.mask
        img = self.cropper(img)

        return img


def remove_prefix_in_state_dict(state_dict, prefix):
    """
    Remove the prefix from state_dict keys.
    """
    return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}


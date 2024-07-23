import torch.nn as nn
import os
import torch
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
import cv2
import numpy as np
from skimage import restoration

def remove_prefix(state_dict, prefix):
    """
    Remove the prefix from state_dict keys.
    """
    return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
class model:
    def __init__(self):
        self.checkpoints = [
            "EfficientNetB0_best_weights_fold1_2024-07-22_10-22-01.pth",
            "EfficientNetB0_best_weights_fold2_2024-07-22_10-22-01.pth",
            "EfficientNetB0_best_weights_fold3_2024-07-22_10-22-01.pth",
            "EfficientNetB0_best_weights_fold4_2024-07-22_10-22-01.pth",
            "EfficientNetB0_best_weights_fold5_2024-07-22_10-22-01.pth"
        ]
        self.device = torch.device("cpu")
        self.models = []

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
        for checkpoint in self.checkpoints:
            model = EfficientNet.from_pretrained('efficientnet-b0')
            in_features = model._fc.in_features
            model._fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, 64),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(64, 1)
            )
            checkpoint_path = os.path.join(dir_path, checkpoint)
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            state_dict = remove_prefix(state_dict, 'model.')  # Remove prefix as on training EfficientNet was wrapped
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.models.append(model)

    def predict(self, input_image):
        """
        perform the prediction given an image.
        input_image is a ndarray read using cv2.imread(path_to_image, 1).

        !! Note that the order of the three channels of the input_image read by cv2.imread is BGR. This is the way we use to read the image.
        !! If you use Image.open() from PIL in your training process, the order of the three channels will be RGB. Please pay attention to this difference.

        :param input_image: the input image to the model.
        :return: a float value indicating the probability of class 1.
        """
        # apply the same transformations as during validation
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            GreenChannelEnhancement(),  # Apply Wiener filter and CLAHE
            transforms.Resize(size=(448, 448)),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # RGB
            #transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]) # BGR
        ])

        image = transform(input_image)
        image = image.unsqueeze(0)  # Add batch dimension
        image = image.to(self.device)

        # Aggregate predictions from all models
        with torch.no_grad():
            outputs = [torch.sigmoid(model(image)) for model in self.models]
            avg_output = torch.mean(torch.stack(outputs), dim=0)

        class_1_prob = avg_output.item()  # Convert to float

        return float(class_1_prob)


class GreenChannelEnhancement:
    def __call__(self, img):
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
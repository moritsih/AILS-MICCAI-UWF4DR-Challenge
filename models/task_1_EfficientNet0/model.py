import os
import torch
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
import cv2
import numpy as np
from skimage import restoration
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
        self.checkpoint = "#checkpoint_file_path#"  # The checkpoint file path will be replaced in the copied model file - see SubmissionBuilder#CHECK_POINT_FILE_PATH_PLACEHOLDER
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
        self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1)
        # join paths
        checkpoint_path = os.path.join(dir_path, self.checkpoint)

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        state_dict = remove_prefix(state_dict, 'model.') # we need to remove the prefix as on training EfficientNet was wrapped

        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

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
        """transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),  # Convert to float32 tensor and scale
            #GreenChannelEnhancement(),  # Apply Wiener filter and CLAHE
            transforms.Resize(size=(400, 508)),
            transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])"""
        transform = A.Compose([
            A.Resize(800, 1016),
            #MultiplyMask(),
            #A.ToGray(p=1),
            #GreenChannelEnhancement,
            A.Equalize(),
            #A.Resize(770, 1022, p=1), # comment whenever not using DinoV2,
            A.Normalize(mean=[0.406, 0.485, 0.456], std=[0.225, 0.229, 0.224]),
            ToTensorV2()])

        #image = transform(input_image)
        image = transform(image=input_image)['image']

        image = image.unsqueeze(0)  # Add batch dimension
        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image)
            prob = torch.sigmoid(output).squeeze(0)  # Using sigmoid for binary classification

        class_1_prob = prob.item()  # Convert to float

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
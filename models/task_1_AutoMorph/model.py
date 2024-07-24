import os
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision import transforms


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
        self.model = build_automorph_model()
        # join paths
        checkpoint_path = os.path.join(dir_path, self.checkpoint)

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        state_dict = remove_prefix_in_state_dict(state_dict,'model.')  # we need to remove the prefix as on training EfficientNet was wrapped

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
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(800, 1016)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # RGB
            transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])  # BGR
        ])

        image = transform(input_image)
        image = image.unsqueeze(0)  # Add batch dimension
        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image)
            prob = torch.sigmoid(output).squeeze(0)  # Using sigmoid for binary classification

        class_1_prob = prob.item()  # Convert to float

        return float(class_1_prob)

def build_automorph_model():
    # code taken from https://github.com/rmaphoh/AutoMorph/blob/main/M1_Retinal_Image_quality_EyePACS/model.py
    model = EfficientNet.from_pretrained('efficientnet-b4')
    model._fc = nn.Identity()
    net_fl = nn.Sequential(
        nn.Linear(1792, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(64, 3)
    )
    model._fc = net_fl
    # add a final layer that outputs single value
    model._fc.add_module("7", nn.Linear(3, 1))
    return model


def remove_prefix_in_state_dict(state_dict, prefix):
    """
    Remove the prefix from state_dict keys.
    """
    return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}

import os
import torch
from torchvision.models import efficientnet_v2_s
from torchvision import transforms
import torch.nn as nn
from torchvision.transforms import v2

def remove_prefix(state_dict, prefix):
    """
    Remove the prefix from state_dict keys.
    """
    return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
class model:
    def __init__(self):
        self.checkpoint = "EffNetV2_S_last_checkpoint.pth"
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
        # Get the EfficientNetV2 model
        self.model = efficientnet_v2_s(weights="IMAGENET1K_V1")

        # Replace the entire classifier block
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1)
        )

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
        transform = transforms.Compose([
            v2.ToPILImage(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            GreenChannelEnhancement(),
        ])

        image = transform(input_image)
        image = image.unsqueeze(0)  # Add batch dimension
        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image)
            prob = torch.sigmoid(output).squeeze(0)  # Using sigmoid for binary classification

        class_1_prob = prob.item()  # Convert to float

        return float(class_1_prob)

import os
import cv2
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet


class model:
    def __init__(self):
        self.checkpoint = "best_model_2024-07-10_15-43-20.pth"
        # The model is evaluated using CPU, please do not change to GPU to avoid error reporting.
        self.device = torch.device("cpu")

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
        self.model = Task1EfficientNetB4()
        # join paths
        checkpoint_path = os.path.join(dir_path, self.checkpoint)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
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
        image = cv2.resize(input_image, (512, 512))
        image = image / 255
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image = image.to(self.device, torch.float)

        with torch.no_grad():
            output = self.model(image)
            prob = torch.softmax(output, dim=1).squeeze(0)

        class_1_prob = prob[1]
        class_1_prob = class_1_prob.detach().cpu()

        return float(class_1_prob)


class Task1EfficientNetB4(nn.Module):
    def __init__(self, learning_rate=1e-3):
        super(Task1EfficientNetB4, self).__init__()

        self.learning_rate = learning_rate

        # Get model and replace the last layer
        self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=1)

        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, x):
        return self.model(x)


import os
import cv2
import torch
from torch import nn
import torchvision.models as models


class model:
    def __init__(self):
        self.checkpoint = "model_weights.pth"
        # The model is evaluated using CPU, please do not change to GPU to avoid error reporting.
        self.device = torch.device("cpu")

    def load(self, dir_path, arch):
        """
        load the model and weights.
        dir_path is a string for internal use only - do not remove it.
        all other paths should only contain the file name, these paths must be
        concatenated with dir_path, for example: os.path.join(dir_path, filename).
        :param dir_path: path to the submission directory (for internal use only).
        :return:
        """
        if 'MobileNetV3Small' in arch:
            self.model = MobileNetV3Small(num_classes=2)
        elif 'SqueezeNet' in arch:
            self.model = SqueezeNet1_1(num_classes=2)
        elif 'resnet' in arch:
            self.model = ResNet(arch=arch, num_classes=2)
        else:
            raise ValueError

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


class ResNet(nn.Module):
    def __init__(self, arch, num_classes=2, pretrained=True):
        super(ResNet, self).__init__()
        if arch == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
        elif arch == 'resnet34':
            self.resnet = models.resnet34(pretrained=pretrained)
        elif arch == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
        elif arch == 'resnet101':
            self.resnet = models.resnet101(pretrained=pretrained)
        elif arch == 'resnet152':
            self.resnet = models.resnet152(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features=num_features, out_features=num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(MobileNetV3Small, self).__init__()
        self.mobilenet_v3_small = models.mobilenet_v3_small(pretrained=pretrained)
        num_features = self.mobilenet_v3_small.classifier[3].in_features
        self.mobilenet_v3_small.classifier[3] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.mobilenet_v3_small(x)


class SqueezeNet1_1(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(SqueezeNet1_1, self).__init__()
        self.squeezenet = models.squeezenet1_1(pretrained=pretrained)
        self.squeezenet.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.num_classes = num_classes

    def forward(self, x):
        return self.squeezenet(x)

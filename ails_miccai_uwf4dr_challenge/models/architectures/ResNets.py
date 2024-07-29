import torch
import torch.nn as nn
import enum

class ResNetVariant(enum.Enum):
    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    RESNET50 = "resnet50"

class ResNet(nn.Module):
    def __init__(self, 
                 model_variant: ResNetVariant = ResNetVariant.RESNET18, 
                 num_classes: int = 1,
                 pretrained: bool =True):
        
        super(ResNet, self).__init__()

        if model_variant == ResNetVariant.RESNET18:
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)
        elif model_variant == ResNetVariant.RESNET34:
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=pretrained)
        elif model_variant == ResNetVariant.RESNET50:
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)
        else:
            raise ValueError(f"Invalid model variant: {model_variant}")
        
        model_name = model_variant.value

        # Change the output layer to have num_classes output features
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        print(f"Number of output features in {model_name} encoder: ", self.model.fc.out_features)
        

    def forward(self, x):
        return self.model(x)


def main():

    model = ResNet()
    print(model)


if __name__ == "__main__":
    main()

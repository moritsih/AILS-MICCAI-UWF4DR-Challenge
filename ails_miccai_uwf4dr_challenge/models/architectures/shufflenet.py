import torch
import torch.nn as nn

class ShuffleNet(nn.Module):
    def __init__(self, 
                 num_classes: int = 1,
                 pretrained: bool = True):
        
        super(ShuffleNet, self).__init__()

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=pretrained)

        # Change the output layer to have num_classes output features
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        print(f"Number of output features in ShuffleNet encoder: ", self.model.fc.out_features)
        

    def forward(self, x):
        return self.model(x)


def main():

    model = ShuffleNet()
    print(model)


if __name__ == "__main__":
    main()

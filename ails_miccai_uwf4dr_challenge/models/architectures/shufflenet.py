import torch
import torch.nn as nn

class ShuffleNet(nn.Module):
    def __init__(self, 
                 num_classes: int = 1,
                 pretrained: bool = True):
        
        super(ShuffleNet, self).__init__()

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=pretrained)

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
            nn.Linear(64, num_classes)
        )

        self.model.fc = net_fl
        

    def forward(self, x):
        return self.model(x)


def main():

    model = ShuffleNet()
    print(model)


if __name__ == "__main__":
    main()

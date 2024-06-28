from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

class ResNet18:
    def __init__(self, device='cpu'):
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.device = device

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 1)

        self.model.to(self.device)
    
    def forward(self, x):
        return self.model(x)
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()

    
if __name__ == '__main__':
    model = ResNet18()
    print(model.model)
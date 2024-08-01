import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from pathlib import Path
from torchvision.models import resnet50, ResNet50_Weights

class Task1Resnet50(nn.Module):
    def __init__(self, learning_rate=1e-3):
        super(Task1Resnet50, self).__init__()

        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 3),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(3, 1)
        )

        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the last layer
        for param in self.model.fc.parameters():
            param.requires_grad = True

        self.learning_rate = learning_rate
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        with torch.no_grad():
            pred = torch.sigmoid(self(x))
        return pred

def main():
    # Initialize model, criterion, optimizer
    model = Task1Resnet50()
    print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    x = torch.randn(1, 3, 512, 512).to(device)
    y_hat = model.predict(x).item()
    print(y_hat)

if __name__ == "__main__":
    main()
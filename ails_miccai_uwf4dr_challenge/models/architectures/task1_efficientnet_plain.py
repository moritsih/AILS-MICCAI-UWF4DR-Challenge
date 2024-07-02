import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from pathlib import Path

class Task1EfficientNetB4(nn.Module):
    def __init__(self, learning_rate=1e-3):
        super(Task1EfficientNetB4, self).__init__()

        self.learning_rate = learning_rate

        # Get model and replace the last layer
        self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=1)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        with torch.no_grad():
            pred = torch.sigmoid(self(x))
        return pred

def main():
    # Initialize model, criterion, optimizer
    model = Task1EfficientNetB4()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    x = torch.randn(1, 3, 512, 512).to(device)
    y_hat = model.predict(x).item()
    print(y_hat)

if __name__ == "__main__":
    main()

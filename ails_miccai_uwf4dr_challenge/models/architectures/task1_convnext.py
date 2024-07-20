from torchvision.models.convnext import ConvNeXt, ConvNeXt_Tiny_Weights, convnext_tiny
from torch import nn
import torch

class Task1ConvNeXt(nn.Module):
    def __init__(self):
        super(Task1ConvNeXt, self).__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        self.model = convnext_tiny(weights=weights)
        # Modify the final layer for binary classification
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, 1)
        
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        with torch.no_grad():
            pred = torch.sigmoid(self(x))
        return pred
    
def main():
    # Initialize model, criterion, optimizer
    model = Task1ConvNeXt()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    x = torch.randn(1, 3, 512, 512).to(device)
    y_hat = model.predict(x).item()
    print(y_hat)

if __name__ == "__main__":
    main()
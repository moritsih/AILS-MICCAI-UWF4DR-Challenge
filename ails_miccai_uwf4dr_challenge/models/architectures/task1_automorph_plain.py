import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from pathlib import Path
import torch.optim as optim

class AutoMorphModel(nn.Module):
    def __init__(self, pretrained=True, enc_frozen=False):
        super(AutoMorphModel, self).__init__()

        # code taken from https://github.com/rmaphoh/AutoMorph/blob/main/M1_Retinal_Image_quality_EyePACS/model.py
        self.model = EfficientNet.from_pretrained('efficientnet-b4')
        self.model._fc = nn.Identity()
        net_fl = nn.Sequential(
            nn.Linear(1792, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 3)
        )
        self.model._fc = net_fl
        if pretrained:
            checkpoint_path = Path().resolve() / "models" / "AutoMorph" / "automorph_best_loss_checkpoint.pth"
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

        # add a final layer that outputs single value
        self.model._fc.add_module("7", nn.Linear(3, 1))

        if enc_frozen:
            self.freeze_encoder()


    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model._fc.parameters():
            param.requires_grad = True


    def forward(self, x):
        return self.model(x)
    
def main():
    # Initialize model, criterion, optimizer
    model = AutoMorphModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    x = torch.randn(1, 3, 512, 512).to(device)
    y_hat = model(x).item()
    print(y_hat)

if __name__ == "__main__":
    main()


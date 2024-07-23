import torch
import torch.nn as nn
import warnings
import enum
warnings.filterwarnings("ignore")

class ModelSize(enum.Enum):
    SMALL = 'small'
    BASE = 'base'

class DinoV2Classifier(nn.Module):
    def __init__(self, size: ModelSize = ModelSize.SMALL):
        super(DinoV2Classifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if size == ModelSize.SMALL:
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg_lc')
        elif size == ModelSize.BASE:
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg_lc')
        else:
            raise ValueError(f"Invalid model size: {size}")
        
        self.model.to(self.device)

        head_in = self.model.linear_head.in_features
        head_out = self.model.linear_head.out_features
        
        classifier = nn.Sequential(
                                nn.Linear(head_in, head_out),
                                nn.ReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(1000, 256),
                                nn.ReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(256, 64), 
                                nn.ReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(64, 1)
                                )
        
        self.model.linear_head = classifier

        print(self.model)

    def forward(self, x):
        return self.model(x)

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def predict(self, input):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input)
            pred = torch.sigmoid(output).item()
        
        return pred


if __name__ == '__main__':
    model = DinoV2Classifier(size=ModelSize.SMALL)

    input = torch.randn(1, 3, 224, 224)
    output = model.predict(input)

    print(output)
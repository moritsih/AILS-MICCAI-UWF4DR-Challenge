from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import torch
from efficientnet_pytorch import EfficientNet
import lightning as L
from torchmetrics import Accuracy, MatthewsCorrCoef, AUROC

class AutoMorphModel(L.LightningModule):
    # https://github.com/lukemelas/EfficientNet-PyTorch

    def __init__(self):

        super(AutoMorphModel, self).__init__()

        # code taken from https://github.com/rmaphoh/AutoMorph/blob/main/M1_Retinal_Image_quality_EyePACS/model.py
        model = EfficientNet.from_pretrained('efficientnet-b4')

        # freeze all layers
        #model.freeze()

        model._fc = nn.Identity()
        net_fl = nn.Sequential(
                nn.Linear(1792, 256),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(256, 64), 
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(64, 3)
                )
        model._fc = net_fl

        checkpoint_path = Path().resolve() / "models" / "AutoMorph" / "best_loss_checkpoint.pth"
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        # add a final layer that outputs single value
        model._fc.add_module("7", nn.Linear(3, 1))

        self.model = model

        self.save_hyperparameters()

        self.loss_func = nn.BCEWithLogitsLoss()


        # Metrics

        task = 'binary'
        num_classes = 2

        self.train_accuracy = Accuracy(task=task, num_classes=num_classes)
        self.val_accuracy = Accuracy(task=task, num_classes=num_classes)
        self.test_accuracy = Accuracy(task=task, num_classes=num_classes)

        self.train_mcc = MatthewsCorrCoef(task=task, num_classes=num_classes)
        self.val_mcc = MatthewsCorrCoef(task=task, num_classes=num_classes)
        self.test_mcc = MatthewsCorrCoef(task=task, num_classes=num_classes)

        self.val_auroc = AUROC(task=task, num_classes=num_classes)
        self.test_auroc = AUROC(task=task, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        with torch.no_grad():
            pred = torch.sigmoid(self(x))
        return pred


    def training_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)

        acc = self.train_accuracy(y_hat, y)
        mcc = self.train_mcc(y_hat, y)

        self.log('train_loss', loss.item())
        self.log('train_acc', acc.float())
        self.log('train_mcc', mcc.float())

        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)

        loss = self.loss_fn(y_hat, y)

        acc = self.val_accuracy(y_hat, y)
        mcc = self.val_mcc(y_hat, y)
        auroc = self.val_auroc(y_hat, y)

        self.log('val_loss', loss.item())
        self.log('val_acc', acc.float())
        self.log('val_mcc', mcc.float())
        self.log('val_auroc', auroc.float())
        
        return loss
    

    def test_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)

        acc = self.test_accuracy(y_hat, y)
        mcc = self.test_mcc(y_hat, y)
        auroc = self.test_auroc(y_hat, y)

        self.log('test_loss', loss.item())
        self.log('test_acc', acc.float())
        self.log('test_mcc', mcc.float())
        self.log('test_auroc', auroc.float())
        
        return loss
    

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer, 
                "lr_scheduler": {'scheduler': scheduler, 'monitor': 'val_loss'}}




checkpoint_dir = Path(__file__).parent / 'checkpoints'
checkpoint_dir.mkdir(parents=True, exist_ok=True)

model_name = Path(__file__).stem

checkpoint_callback = L.callbacks.ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename='{model_name}-{epoch}-{val_loss:.2f}-{val_acc:.2f}',
    save_top_k=3, 
    monitor='val_loss',
    mode='min' 
)





if __name__ == "__main__":
    model = AutoMorphModel()
    print(model)

    y_hat = model.predict(torch.randn(1, 3, 512, 512)).item()
    print(y_hat)

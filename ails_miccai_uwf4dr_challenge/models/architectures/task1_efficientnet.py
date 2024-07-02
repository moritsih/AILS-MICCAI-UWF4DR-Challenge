import torch.optim as optim
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy, MatthewsCorrCoef, AUROC
from pathlib import Path
import torch
from efficientnet_pytorch import EfficientNet



class Task1EfficientNetB4(L.LightningModule):
    def __init__(self, learning_rate = 1e-3):

        super(Task1EfficientNetB4, self).__init__()

        # for logging hyperparameters in wandb
        self.save_hyperparameters()

        self.learning_rate = learning_rate

        # get model and replace the last layer
        self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=1)

        self.loss_fn = nn.BCEWithLogitsLoss()

        task = 'binary'
        num_classes = 2

        # Metrics
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



if __name__ == '__main__':
    model = Task1EfficientNetB4()
    #print(model)

    y_hat = model.predict(torch.randn(1, 3, 512, 512)).item()
    print(y_hat)



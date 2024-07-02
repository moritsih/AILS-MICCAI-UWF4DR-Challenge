from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
import torch.nn as nn
import torch.optim as optim
import torch
from efficientnet_pytorch import EfficientNet
import lightning as L
from torchmetrics import Accuracy, MatthewsCorrCoef, AUROC

class AutoMorphModel(L.LightningModule):
    # https://github.com/lukemelas/EfficientNet-PyTorch

    def __init__(self, learning_rate=1e-3):
        super(AutoMorphModel, self).__init__()

        self.learning_rate = learning_rate

        # code taken from https://github.com/rmaphoh/AutoMorph/blob/main/M1_Retinal_Image_quality_EyePACS/model.py
        model = EfficientNet.from_pretrained('efficientnet-b4')
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

        checkpoint_path = Path().resolve() / "models" / "AutoMorph" / "automorph_best_loss_checkpoint.pth"
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

        # To store validation outputs
        self.validation_outputs = []
        self.test_outputs = []

        # Initialize cumulative loss and batch count for training
        self.train_cumulative_loss = 0.0
        self.train_batch_count = 0

        # Initialize cumulative loss and batch count for validation
        self.val_cumulative_loss = 0.0
        self.val_batch_count = 0
        
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        with torch.no_grad():
            pred = torch.sigmoid(self(x))
        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_func(y_hat, y)

        acc = self.train_accuracy(y_hat, y)
        mcc = self.train_mcc(y_hat, y)

        return {'loss': loss, 'acc': acc, 'mcc': mcc}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Update cumulative loss and batch count
        self.train_cumulative_loss += outputs['loss'].item()
        self.train_batch_count += 1

        # Calculate average loss
        avg_loss = self.train_cumulative_loss / self.train_batch_count

        # Log average loss
        self.log('train_loss', avg_loss, prog_bar=True)
        self.log('train_acc', outputs['acc'].float())
        self.log('train_mcc', outputs['mcc'].float())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_func(y_hat, y)

        acc = self.val_accuracy(y_hat, y)
        mcc = self.val_mcc(y_hat, y)
        auroc = self.val_auroc(y_hat, y)

        self.validation_outputs.append({'val_loss': loss, 'val_y': y, 'val_y_hat': y_hat})

        return {'val_loss': loss, 'acc': acc, 'mcc': mcc, 'auroc': auroc}

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        # Update cumulative loss and batch count
        self.val_cumulative_loss += outputs['val_loss'].item()
        self.val_batch_count += 1

        # Calculate average loss
        avg_val_loss = self.val_cumulative_loss / self.val_batch_count

        # Log average loss
        self.log('val_loss', avg_val_loss, prog_bar=True)
        self.log('val_acc', outputs['acc'].float())
        self.log('val_mcc', outputs['mcc'].float())
        self.log('val_auroc', outputs['auroc'].float())

    def on_validation_epoch_end(self):
        y_true = torch.cat([x['val_y'] for x in self.validation_outputs]).cpu().numpy()
        y_score = torch.cat([x['val_y_hat'] for x in self.validation_outputs]).cpu().numpy()

        metrics = self.classification_metrics(y_true, y_score)
        self.log('val_auprc', metrics['auprc'])
        self.log('val_sensitivity', metrics['sensitivity'])
        self.log('val_specificity', metrics['specificity'])
        
        # Clear outputs
        self.validation_outputs.clear()

        # Reset cumulative loss and batch count
        self.val_cumulative_loss = 0.0
        self.val_batch_count = 0

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_func(y_hat, y)

        acc = self.test_accuracy(y_hat, y)
        mcc = self.test_mcc(y_hat, y)
        auroc = self.test_auroc(y_hat, y)

        self.log('test_loss', loss.item())
        self.log('test_acc', acc.float())
        self.log('test_mcc', mcc.float())
        self.log('test_auroc', auroc.float())

        self.test_outputs.append({'test_loss': loss, 'test_y': y, 'test_y_hat': y_hat})

    def on_test_epoch_end(self):
        y_true = torch.cat([x['test_y'] for x in self.test_outputs]).cpu().numpy()
        y_score = torch.cat([x['test_y_hat'] for x in self.test_outputs]).cpu().numpy()

        metrics = self.classification_metrics(y_true, y_score)
        self.log('test_auprc', metrics['auprc'])
        self.log('test_sensitivity', metrics['sensitivity'])
        self.log('test_specificity', metrics['specificity'])
        
        # Clear outputs
        self.test_outputs.clear()

    def classification_metrics(self, y_true, y_score):
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        best_threshold_index = np.argmax(tpr - fpr)

        sensitivity = tpr[best_threshold_index]
        specificity = 1 - fpr[best_threshold_index]

        return dict(auroc=auroc, auprc=auprc, sensitivity=sensitivity, specificity=specificity)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer, 
                "lr_scheduler": {'scheduler': scheduler, 'monitor': 'val_loss'}}



checkpoint_dir = Path(__file__).parent / 'checkpoints'
checkpoint_dir.mkdir(parents=True, exist_ok=True)

model_name = Path(__file__).stem

"""" TODO clarify with Moritz his lightning version
checkpoint_callback = L.callbacks.ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename='{model_name}-{epoch}-{val_loss:.2f}-{val_acc:.2f}',
    save_top_k=3, 
    monitor='val_loss',
    mode='min' 
)
"""


if __name__ == "__main__":
    model = AutoMorphModel()
    print(model)

    y_hat = model.predict(torch.randn(1, 3, 512, 512)).item()
    
    #print the parameters per model layer:
    for name, param in model.named_parameters():
        print(name, param.size())
    
    print(y_hat)

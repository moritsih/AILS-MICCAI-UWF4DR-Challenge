import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import roc_auc_score, average_precision_score

from ails_miccai_uwf4dr_challenge.models.metrics import sensitivity_score, specificity_score

from ails_miccai_uwf4dr_challenge.dataset import ChallengeTaskType, CustomDataset, DatasetBuilder, DatasetOriginationType
from ails_miccai_uwf4dr_challenge.augmentations import rotate_affine_flip_choice, resize_only
from torch.utils.data import DataLoader

from ails_miccai_uwf4dr_challenge.models.trainer import DefaultMetricsEvaluationStrategy, Metric, MetricCalculatedHook, NumBatches, Trainer, EpochTrainingStrategy, EpochValidationStrategy, DefaultEpochTrainingStrategy, DefaultBatchTrainingStrategy, TrainingContext
from ails_miccai_uwf4dr_challenge.models.architectures.task1_automorph_plain import AutoMorphModel
from ails_miccai_uwf4dr_challenge.models.architectures.task1_efficientnet_plain import Task1EfficientNetB4

LEARNING_RATE = 1e-3
EPOCHS = 15

config={
    "learning_rate": LEARNING_RATE,
    "dataset": "UWF4DR-DEEPDRID",
    "epochs": EPOCHS,
    }



def main():
    dataset = DatasetBuilder(dataset=DatasetOriginationType.ALL, task=ChallengeTaskType.TASK1)
    train_data, val_data = dataset.get_train_val()

    train_dataset = CustomDataset(train_data, transform=rotate_affine_flip_choice)
    val_dataset = CustomDataset(val_data, transform=resize_only)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" if torch.backends.mps.is_available() else "cpu") #don't use mps, it takes ages, whyever that is the case!?!
    print(f"Using device: {device}")

    model1 = AutoMorphModel()
    model2 = Task1EfficientNetB4()
    
    for model in [model2, model1]:        
        model.to(device)
        
        model_class_name = model.__class__.__name__
        print("Training model: ", model_class_name)    
        wandb.init(project="task1", config=config.update({"model": model_class_name}))

        metrics = [
            Metric('auroc', roc_auc_score),
            Metric('auprc', average_precision_score),
            Metric('accuracy', lambda y_true, y_pred: (y_pred.round() == y_true).mean()),
            Metric('sensitivity', sensitivity_score),
            Metric('specificity', specificity_score)
        ]

        class WandbLoggingHook(MetricCalculatedHook):
            def on_metric_calculated(self, training_context: TrainingContext, metric: Metric, result, last_metric_for_epoch: bool):
                import wandb
                print(f"Logging metric {metric.name} with value {result} to wandb - commit: {last_metric_for_epoch}")
                wandb.log(data={metric.name: result}, commit=last_metric_for_epoch)

        metrics_eval_strategy = DefaultMetricsEvaluationStrategy(metrics).register_metric_calculated_hook(WandbLoggingHook())
    
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, device, 
                          metrics_eval_strategy=metrics_eval_strategy)

        print("First train 2 epochs 2 batches to check if everything works - you can comment these two lines after the code has stabilized...")
        trainer.train(num_epochs=2, num_batches=NumBatches.TWO_FOR_INITIAL_TESTING)
        
        print("Now train train train")
        trainer.train(num_epochs=EPOCHS)



        print("Finished training")

if __name__ == "__main__":
    main()

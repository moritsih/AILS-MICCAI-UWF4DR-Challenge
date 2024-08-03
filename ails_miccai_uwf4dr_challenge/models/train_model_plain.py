import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

import wandb
# augmentation
from ails_miccai_uwf4dr_challenge.augmentations import transforms_train, transforms_val
from ails_miccai_uwf4dr_challenge.config import WANDB_API_KEY

# data
from ails_miccai_uwf4dr_challenge.dataset_strategy import CustomDataset, CombinedDatasetStrategy, \
    Task2Strategy, DatasetBuilder, OriginalDatasetStrategy, Task3Strategy, Task1Strategy
from ails_miccai_uwf4dr_challenge.models.architectures.ResNets import ResNet, ResNetVariant
from ails_miccai_uwf4dr_challenge.models.architectures.task1_automorph_plain import AutoMorphModel
from ails_miccai_uwf4dr_challenge.models.architectures.task1_convnext import Task1ConvNeXt
from ails_miccai_uwf4dr_challenge.models.architectures.task1_efficientnet_plain import Task1EfficientNetB4
from ails_miccai_uwf4dr_challenge.models.architectures.task2_efficientnetb0_plain import Task2EfficientNetB0
from ails_miccai_uwf4dr_challenge.models.architectures.task3_efficientnetb0_plain import Task3EfficientNetB0
from ails_miccai_uwf4dr_challenge.models.metrics import sensitivity_score, specificity_score
from ails_miccai_uwf4dr_challenge.models.trainer import DefaultMetricsEvaluationStrategy, Metric, MetricCalculatedHook, \
    NumBatches, Trainer, TrainingContext, PersistBestModelOnEpochEndHook, UndersamplingResamplingStrategy, WeightedSamplingStrategy, \
    SamplingStrategy, SigmoidFocalLoss

LEARNING_RATE = 1e-3
EPOCHS = 15
BATCH_SIZE = 8
MODEL_TYPE = Task3EfficientNetB0() # Task1EfficientNetB4(), Task1ConvNeXt(), ResNet(), Task2EfficientNetB0(), Task3EfficientNetB0()
LOSS = nn.BCEWithLogitsLoss() # nn.BCEWithLogitsLoss(), SigmoidFocalLoss()
TASK = Task3Strategy() # Task1Strategy(), Task2Strategy(), Task3Strategy()
DATASET = CombinedDatasetStrategy() # OriginalDatasetStrategy(), CombinedDatasetStrategy()
LOSS_TYPE = "with_weights" #None


def train(config=None, data= DATASET, task= TASK, loss=LOSS):
    wandb.init(project="task3", config=config)
    config = wandb.config

    dataset_strategy = data
    task_strategy = task
    builder = DatasetBuilder(dataset_strategy, task_strategy, split_ratio=0.8)
    train_data, val_data = builder.build()

    train_dataset = CustomDataset(train_data, transform=transforms_train)
    val_dataset = CustomDataset(val_data, transform=transforms_val)

    #sampler = WeightedSamplingStrategy(train_dataset).sampler()

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, sampler=None)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu" if torch.backends.mps.is_available() else "cpu")  #don't use mps, it takes ages, whyever that is the case!?!
    print(f"Using device: {device}")

    if config.model_type == 'AutoMorphModel':
        model = AutoMorphModel()
    elif config.model_type == 'Task1EfficientNetB4':
        model = Task1EfficientNetB4()
    elif config.model_type == 'Task1ConvNeXt':
        model = Task1ConvNeXt()
    elif config.model_type == 'ResNet':
        model = ResNet(model_variant=ResNetVariant.RESNET18),  # or RESNET34, RESNET50
    elif config.model_type == 'Task2EfficientNetB0':
        model = Task2EfficientNetB0()
    elif config.model_type == 'Task3EfficientNetB0':
        model = Task3EfficientNetB0()
    else:
        raise ValueError(f"Unknown model: {config.model_type}")

    model.to(device)

    print("Training model: ", model.__class__.__name__)

    metrics = [
        Metric('auroc', roc_auc_score),
        Metric('auprc', average_precision_score),
        Metric('accuracy', lambda y_true, y_pred: (y_pred.round() == y_true).mean()),
        Metric('sensitivity', sensitivity_score),
        Metric('specificity', specificity_score)
    ]

    class WandbLoggingHook(MetricCalculatedHook):
        def on_metric_calculated(self, training_context: TrainingContext, metric: Metric, result,
                                 last_metric_for_epoch: bool):
            import wandb
            wandb.log(data={metric.name: result}, commit=last_metric_for_epoch)

    metrics_eval_strategy = DefaultMetricsEvaluationStrategy(metrics).register_metric_calculated_hook(
        WandbLoggingHook())

    criterion = loss
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, device,
                      metrics_eval_strategy=metrics_eval_strategy,
                      val_dataloader_adapter=None,
                      train_dataloader_adapter=None)

    # build a file name for the model weights containing current timestamp and the model class
    training_date = time.strftime("%Y-%m-%d")
    weight_file_name = f"{config.model_type}_weights_{training_date}_{wandb.run.name}.pth"
    persist_model_hook = PersistBestModelOnEpochEndHook(weight_file_name, print_train_results=True)
    trainer.add_epoch_end_hook(persist_model_hook)

    print(
        "First train 2 epochs 2 batches to check if everything works - you can comment these two lines after the code has stabilized...")
    trainer.train(num_epochs=2, num_batches=NumBatches.TWO_FOR_INITIAL_TESTING)

    print("Now train train train")
    trainer.train(num_epochs=config["epochs"])

    print("Finished training")


if __name__ == "__main__":
    wandb.require(
        "core")  # The new W&B backend becomes opt-out in version 0.18.0; try it out with `wandb.require("core")`! See https://wandb.me/wandb-core for more information.




    config = {
        "learning_rate": LEARNING_RATE,
        "dataset": DATASET,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "model_type": MODEL_TYPE.__class__.__name__
        "loss": LOSS.__class__.__name__
        "task": TASK.__class__.__name__
        "dataset": DATASET.__class__.__name__
    }

    wandb.login(key=WANDB_API_KEY)

    train(config)

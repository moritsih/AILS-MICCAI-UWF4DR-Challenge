import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

import wandb
# augmentation
from ails_miccai_uwf4dr_challenge.augmentations import rotate_affine_flip_choice, resize_only
from ails_miccai_uwf4dr_challenge.config import WANDB_API_KEY
# data
from ails_miccai_uwf4dr_challenge.dataset_strategy import CustomDataset, CombinedDatasetStrategy, \
    Task2Strategy, DatasetBuilder
from ails_miccai_uwf4dr_challenge.models.architectures.ResNets import ResNet, ResNetVariant
from ails_miccai_uwf4dr_challenge.models.architectures.task1_automorph_plain import AutoMorphModel
from ails_miccai_uwf4dr_challenge.models.architectures.task1_convnext import Task1ConvNeXt
from ails_miccai_uwf4dr_challenge.models.architectures.task1_efficientnet_plain import Task1EfficientNetB4
from ails_miccai_uwf4dr_challenge.models.metrics import sensitivity_score, specificity_score
from ails_miccai_uwf4dr_challenge.models.trainer import DefaultMetricsEvaluationStrategy, Metric, MetricCalculatedHook, \
    NumBatches, Trainer, TrainingContext, PersistBestModelOnEpochEndHook, UndersamplingResamplingStrategy


def train(config=None):
    wandb.init(project="task1", config=config)
    config = wandb.config

    dataset_strategy = CombinedDatasetStrategy()
    task_strategy = Task2Strategy()

    builder = DatasetBuilder(dataset_strategy, task_strategy, split_ratio=0.8)
    train_data, val_data = builder.build()

    train_dataset = CustomDataset(train_data, transform=rotate_affine_flip_choice)
    val_dataset = CustomDataset(val_data, transform=resize_only)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # don't use mps, it takes ages, why ever that is the case!?!
    # --> with my new m3, mps works fine!
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if config.model_type == 'AutoMorphModel':
        model = AutoMorphModel()
    elif config.model_type == 'Task1EfficientNetB4':
        model = Task1EfficientNetB4()
    elif config.model_type == 'Task1ConvNeXt':
        model = Task1ConvNeXt()
    elif config.model_type == 'ResNet':
        model = ResNet(model_variant=ResNetVariant.RESNET18)  # or RESNET34, RESNET50
    else:
        raise ValueError(f"Unknown model: {config.model_type}")

    model.to(device)

    print("Training model: ", model.__class__.__name__)

    def safe_roc_auc_score(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError as e:
            if str(e) == 'Only one class present in y_true. ROC AUC score is not defined in that case.':
                print(f"Could not evaluate metric auroc: {e}")
                return None
            else:
                raise

    metrics = [
        Metric('auroc', safe_roc_auc_score),
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

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, device,
                      metrics_eval_strategy=metrics_eval_strategy,
                      val_dataloader_adapter=UndersamplingResamplingStrategy(),
                      train_dataloader_adapter=UndersamplingResamplingStrategy())

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

    LEARNING_RATE = 1e-3
    EPOCHS = 15

    config = {
        "learning_rate": LEARNING_RATE,
        "dataset": "UWF4DR-DEEPDRID",
        "epochs": EPOCHS,
        "batch_size": 4,
        "model_type": Task1ConvNeXt().__class__.__name__
    }

    # wandb.login(key=WANDB_API_KEY)

    train(config)

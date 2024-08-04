import time
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

import wandb
from ails_miccai_uwf4dr_challenge.config import WANDB_API_KEY
# augmentation
from ails_miccai_uwf4dr_challenge.augmentations import rotate_affine_flip_choice, resize_only
# data
from ails_miccai_uwf4dr_challenge.dataset_strategy import CustomDataset, CombinedDatasetStrategy, \
    Task2Strategy, DatasetBuilder, TrainAndValData
# models
from ails_miccai_uwf4dr_challenge.models.architectures.ResNets import ResNet, ResNetVariant
from ails_miccai_uwf4dr_challenge.models.architectures.task1_automorph_plain import AutoMorphModel
from ails_miccai_uwf4dr_challenge.models.architectures.task1_convnext import Task1ConvNeXt
from ails_miccai_uwf4dr_challenge.models.architectures.task1_efficientnet_plain import Task1EfficientNetB4
# metrics
from ails_miccai_uwf4dr_challenge.models.metrics import sensitivity_score, specificity_score
# training
from ails_miccai_uwf4dr_challenge.models.trainer import DefaultMetricsEvaluationStrategy, Loaders, Metric, MetricCalculatedHook, \
    NumBatches, Trainer, TrainingContext, PersistBestModelOnEpochEndHook, UndersamplingResamplingStrategy, OversamplingResamplingStrategy, \
        TrainingRunHardware, DefaultTrainingRunStartHook, DefaultTrainingRunEndHook, WandbLoggingHook

def train(config=None):

    metrics = [
        Metric('auroc', roc_auc_score),
        Metric('auprc', average_precision_score),
        Metric('accuracy', lambda y_true, y_pred: (y_pred.round() == y_true).mean()),
        Metric('sensitivity', sensitivity_score),
        Metric('specificity', specificity_score)
    ]

    wandb.init(project="task1", config=config, group=training_run_grouping)
    config = wandb.config

    dataset_strategy = CombinedDatasetStrategy()
    task_strategy = Task2Strategy()

    builder = DatasetBuilder(dataset_strategy, task_strategy, split_ratio=0.8, n_folds=config["num_folds"], 
                             train_set_transformations=rotate_affine_flip_choice, val_set_transformations=resize_only)
    
    train_and_val_data : List[TrainAndValData] = builder.build()
    
    loaders = []
    
    for i, train_val_data in enumerate(train_and_val_data):    

        training_date = time.strftime("%Y-%m-%d")
        training_run_grouping = f"{config.model_type}_{training_date}_fold{i+1}" 


        train_loader = DataLoader(train_val_data.train_data, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(train_val_data.val_data, batch_size=config['batch_size'], shuffle=False)        
        loaders.append(Loaders(train_loader, val_loader))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu" if torch.backends.mps.is_available() else "cpu")  #don't use mps, it takes ages, whyever that is the case!?!
    print(f"Using device: {device}")


    metrics_eval_strategy = DefaultMetricsEvaluationStrategy(metrics).register_metric_calculated_hook(WandbLoggingHook())

    model_run_specific_stuff = create_training_run_hardware(config, device)

    #on_training_run_start_hook = function which creates all model run specific stuff and returns it to the trainer

    model, criterion, optimizer, lr_scheduler = model_run_specific_stuff

    trainer = Trainer(TrainingRunHardware(model, criterion, optimizer, lr_scheduler), device, multiple_loaders = loaders,
                      metrics_eval_strategy=metrics_eval_strategy, on_training_run_start_hook=DefaultTrainingRunStartHook())

    # build a file name for the model weights containing current timestamp and the model class
    weight_file_name = f"{config.model_type}_weights_{training_date}_{wandb.run.name}.pth"
    persist_model_hook = PersistBestModelOnEpochEndHook(weight_file_name, print_train_results=True)
    trainer.add_epoch_end_hook(persist_model_hook)

    #print(
    #    "First train 2 epochs 2 batches to check if everything works - you can comment these two lines after the code has stabilized...")
    #trainer.train(num_epochs=2, num_batches=NumBatches.TWO_FOR_INITIAL_TESTING)

    print("Now train train train")
    trainer.train(num_epochs=config["epochs"])

    print("Finished training")
    

def create_training_run_hardware(config, device):
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

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    return model, criterion, optimizer, lr_scheduler


if __name__ == "__main__":
    wandb.require(
        "core")  # The new W&B backend becomes opt-out in version 0.18.0; try it out with `wandb.require("core")`! See https://wandb.me/wandb-core for more information.

    LEARNING_RATE = 1e-3
    EPOCHS = 15
    NUM_FOLDS = 1

    config = {
        "learning_rate": LEARNING_RATE,
        "dataset": "UWF4DR-DEEPDRID",
        "epochs": EPOCHS,
        "num_folds": NUM_FOLDS,
        "batch_size": 4,
        "model_type": Task1ConvNeXt().__class__.__name__
    }

    wandb.login(key=WANDB_API_KEY)

    train(config)

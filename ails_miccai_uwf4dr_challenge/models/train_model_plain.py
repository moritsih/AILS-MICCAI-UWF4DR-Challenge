import time
from typing import List
from faker import Faker

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
import torch.nn.functional as F

import wandb
from ails_miccai_uwf4dr_challenge.config import WANDB_API_KEY, Config
# augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ails_miccai_uwf4dr_challenge.augmentations import rotate_affine_flip_choice, resize_only
from ails_miccai_uwf4dr_challenge.preprocess_augmentations import ResidualGaussBlur, MultiplyMask
# data
from ails_miccai_uwf4dr_challenge.dataset_strategy import CustomDataset, CombinedDatasetStrategy, \
    OriginalDatasetStrategy, DeepDridDatasetStrategy, Task1Strategy, Task2Strategy, Task3Strategy, \
    DatasetBuilder, Loaders, MiniDatasetStrategy
# models
from ails_miccai_uwf4dr_challenge.models.architectures.ResNets import ResNet, ResNetVariant
from ails_miccai_uwf4dr_challenge.models.architectures.task1_automorph_plain import AutoMorphModel
from ails_miccai_uwf4dr_challenge.models.architectures.task1_convnext import Task1ConvNeXt
from ails_miccai_uwf4dr_challenge.models.architectures.task1_efficientnet_plain import Task1EfficientNetB4
from ails_miccai_uwf4dr_challenge.models.architectures.shufflenet import ShuffleNet
# metrics
from ails_miccai_uwf4dr_challenge.models.metrics import sensitivity_score, specificity_score
# training
from ails_miccai_uwf4dr_challenge.models.trainer import DefaultMetricsEvaluationStrategy, Loaders, Metric, MetricCalculatedHook, \
    NumBatches, Trainer, TrainingContext, PersistBestModelOnEpochEndHook, UndersamplingResamplingStrategy, OversamplingResamplingStrategy, \
    DoNothingDataloaderPerEpochAdapter, TrainingRunHardware, WandbLoggingHook, InitWandbTrainingStartHook, FinishWandbTrainingEndHook


def create_training_run_hardware(config, device):
    '''
    Use this method to create the model, criterion, optimizer and lr_scheduler for your training run
    '''

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
    elif config.model_type == 'ShuffleNet':
        model = ShuffleNet()
    else:
        raise ValueError(f"Unknown model: {config.model_type}")

    model.to(device)

    print("Training model: ", model.__class__.__name__)


    def bce_smoothl1_combined(pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target) * config.loss_weight
        smooth_l1 = F.smooth_l1_loss(pred, target) * (1 - config.loss_weight)
        return bce + smooth_l1


    criterion = bce_smoothl1_combined # or nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.lr_scheduler_cycle_epochs, eta_min=config.lr_scheduler_min_lr)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.lr_scheduler_factor, patience=3, verbose=True)

    return TrainingRunHardware(model, criterion, optimizer, lr_scheduler)


def get_augmentations(config):
    transforms_train = A.Compose([
        A.Resize(800, 1016, p=1),
        MultiplyMask(p=1), # comment out whenever not doing task 1
        ResidualGaussBlur(p=config.p_gaussblur),
        A.Equalize(p=config.p_equalize),
        A.CLAHE(clip_limit=5., p=config.p_clahe),
        A.HorizontalFlip(p=config.p_horizontalflip),
        A.Affine(rotate=config.rotation, rotate_method='ellipse', p=config.p_affine),
        A.Normalize(mean=[0.406, 0.485, 0.456], std=[0.225, 0.229, 0.224], p=1),
        #A.Resize(770, 1022, p=1), # comment whenever not using DinoV2
        ToTensorV2(p=1)
    ])

    transforms_val = A.Compose([
            A.Resize(800, 1016, p=1),
            MultiplyMask(p=1),
            A.Normalize(mean=[0.406, 0.485, 0.456], std=[0.225, 0.229, 0.224], p=1),
            #A.Resize(770, 1022, p=1), # comment whenever not using DinoV2
            ToTensorV2(p=1)
        ])
    
    return transforms_train, transforms_val




def train(config=None):

    assert config is not None, "Config must be provided"

    fake = Faker()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu" if torch.backends.mps.is_available() else "cpu")  #don't use mps, it takes ages, whyever that is the case!?!
    print(f"Using device: {device}")

    metrics = [
        Metric('auroc', roc_auc_score),
        Metric('auprc', average_precision_score),
        Metric('accuracy', lambda y_true, y_pred: (y_pred.round() == y_true).mean()),
        Metric('sensitivity', sensitivity_score),
        Metric('specificity', specificity_score)
    ]

    dataset_strategy = config.dataset
    task_strategy = config.task

    transform_train, transform_val = get_augmentations(config)

    builder = DatasetBuilder(dataset_strategy, task_strategy, 
                             split_ratio=0.8, n_folds=config.num_folds, batch_size=config.batch_size,
                             train_set_transformations=transform_train, val_set_transformations=transform_val)
    
    loaders : List[Loaders] = builder.build()

    metrics_eval_strategy = DefaultMetricsEvaluationStrategy(metrics).register_metric_calculated_hook(WandbLoggingHook())

    model_run_hardware = create_training_run_hardware(config, device)

    trainer = Trainer(model_run_hardware, loaders, device,
                      metrics_eval_strategy=metrics_eval_strategy,
                      train_dataloader_adapter=config.resampling_strategy,
                      val_dataloader_adapter=config.resampling_strategy) 
    
    # what should happen when a training run starts?
    wandb_group_name = fake.word() + "-" + fake.word() # generates a random name for the training run like wandb does
    wandb_config = config
    wandb_notes = config.notes
    wandb_init_hook = InitWandbTrainingStartHook(config.wandb_task, wandb_group_name, wandb_config, wandb_notes)
    trainer.add_training_run_start_hook(wandb_init_hook)

    # what should happend when an epoch ends?
    # build a file name for the model weights containing current timestamp and the model class
    training_date = time.strftime("%Y-%m-%d")
    file_name = f"{config.model_type}_weights_{training_date}"
    model_path = f"models/{wandb_group_name}/{file_name}"
    persist_model_hook = PersistBestModelOnEpochEndHook(model_path, print_train_results=True)
    trainer.add_epoch_end_hook(persist_model_hook)

    # what should happen when a training run ends?
    trainer.add_training_run_end_hook(FinishWandbTrainingEndHook())


    # "First train 2 epochs 2 batches to check if everything works - you can comment this line after the code has stabilized..."
    #print("First train 2 epochs 2 batches to check if everything works - "
    #      "you can comment these lines after the code has stabilized...")
    #trainer.train(num_epochs=2, num_batches=NumBatches.TWO_FOR_INITIAL_TESTING)

    print("Now train train train")
    trainer.train(num_epochs=config.epochs)

    print("Finished training")
    



if __name__ == "__main__":
    wandb.require(
        "core")  # The new W&B backend becomes opt-out in version 0.18.0; try it out with `wandb.require("core")`! See https://wandb.me/wandb-core for more information.

    LEARNING_RATE = 1e-3
    EPOCHS = 2
    NUM_FOLDS = 3
    BATCH_SIZE = 4

    config = Config(

        dataset=MiniDatasetStrategy(), #CombinedDatasetStrategy(),
        task=Task2Strategy(),
        resampling_strategy=OversamplingResamplingStrategy(), # or UndersamplingResamplingStrategy() or DoNothingDataloaderPerEpochAdapter()
        wandb_task="test", # or task 1, task2 or task3

        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        num_folds=NUM_FOLDS,
        batch_size=BATCH_SIZE,

        lr_scheduler_factor=0.5, # lr reduction multiplicative factor for ReduceLRonPlateau
        lr_scheduler_cycle_epochs=7, # cosine annealing cycle length
        lr_scheduler_min_lr=1e-6, # minimum learning rate for cosine annealing

        # probabilities for augmentations
        p_gaussblur=5,
        p_equalize=0.0,
        p_clahe=0.5,
        p_horizontalflip=0.5,
        rotation=15,
        p_affine=0.3,

        loss_weight=0.5, # weight for BCE loss in combined loss function

        model_type=ShuffleNet().__class__.__name__,
        notes="Shufflenet with 4 batch size" # use this field to describe the experiment - it will show up in wandb,
    )

    wandb.login(key=WANDB_API_KEY)

    train(config)


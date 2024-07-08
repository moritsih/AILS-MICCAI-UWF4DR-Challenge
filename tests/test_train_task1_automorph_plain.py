from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from ails_miccai_uwf4dr_challenge.models.metrics import sensitivity_score, specificity_score

from ails_miccai_uwf4dr_challenge.augmentations import rotate_affine_flip_choice, resize_only
from ails_miccai_uwf4dr_challenge.dataset import DatasetBuilder, DatasetOriginationType, ChallengeTaskType, \
    CustomDataset
from ails_miccai_uwf4dr_challenge.models.trainer import DefaultMetricsEvaluationStrategy, Trainer, NumBatches, Metric
from ails_miccai_uwf4dr_challenge.models.architectures.task1_automorph_plain import AutoMorphModel


def test_train_task1_automorph_plain():
    """
    simple test to verify that automorph model is working
    """

    dataset_builder = DatasetBuilder(dataset=DatasetOriginationType.ALL, task=ChallengeTaskType.TASK1)
    train_data, val_data = dataset_builder.get_train_val()

    train_dataset = CustomDataset(train_data, transform=rotate_affine_flip_choice)
    val_dataset = CustomDataset(val_data, transform=resize_only)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1)

    model = AutoMorphModel()
    learning_rate = 1e-3
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    metrics = [
        Metric('auroc', roc_auc_score),
        Metric('auprc', average_precision_score),
        Metric('accuracy', lambda y_true, y_pred: (y_pred.round() == y_true).mean()),
        Metric('sensitivity', sensitivity_score),
        Metric('specificity', specificity_score)
    ]

    metrics_eval_strategy = DefaultMetricsEvaluationStrategy(metrics)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, 'cpu',
                      metrics_eval_strategy=metrics_eval_strategy)

    trainer.train(num_epochs=1, num_batches=NumBatches.ONE_FOR_INITIAL_TESTING)
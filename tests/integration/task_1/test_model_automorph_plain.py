import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

from ails_miccai_uwf4dr_challenge.augmentations import rotate_affine_flip_choice, resize_only
from ails_miccai_uwf4dr_challenge.dataset import ChallengeTaskType
from ails_miccai_uwf4dr_challenge.models.architectures.task1_automorph_plain import AutoMorphModel
from ails_miccai_uwf4dr_challenge.models.metrics import sensitivity_score, specificity_score
from ails_miccai_uwf4dr_challenge.models.trainer import DefaultMetricsEvaluationStrategy, Trainer, NumBatches, Metric
from tests.test_utils.mock_dataset import MockDatasetBuilder, MockDataset


class TestModelAutomorphPlain:
    @pytest.mark.skip(reason="This test is quite resource intensive and should be run only when needed")
    def test_train_with_mock_data(self):
        task_type_task_1 = ChallengeTaskType.TASK1

        mock_dataset_builder = MockDatasetBuilder(num_samples=100, task_type=task_type_task_1)
        train_data, val_data = mock_dataset_builder.get_train_val()

        train_dataset = MockDataset(data=train_data, task_type=task_type_task_1, transform=rotate_affine_flip_choice)
        val_dataset = MockDataset(data=val_data, task_type=task_type_task_1, transform=resize_only)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1)

        model = AutoMorphModel(pretrained=False)

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

        trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, lr_scheduler,  torch.device("cpu"),
                          metrics_eval_strategy=metrics_eval_strategy)
        trainer.train(num_epochs=1, num_batches=NumBatches.ONE_FOR_INITIAL_TESTING)

    def test_inference_with_mock_data(self):
        task_type_task_1 = ChallengeTaskType.TASK1

        mock_dataset_builder = MockDatasetBuilder(num_samples=100, task_type=task_type_task_1)
        train_data, val_data = mock_dataset_builder.get_train_val()

        train_dataset = MockDataset(data=train_data, task_type=task_type_task_1,
                                          transform=rotate_affine_flip_choice)
        val_dataset = MockDataset(data=val_data, task_type=task_type_task_1, transform=resize_only)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1)

        model = AutoMorphModel(pretrained=False)
        device = torch.device("cpu")
        model.to(device)

        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                print(f"Inference output: {outputs}")
                break  # run for first batch only

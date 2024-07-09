import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from ails_miccai_uwf4dr_challenge.dataset import ChallengeTaskType

class MockDatasetBuilder:
    def __init__(self, num_samples=100, task_type=ChallengeTaskType.TASK1, split_ratio=0.8):
        self.num_samples = num_samples
        self.task_type = task_type
        self.split_ratio = split_ratio
        self.data = self._create_mock_data()

    def _create_mock_data(self):
        if self.task_type == ChallengeTaskType.TASK1:
            data = {
                'image': [f'image_{i}.jpg' for i in range(self.num_samples)],
                'quality': np.random.choice([0, 1], self.num_samples)  # 0: Bad quality, 1: Good quality
            }
        elif self.task_type == ChallengeTaskType.TASK2:
            data = {
                'image': [f'image_{i}.jpg' for i in range(self.num_samples)],
                'dr': np.random.choice([0, 1], self.num_samples)  # 0: No DR, 1: DR
            }
        elif self.task_type == ChallengeTaskType.TASK3:
            data = {
                'image': [f'image_{i}.jpg' for i in range(self.num_samples)],
                'dme': np.random.choice([0, 1], self.num_samples)  # 0: No DME, 1: DME
            }
        else:
            raise ValueError(f"Invalid task type: {self.task_type}")
        return pd.DataFrame(data)

    def get_data(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def get_train_val(self):
        train_data = self.data.sample(frac=self.split_ratio, random_state=42)
        val_data = self.data.drop(train_data.index).reset_index(drop=True)
        train_data = train_data.reset_index(drop=True)
        return train_data, val_data

class MockDataset(Dataset):
    def __init__(self, data, task_type=ChallengeTaskType.TASK1, transform=None):
        self.data = data
        self.task_type = task_type
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        if self.task_type == ChallengeTaskType.TASK1:
            label = self.data.iloc[idx, 1]  # quality
        elif self.task_type == ChallengeTaskType.TASK2:
            label = self.data.iloc[idx, 1]  # dr
        elif self.task_type == ChallengeTaskType.TASK3:
            label = self.data.iloc[idx, 1]  # dme

        # Create a mock image (e.g., 224x224 with 3 channels)
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

        if self.transform:
            img = self.transform(img)

        # Convert image and label to tensor
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        return img, label

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from ails_miccai_uwf4dr_challenge.config import RAW_DATA_DIR, EXTERNAL_DATA_DIR


# add seeds
torch.manual_seed(42)
np.random.seed(42)


def sample_weights(labels):
    # Calculate class weights
    class_weights = 1 / labels.value_counts(normalize=True).sort_index().values
    sample_weights = class_weights[labels.values]
    return sample_weights

class DatasetStrategy(ABC):
    @abstractmethod
    def get_data(self):
        pass

    def clean_data(self, data):
        data['quality'] = data['quality'].astype("Int64")
        data['dr'] = data['dr'].astype(float).round().astype("Int64")
        data['dme'] = data['dme'].astype("Int64")
        return data
    



class OriginalDatasetStrategy(DatasetStrategy):
    def get_data(self):
        # Implementation from your original OriginalDataset class
        task1_dir = Path(RAW_DATA_DIR / 'Task 1 Image Quality Assessment')
        task23_dir = Path(RAW_DATA_DIR / 'Task 2 Referable DR and Task 3 DME')

        imgs_task1 = pd.DataFrame(Path(task1_dir / '1. Images' / '1. Training').glob('*.jpg'), columns=['image'])
        imgs_task23 = pd.DataFrame(Path(task23_dir / '1. Images' / '1. Training').glob('*.jpg'), columns=['image'])

        data = pd.concat([imgs_task1, imgs_task23], ignore_index=True)

        labels_task1 = pd.read_csv(task1_dir / '2. Groundtruths' / '1. Training.csv')
        labels_task23 = pd.read_csv(task23_dir / '2. Groundtruths' / '1. Training.csv')

        labels = pd.merge(labels_task1, labels_task23, on='image', how='outer')

        for i in range(len(data)):
            img_name = data['image'].iloc[i].name
            row = labels[labels['image'] == img_name]
            if len(row) > 0:
                data.loc[i, 'quality'] = row['image quality level'].values[0]
                data.loc[i, 'dr'] = row['referable diabetic retinopathy'].values[0]
                data.loc[i, 'dme'] = row['diabetic macular edema'].values[0]

        data['quality'] = data['quality'].apply(lambda x: int(x) if not np.isnan(x) else x).astype("Int64")
        data['dr'] = data['dr'].apply(lambda x: int(x) if not np.isnan(x) else x).astype("Int64")
        data['dme'] = data['dme'].apply(lambda x: int(x) if not np.isnan(x) else x).astype("Int64")

        return self.clean_data(data)


class DeepDridDatasetStrategy(DatasetStrategy):
    def get_data(self):
        # Implementation from your original DeepDridDataset class
        deepdrid_path = Path(EXTERNAL_DATA_DIR) / 'DeepDRiD' / 'ultra-widefield_images'
        images = pd.DataFrame(deepdrid_path.rglob('*.jpg'), columns=['image_path'])

        training_labels = pd.read_csv(deepdrid_path / 'ultra-widefield-training' / 'ultra-widefield-training.csv')
        test_labels = pd.read_excel(deepdrid_path / 'Online-Challenge3-Evaluation' / 'Challenge3_labels.xlsx')
        validation_labels = pd.read_csv(deepdrid_path / 'ultra-widefield-validation' / 'ultra-widefield-validation.csv')

        training_labels = training_labels.apply(self._standardize_label_df, axis=1)
        test_labels = test_labels.apply(self._standardize_label_df, axis=1)
        validation_labels = validation_labels.apply(self._standardize_label_df, axis=1)

        labels = pd.concat([training_labels, test_labels, validation_labels], axis=0).reset_index(drop=True)
        labels['dr'] = labels['dr'].apply(self._translate_labels)

        images['img_name'] = images.image_path.apply(lambda x: Path(x).stem)

        data = pd.merge(images, labels, left_on='img_name', right_on='image_path', how='inner').drop(
            columns=['img_name', 'image_path_y'])
        data = data.rename(columns={'image_path_x': 'image'})

        data['quality'] = None
        data = data.apply(self._make_quality_labels, axis=1)
        data['dme'] = None

        return self.clean_data(data)

    @staticmethod
    def _standardize_label_df(row):
        try:
            row['dr'] = row['DR_level']
            row['image_path'] = Path(row['image_path']).stem.split('\\')[-1]
        except KeyError:
            row['dr'] = row['UWF_DR_Levels']
            row['image_path'] = Path(row['image_id'])
        return row[['image_path', 'dr']]

    @staticmethod
    def _translate_labels(label) -> int:
        '''
        In our challenge, labels for diabetic retinopathy are:
        0: No DR
        1: DR

        In the DeepDRiD dataset, labels are:
        0: No DR
        1: Mild NPDR
        2: Moderate NPDR
        3: Severe NPDR
        4: PDR
        5: Bad image quality/indiscernible
        '''
        if label == 0:
            return 0
        elif label in [1, 2, 3, 4]:
            return 1
        elif label == 5:
            return 5
        else:
            raise ValueError(f"Invalid label in DeepDRiD dataset: {label}")

    @staticmethod
    def _make_quality_labels(row):
        '''
        When the 'dr' label in DeepDRiD is 5, the image quality is bad.
        Thus we add a new column "quality" that reflects our knowledge about image quality:
        0: Bad image quality
        1: Good image quality
        '''

        if int(row['dr']) == 5:
            row['dr'] = np.nan
            row['quality'] = 0
        elif int(row['dr']) in [0, 1, 2, 3, 4]:
            row['quality'] = 1
        else:
            raise ValueError(f"Invalid label in DeepDRiD dataset: {row['dr']}")
        return row


class CombinedDatasetStrategy(DatasetStrategy):
    def get_data(self):
        # Get data from both original strategies
        original_strategy = OriginalDatasetStrategy()
        deepdrid_strategy = DeepDridDatasetStrategy()

        original_data = original_strategy.get_data()
        deepdrid_data = deepdrid_strategy.get_data()

        # Ensure both datasets have the same columns
        columns_to_use = ['image', 'quality', 'dr', 'dme']
        original_data = original_data[columns_to_use]
        deepdrid_data = deepdrid_data[columns_to_use]

        # Combine the datasets
        combined_data = pd.concat([original_data, deepdrid_data], ignore_index=True)

        # Reset index and drop any potential duplicates
        combined_data = combined_data.reset_index(drop=True)

        # save csv
        #combined_data.to_csv(PROCESSED_DATA_DIR / 'combined_data.csv', index=False)

        return combined_data


class MiniDatasetStrategy(DatasetStrategy):
    '''
    This strategy may be used for checking if a model works at all. Overfit on few samples to check workability
    '''

    def get_data(self):
        # Get data from both original strategies
        original_strategy = OriginalDatasetStrategy()
        deepdrid_strategy = DeepDridDatasetStrategy()

        original_data = original_strategy.get_data()
        deepdrid_data = deepdrid_strategy.get_data()

        # Ensure both datasets have the same columns
        columns_to_use = ['image', 'quality', 'dr', 'dme']
        original_data = original_data[columns_to_use]
        deepdrid_data = deepdrid_data[columns_to_use]

        # Combine the datasets
        combined_data = pd.concat([original_data, deepdrid_data], ignore_index=True)

        # Reset index and drop any potential duplicates
        combined_data = combined_data.reset_index(drop=True)

        mini_data = combined_data.sample(frac=0.05, replace=False, random_state=42, axis=0)

        return mini_data


class TaskStrategy(ABC):
    @abstractmethod
    def apply(self, data):
        pass


class Task1Strategy(TaskStrategy):
    def apply(self, data):
        data = data.dropna(subset=['quality'])
        return data.drop(columns=['dr', 'dme']).reset_index(drop=True)


class Task2Strategy(TaskStrategy):
    def apply(self, data):
        data = data.dropna(subset=['dr'])
        return data.drop(columns=['quality', 'dme']).reset_index(drop=True)


class Task3Strategy(TaskStrategy):
    def apply(self, data):
        # Convert -1 values in the 'dme' column to NaN
        data['dme'] = data['dme'].replace(-1, np.nan)

        # Drop rows where 'dme' is NaN
        data = data.dropna(subset=['dme'])

        # Drop unnecessary columns and reset index
        return data.drop(columns=['quality', 'dr']).reset_index(drop=True)


class DatasetBuilder:
    def __init__(self, dataset_strategy: DatasetStrategy,
                 task_strategy: TaskStrategy, split_ratio: float = 0.8):
        self.dataset_strategy = dataset_strategy
        self.task_strategy = task_strategy
        self.split_ratio = split_ratio

    def build(self):
        data = self.dataset_strategy.get_data()
        data = self.task_strategy.apply(data)
        data["weight"] = sample_weights(data.iloc[:, 1])
        train_data, val_data = train_test_split(data, test_size=1 - self.split_ratio, random_state=42,
                                                stratify=data.iloc[:, 1])
        return train_data, val_data


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        weight = self.data.iloc[idx, 2]
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        img = cv2.imread(str(img_path))
        try:
            if self.transform:
                img = self.transform(img)
        except KeyError:
            if self.transform:
                img = self.transform(image=img)['image'] # when using Albumentations
                augmented = self.transform(image=img)
                img = augmented['image']

        return img, label, weight



class Loaders:
    def __init__(self, train_loader : DataLoader, val_loader : DataLoader):
        assert len(train_loader.dataset) > 0
        assert len(val_loader.dataset) > 0
        self.train_loader : DataLoader = train_loader
        self.val_loader : DataLoader = val_loader


class DatasetBuilder:
    def __init__(self, dataset_strategy: DatasetStrategy, 
                 task_strategy: TaskStrategy, 
                 batch_size: int, 
                 train_dataloader_shuffle: bool = True,
                 val_dataloader_shuffle: bool = False,
                 split_ratio: float = 0.8, n_folds: int = 1, 
                 train_set_transformations=None, val_set_transformations=None):
        
        self.dataset_strategy = dataset_strategy
        self.task_strategy = task_strategy
        self.split_ratio = split_ratio
        self.n_folds = n_folds
        self.batch_size = batch_size
        self.train_dataloader_shuffle = train_dataloader_shuffle
        self.val_dataloader_shuffle = val_dataloader_shuffle
        self.train_set_transformations = train_set_transformations
        self.val_set_transformations = val_set_transformations

    def build(self) -> List[Loaders]:

        loaders = []    

        data = self.dataset_strategy.get_data()
        data = self.task_strategy.apply(data)

        if self.n_folds == 1:
            train_data, val_data = train_test_split(data, test_size=1-self.split_ratio, random_state=42, stratify=data.iloc[:, 1])
            train_loader = DataLoader(CustomDataset(train_data, self.train_set_transformations), batch_size=self.batch_size, shuffle=self.train_dataloader_shuffle)
            val_loader = DataLoader(CustomDataset(train_data, self.train_set_transformations), batch_size=self.batch_size, shuffle=self.val_dataloader_shuffle)

            loaders.append(Loaders(train_loader, val_loader))

        elif self.n_folds > 1:
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

            for train_index, test_index in kf.split(data):
                train_data, val_data = data.iloc[train_index], data.iloc[test_index]
                train_loader = DataLoader(CustomDataset(train_data, self.train_set_transformations), batch_size=self.batch_size, shuffle=self.train_dataloader_shuffle)
                val_loader = DataLoader(CustomDataset(val_data, self.val_set_transformations), batch_size=self.batch_size, shuffle=self.val_dataloader_shuffle)
                loaders.append(Loaders(train_loader, val_loader))
        else:
            raise ValueError("n_folds must be a positive integer")

        assert len(loaders) == self.n_folds
        return loaders



def main():
    # Example: use combined dataset and task1
    dataset_strategy = CombinedDatasetStrategy()
    task_strategy = Task1Strategy()

    # Build dataset
    builder = DatasetBuilder(dataset_strategy, task_strategy, batch_size=16, split_ratio=0.8, n_folds=1)
    loaderssets = builder.build()

    for loadersset in loaderssets:
        print("Train dataset size:", len(loadersset.train_loader.dataset))


if __name__ == "__main__":
    main()

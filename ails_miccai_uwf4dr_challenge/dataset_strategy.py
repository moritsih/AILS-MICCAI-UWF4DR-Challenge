import enum
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from ails_miccai_uwf4dr_challenge.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, EXTERNAL_DATA_DIR
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from imblearn.over_sampling import RandomOverSampler

# add seeds
torch.manual_seed(42)
np.random.seed(42)

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

        data = pd.merge(images, labels, left_on='img_name', right_on='image_path', how='inner').drop(columns=['img_name', 'image_path_y'])
        data = data.rename(columns={'image_path_x': 'image'})

        data['quality'] = None
        data = data.apply(self._make_quality_labels, axis=1)
        data['dme'] = None

        return self.clean_data(data)

    def _standardize_label_df(self, row):
        try:
            row['dr'] = row['DR_level']
            row['image_path'] = Path(row['image_path']).stem.split('\\')[-1]
        except KeyError:
            row['dr'] = row['UWF_DR_Levels']
            row['image_path'] = Path(row['image_id'])
        return row[['image_path', 'dr']]

    def _translate_labels(self, label) -> int:
        if label in [0, 1, 2, 3, 4]:
            return 1
        elif label == 5:
            return 5
        else:            
            raise ValueError(f"Invalid label in DeepDRiD dataset: {label}")

    def _make_quality_labels(self, row):
        if int(row['dr']) == 5:
            row['dr'] = np.nan
            row['quality'] = 0
        elif int(row['dr']) in [0, 1, 2, 3, 4]:
            row['quality'] = 0
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
        
        return combined_data
    

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
    
class SplitStrategy(ABC):
    @abstractmethod
    def split(self, data):
        pass

class TrainValSplitStrategy(SplitStrategy):
    def __init__(self, split_ratio=0.8):
        self.split_ratio = split_ratio
    
    def split(self, data):
        train_data, val_data = train_test_split(data, test_size=1-self.split_ratio, random_state=42, stratify=data.iloc[:, 1])
        return train_data, val_data
    

class ResamplingStrategy(ABC):
    @abstractmethod
    def apply(self, data):
        pass

class RandomOverSamplingStrategy(ResamplingStrategy):
    def apply(self, data):
        X = data.drop(columns=[data.columns[1]])
        y = data[data.columns[1]]
        
        assert len(X) == len(y)
        assert len(y.unique()) > 1, "Resampling not needed for single class"

        print("\nOriginal class distribution:")
        print("Class 0:", (y == 0).sum())
        print("Class 1:", (y == 1).sum())
        print(y.value_counts(normalize=True))

        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)

        print("\nResampled class distribution:")
        print(pd.Series(y_resampled).value_counts(normalize=True))

        return pd.concat([X_resampled, y_resampled], axis=1)
    

class DatasetBuilder:
    def __init__(self, dataset_strategy: DatasetStrategy, task_strategy: TaskStrategy, 
                 split_strategy: SplitStrategy, resampling_strategy: ResamplingStrategy = None):
        self.dataset_strategy = dataset_strategy
        self.task_strategy = task_strategy
        self.split_strategy = split_strategy
        self.resampling_strategy = resampling_strategy
        
    def build(self):
        data = self.dataset_strategy.get_data()
        data = self.task_strategy.apply(data)
        if self.resampling_strategy:
            data = self.resampling_strategy.apply(data)
        return self.split_strategy.split(data)

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        img = cv2.imread(str(img_path))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, label

def main():

    # Example: use combined dataset and task1
    dataset_strategy = CombinedDatasetStrategy()
    task_strategy = Task1Strategy()
    
    split_strategy = TrainValSplitStrategy(split_ratio=0.8)
    resampling_strategy = RandomOverSamplingStrategy()

    # Build dataset
    builder = DatasetBuilder(dataset_strategy, task_strategy, split_strategy, resampling_strategy)
    train_data, val_data = builder.build()

    # Create PyTorch datasets
    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)

    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))


if __name__ == "__main__":
    main()

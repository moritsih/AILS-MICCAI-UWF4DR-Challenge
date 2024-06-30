from pathlib import Path
import typer
from loguru import logger
from tqdm import tqdm
from ails_miccai_uwf4dr_challenge.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, EXTERNAL_DATA_DIR
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import torch

app = typer.Typer()


class OriginalDataset:
    def __init__(self):
        self.data = self._merge_all_data_into_global_df()
        

    def get_data(self):
        return self.data
    

    def __len__(self):
        return len(self.data)
    

    def _merge_all_data_into_global_df(self):

        task1_dir = Path(RAW_DATA_DIR / 'Task 1 Image Quality Assessment')
        task23_dir = Path(RAW_DATA_DIR / 'Task 2 Referable DR and Task 3 DME')

        # get iterable of image paths for both tasks before combining
        imgs_task1 = Path(task1_dir / '1. Images' / '1. Training').glob('*.jpg')
        imgs_task23 = Path(task23_dir / '1. Images' / '1. Training').glob('*.jpg')

        imgs_task1 = pd.DataFrame(imgs_task1, columns=['image'])
        imgs_task23 = pd.DataFrame(imgs_task23, columns=['image'])

        # combine the two dataframes
        data = pd.concat([imgs_task1, imgs_task23], ignore_index=True)

        # label file only contains image names and their corresponding labels, but
        # in the label file, ALL images (from both tasks) are listed
        label_dir_task1 = Path(task1_dir / '2. Groundtruths')
        labels_task1 = pd.read_csv(label_dir_task1 / '1. Training.csv')

        label_dir_task23 = Path(task23_dir / '2. Groundtruths')
        labels_task23 = pd.read_csv(label_dir_task23 / '1. Training.csv')

        # merge labels according to image names
        labels = pd.merge(labels_task1, labels_task23, on='image', how='outer')

        # iterate through the rows in imgs
        for i in range(len(data)):
            # get the image name
            img_name = data['image'].iloc[i].name
            # get the row in labels that corresponds to the image
            row = labels[labels['image'] == img_name]
            # if the image is found in labels, add the image quality level to the new column
            if len(row) > 0:
                data.loc[i, 'quality'] = row['image quality level'].values[0]
                data.loc[i, 'dr'] = row['referable diabetic retinopathy'].values[0]
                data.loc[i, 'dme'] = row['diabetic macular edema'].values[0]

        data['quality'] = data['quality'].apply(lambda x: int(x) if not np.isnan(x) else x).astype("Int64")
        data['dr'] = data['dr'].apply(lambda x: int(x) if not np.isnan(x) else x).astype("Int64")
        data['dme'] = data['dme'].apply(lambda x: int(x) if not np.isnan(x) else x).astype("Int64")

        return data



class DeepDridDataset:
    def __init__(self):

        # find all images in the entire DeepDRiD dataset folder by recursively searching for all jpg files
        deepdrid_path = Path(EXTERNAL_DATA_DIR) / 'DeepDRiD' / 'ultra-widefield_images'
        images = pd.DataFrame(deepdrid_path.rglob('*.jpg'), columns=['image_path'])

        # read in all the different label files
        training_labels = pd.read_csv(deepdrid_path / 'ultra-widefield-training' / 'ultra-widefield-training.csv')
        test_labels = pd.read_excel(deepdrid_path / 'Online-Challenge3-Evaluation' / 'Challenge3_labels.xlsx')
        validation_labels = pd.read_csv(deepdrid_path / 'ultra-widefield-validation' / 'ultra-widefield-validation.csv')

        # standardize_label_df changes some column names so concatenating will work
        training_labels = training_labels.apply(self._standardize_label_df, axis=1)
        test_labels = test_labels.apply(self._standardize_label_df, axis=1)
        validation_labels = validation_labels.apply(self._standardize_label_df, axis=1)

        # concat all the label files into one large df
        labels = pd.concat([training_labels, test_labels, validation_labels], axis=0)
        labels = labels.reset_index(drop=True)

        # since labels in DeepDRiD are from 1-5 (see _translate_labels), we need to translate them to 0-1
        labels['dr'] = labels['dr'].apply(self._translate_labels)

        # for easy merging, we need to extract the image name from the image path
        images['img_name'] = images.image_path.apply(lambda x: Path(x).stem)

        # merge labels and images on img_name and img_path but drop the new img_name column
        data = pd.merge(images, labels, left_on='img_name', right_on='image_path', how='inner').drop(columns=['img_name', 'image_path_y'])
        
        # rename column
        data = data.rename(columns={'image_path_x': 'image'})

        # add a new column that reflects our knowledge about image quality
        data['quality'] = None
        data = data.apply(self._make_quality_labels, axis=1)

        # just to keep the same column names as the original dataset
        data['dme'] = np.ones_like(data['dr']) * -1

        self.data = data


    def __len__(self):
        return len(self.data)
    

    def get_data(self):
        return self.data
    

    def _make_quality_labels(self, row):
        '''
        When the 'dr' label in DeepDRiD is 5, the image quality is bad.
        Thus we add a new column "quality" that reflects our knowledge about image quality:
        0: Bad image quality
        1: Good image quality
        '''

        if int(row['dr']) == 5:
            row['quality'] = 0
        else:
            row['quality'] = 1
        
        return row


    def _standardize_label_df(self, row):
        
        try:
            row['dr'] = row['DR_level']
            row['image_path'] = Path(row['image_path']).stem.split('\\')[-1]
        except KeyError:
            row['dr'] = row['UWF_DR_Levels']
            row['image_path'] = Path(row['image_id'])

        #drop all other columns
        row = row[['image_path', 'dr']]
        return row
    


    def _translate_labels(self, label):
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
        else:
            return 5


class DatasetBuilder:

    '''
    Args:
    dataset: str
        'all' (default), 'original', 'deepdrid'
    task: str
        'full' (default), 'task1', 'task2', 'task3'

    Returns:
    Dataframe with the corresponding data

    Example:
    dataset = DatasetBuilder(dataset='all', task='task1')
    
    '''

    def __init__(self, dataset: str = 'all', task: str = 'full', split_ratio: float = 0.8):


        ############################################
        # FILTER FOR DATASET
        ############################################
        if dataset == 'all':
            self.original = OriginalDataset()
            self.deepdrid = DeepDridDataset()
            self.data = self._concat_datasets([self.original.get_data(), self.deepdrid.get_data()]) 

        elif dataset == 'original':
            self.data = OriginalDataset().get_data()

        elif dataset == 'deepdrid':
            self.data = DeepDridDataset().get_data()

        else:
            raise ValueError('Invalid dataset name. Please enter a valid dataset name (original, deepdrid, or all)')
        

        ############################################
        # FILTER FOR TASK
        ############################################
        if task == 'full':
            pass # simply returns self.data once get_data is called

        elif task in ['task1', '1', 'Task 1', 'Task1']:
            self.data = self.data.dropna(subset=['quality'])
            self.data = self.data.drop(columns=['dr', 'dme']).reset_index(drop=True)

        elif task in ['task2', '2', 'Task 2', 'Task2']:
            # filter out rows with null values in the 'dr' column and drop 
            # the 'dme' column and the quality column
            self.data = self.data.dropna(subset=['dr'])
            self.data = self.data.drop(columns=['quality', 'dme']).reset_index(drop=True)

        elif task in ['task3', '3', 'Task 3', 'Task3']:
            self.data = self.data.dropna(subset=['dme'])
            self.data = self.data.drop(columns=['quality', 'dr']).reset_index(drop=True)

        else:
            raise ValueError('Invalid task name. Please enter a valid task name (task1, task2, or task3)')
        

        self.train_data, self.val_data = self._split_data(self.data, split_ratio=split_ratio)

        
    def _split_data(self, data, split_ratio=0.8):
        '''
        Split the data into training and validation sets
        '''
        #stratify second column in df

        train_data, val_data = train_test_split(data, test_size=1-split_ratio, random_state=42, stratify=data.iloc[:, 1])

        return train_data, val_data
    
    def get_train_val(self):
        '''
        Returns the training and validation datasets
        '''

        return self.train_data, self.val_data

    def get_unsplit_dataframe(self):
        '''
        Returns the entire dataset as a pandas dataframe without splitting into training and validation sets
        '''
        return self.data

    def _concat_datasets(self, datasets: list):
        return pd.concat(datasets, ignore_index=True, axis=0).reset_index(drop=True)



class CustomDataset(Dataset):

    '''
    Use this class to create a custom dataset for PyTorch

    Example:
    dataset = DatasetBuilder(dataset='all', task='task1')
    train_data, val_data = dataset.get_train_val()

    train_dataset = CustomDataset(train_data, transform=transforms)
    val_dataset = CustomDataset(val_data, transform=transforms)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    '''

    def __init__(self, data, transform=None):

        self.transform = transform
        self.data = data

        print("Dataset length: ", len(self.data))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        # convert label to tensor and add an extra dimension so it can be used in the loss function
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        img = cv2.imread(str(img_path))

        # in the challenge description they say that they use BGR color for evaluation
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # DO NOT USE THIS LINE, JUST FOR CLARIFICATION

        if self.transform:
            img = self.transform(img)

        return img, label
    


@app.command()
def main():

    # example for how to use: run this file

    dataset = DatasetBuilder(dataset='original', task='task2')
    train_data, val_data = dataset.get_train_val()

    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)


if __name__ == "__main__":
    app()

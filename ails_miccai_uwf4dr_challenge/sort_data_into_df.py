from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np


class CombineOriginalData:

    '''
    Manages data for each task. 
    First, a merged dataframe holding all paths and labels is created.

    Depending on the task chosen, the dataset object will hold the corresponding data.
    To get the full dataframe, write "full". This is default.
    For Task 1, the dataset will only hold data with non-null quality labels. (434 images)
    For Task 2, the dataset will only hold data with non-null DR labels. (201 images)
    For Task 3, the dataset will only hold data with non-null DME labels. (167 images)

    To specify the task, pass in the task name as a string when creating the object.

    '''

    def __init__(self, task: str = 'full', transform=None):

        self.task = task
        self.transform = transform

        self.PROJECT_PATH = Path().resolve().parent # gets path of main project folder

        self.data = self._merge_all_data_into_global_df()

        if self.task == 'full':
            pass # simply returns self.data once get_data is called

        elif self.task in ['task1', '1', 'Task 1', 'Task1']:
            self.data = self.data.dropna(subset=['quality'])
            self.data = self.data.drop(columns=['dr', 'dme'])

        elif self.task in ['task2', '2', 'Task 2', 'Task2']:
            # filter out rows with null values in the 'dr' column and drop 
            # the 'dme' column and the quality column
            self.data = self.data.dropna(subset=['dr'])
            self.data = self.data.drop(columns=['quality', 'dme'])

        elif self.task in ['task3', '3', 'Task 3', 'Task3']:
            self.data = self.data.dropna(subset=['dme'])
            self.data = self.data.drop(columns=['quality', 'dr'])

        else:
            raise ValueError('Invalid task name. Please enter a valid task name (task1, task2, or task3)')
        

    def get_data(self):
        return self.data
    


    def __len__(self):
        return len(self.data)
    

    def _merge_all_data_into_global_df(self):

        task1_dir = Path(self.PROJECT_PATH / 'data' / 'raw' / 'Task 1 Image Quality Assessment')
        task23_dir = Path(self.PROJECT_PATH / 'data' / 'raw' / 'Task 2 Referable DR and Task 3 DME')

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



class MakeDataset(Dataset):

    def __init__(self, df: pd.DataFrame, transform=None):

        assert df.columns == 2, "Dataframe must have 2 columns. Make sure you only enter task-specific dataframes."
        
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        img = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 1]

        if self.transform:
            img = self.transform(img)

        return img, label
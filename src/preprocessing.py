import numpy as np
import pandas as pd
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display

class data_loader:

    def __init__(self):
        self.selected_dataset=None
        self.df=None
        self.X=None
        self.Y=None
    
    def chose_dataset(self):
        folder_path = Path("data")
        folder_name = folder_path.name
        file_paths = [f"{folder_name}/{f.name}" for f in folder_path.iterdir() if f.is_file()]
        dataset_dropdown = widgets.Dropdown(
            options=file_paths,
            description='Dataset:',
            value=file_paths[0]
            )
        display(dataset_dropdown)
        selected_dataset = dataset_dropdown.value
        self.selected_dataset= selected_dataset
        return Path(selected_dataset).stem


    def load(self):
        data_url = self.selected_dataset
        raw_df = pd.read_csv(data_url, sep=",")
        df=raw_df.copy()

        # Replace '?' with NaN
        if (df == '?').any().any():
            df.replace('?', np.nan, inplace=True)

        # Drop rows with NaN values
        if df.isnull().any().any():
            df.dropna(inplace=True)
            print('Number of rows with NaN values: ', raw_df.shape[0] - df.shape[0])

        # Convert data type from int64 to float
        for col in df.columns:
            if df[col].dtype == 'int64':
                df[col] = df[col].astype(float)
        
        self.df = df
        return df


        
    def split_features_target(self):
        
        # Prepare labels array Y
        self.Y = np.array(self.df.iloc[:, -1])

        # Prepare features array X
        self.X = np.array(self.df.iloc[:, :-1])

        unique, counts = np.unique(self.Y, return_counts=True)
        print('Class Distribution', dict(zip(unique, counts)))
        print('Class Distribution %', dict(zip(unique, np.round((counts/self.Y.shape[0])*100, 2))))
        
        return self.X, self.Y
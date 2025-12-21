## Feature Transforms
# get the df locally from hopsworks

import pandas as pd
import hopsworks
import os
import warnings
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from tqdm import tqdm_notebook as tqdm
from dataclasses import dataclass



warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
project = hopsworks.login(project="BirdUp", api_key_value=HOPSWORKS_API_KEY)


@dataclass
class DataSet:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    @property
    def train_tensor(self) -> torch.Tensor:
        return torch.tensor(self.train.to_numpy(), dtype=torch.float32)

    @property
    def val_tensor(self) -> torch.Tensor:
        return torch.tensor(self.val.to_numpy(), dtype=torch.float32)

    @property
    def test_tensor(self) -> torch.Tensor:
        return torch.tensor(self.test.to_numpy(), dtype=torch.float32)


def get_feature_group_as_df(feature_group_name: str = "birding", feature_group_version: int = 1):
    fs = project.get_feature_store()
    fg = fs.get_feature_group(
        name=feature_group_name, version=feature_group_version
    )
    df = fg.read()
    return df

def sort_by_date(df:pd.DataFrame, date_col:str='observation_date'):
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    return df

def true_false_to_01(df:pd.DataFrame):
    # Vectorized conversion is significantly faster than lambda mapping
    return df.astype({col: int for col in df.select_dtypes(include='bool').columns})

def encode_reigon(df:pd.DataFrame, region_col:str='region', premade_mapping:dict=None):
    # calculate means based on observation_count
    if premade_mapping is not None:
        region_mapping = premade_mapping
        df[region_col] = df[region_col].map(region_mapping)
        return df, region_mapping

    region_means = df.groupby(region_col)['observation_count'].mean().sort_values()
    
    # map to ordinal integers based on rank (0 = lowest avg count, 24 = highest)
    region_mapping = {region: i for i, region in enumerate(region_means.index)}
    df[region_col] = df[region_col].map(region_mapping)
    
    return df, region_mapping

def get_reigon_mapping():
    map ={'Dalsland': 0, 'Ångermanland': 1, 'Medelpad': 2, 'Jämtland': 3, 'Härjedalen': 4, 'Bohuslän': 5, 'Hälsingland': 6, 'Dalarna': 7, 'Norrbotten': 8, 'Blekinge': 9, 'Lappland': 10, 'Värmland': 11, 'Västmanland': 12, 'Gästrikland': 13, 'Småland': 14, 'Östergötland': 15, 'Närke': 16, 'Västerbotten': 17, 'Gotland': 18, 'Västergötland': 19, 'Halland': 20, 'Uppland': 21, 'Öland': 22, 'Södermanland': 23, 'Skåne': 24}
    return map

def drop_unused_columns(df:pd.DataFrame, drop_date:bool=True):
    unused_cols = ["observation_date", "time_observations_started", "weathercode", "obs_count_lag_1", "obs_count_lag_2", "obs_count_lag_3", "obs_count_lag_4", "obs_count_lag_5"]
    if not drop_date:
        unused_cols.remove("observation_date")
    df = df.drop(columns=unused_cols)
    return df

def birdcount_binarization(df:pd.DataFrame):
    df['bird_count_binary'] = (df['observation_count'] > 0).astype(int)
    df = df.drop(columns=['observation_count'])
    return df

def split_data(df:pd.DataFrame, train_size:float=0.8, val_size:float=0.1, shuffle = True) -> dict[str, DataSet]:
    # ------Helper function -----
    def show_dataset_end_dates(data_dict: dict):
        for bird, ds in data_dict.items():
            print(f"Bird: {bird}")
            # Check train, val, and test splits
            for split_name in ['train', 'val', 'test']:
                df = getattr(ds, split_name)
                
                if 'observation_date' in df.columns:
                    end_val = int(df['observation_date'].max()) 
                elif 'year' in df.columns:
                    # Fallback if date was dropped but year remains
                    end_val = f"Year: {df['year'].max()}"
                else:
                    end_val = "Date column not found"
                    
                print(f"  {split_name.capitalize()} End: {end_val}")

    # ---- Main function body ----            
    if shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    unique_birds=df["bird_type"].unique()
    # make dict with the keys train, val, test and empty dataframes as values
    ret_dict = {}
    for bird in unique_birds:
        bird_df = df[df["bird_type"]==bird]
        bird_df = bird_df.drop(columns=["bird_type"]).reset_index(drop=True)
        n = bird_df.shape[0]
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))
        
        bird_df_train = bird_df.iloc[:train_end]
        bird_df_val = bird_df.iloc[train_end:val_end]
        bird_df_test = bird_df.iloc[val_end:]
        
        ret_dict[bird] = DataSet(train=bird_df_train, val=bird_df_val, test=bird_df_test)
    if not shuffle:
        print("Dataset end dates (no shuffle):")
        show_dataset_end_dates(ret_dict)
    return ret_dict

def drop_x_negative_samples(df, drop_percentage=0.2):
    """
    Here we downsample the negative samples (bird_count_binary == 0) by a given percentage.
    This helps to balance the dataset, we have a large number of negative samples compared to positive ones.
    """
    neg = df[(df['bird_count_binary'] == 0)]
    pos = df[~((df['bird_count_binary'] == 0))]
    
    # Calculate number of samples to drop
    num_to_drop = int(len(neg) * drop_percentage)
    
    # Randomly sample the indices to keep
    neg_downsampled = neg.sample(n=len(neg) - num_to_drop, random_state=42)
    
    # Combine back and shuffle
    new_df = pd.concat([pos, neg_downsampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    return new_df


def make_feature_compatible(df:pd.DataFrame, reference_df:pd.DataFrame):
    # ensure the dataframe has all the feats and as many rows in the right order
    cols_to_add =['sighted_lag_5', 'sighted_lag_3', 'sighted_lag_4', 'sighted_lag_2', 'bird_count_binary', 'sighted_lag_1']
    for col in cols_to_add:
        if col not in df.columns:
            df[col] = 0
    # dupplicate all rows and have half of them bird_type = "whteag" and half = goleag
    df = pd.concat([df.assign(bird_type='whteag'), df.assign(bird_type='goleag')], ignore_index=True)
    # sort the colums like the reference df
    df = df[reference_df.columns]
    return df

def fill_sighted_lag(filled: pd.DataFrame, original: pd.DataFrame):
    filled['observation_date'] = pd.to_datetime(filled['observation_date'])
    original['observation_date'] = pd.to_datetime(original['observation_date'])
    
    # Combine datasets to ensure rolling windows carry over from history to future
    combined = pd.concat([original, filled]).drop_duplicates(['observation_date', 'bird_type', 'region'], keep='last')
    combined = combined.sort_values(['bird_type', 'region', 'observation_date'])

    # Shift bird_count_binary within each group to create lags
    for i in range(1, 6):
        combined[f'sighted_lag_{i}'] = combined.groupby(['bird_type', 'region'])['bird_count_binary'].shift(i)

    # Update filled dataframe with calculated lags
    filled = filled.drop(columns=[f'sighted_lag_{i}' for i in range(1, 6)])
    filled = filled.merge(
        combined[['observation_date', 'bird_type', 'region'] + [f'sighted_lag_{i}' for i in range(1, 6)]],
        on=['observation_date', 'bird_type', 'region'],
        how='left'
    )
    # convert sighted_lag columns to int
    for i in range(1, 6):
        filled[f'sighted_lag_{i}'] = filled[f'sighted_lag_{i}'].astype(int)
    return filled

def format_observation_date(df: pd.DataFrame):
    df['observation_date'] = pd.to_datetime(df['observation_date']).dt.strftime('%Y-%m-%d') # Changed: Formatted date to y-m-d string
    return df

def divide_ds(ds:pd.DataFrame):
    goleag_ds = ds[ds["bird_type"]=="goleag"].drop(columns=["bird_type"]).reset_index(drop=True)
    whteag_ds = ds[ds["bird_type"]=="whteag"].drop(columns=["bird_type"]).reset_index(drop=True)
    return goleag_ds, whteag_ds
    
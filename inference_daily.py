"""
Daily inference script for bird sighting predictions, does predictions with inference_daily and upploads to hopsworks feature store.
"""


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
from tqdm import tqdm
from dataclasses import dataclass

# our own
from Features.inference import * 
from Features.df_functions import features, to_hopsworks_df
from Models.Bird_percent import BirdPercentModel

warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
project = hopsworks.login(project="BirdUp", api_key_value=HOPSWORKS_API_KEY)

# helper functions 
def get_predictions_for_date(model: nn.Module, df: pd.DataFrame, date: int = 0) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        input_tensor = df_to_tensor(df, date)
        outputs = model(input_tensor).squeeze().numpy()
    return outputs



def df_to_tensor(df: pd.DataFrame, date: int = 0) -> torch.Tensor:
    # Get the specific date value
    unique_dates = df['observation_date'].unique()
    target_date = unique_dates[date]
    
    # Filter and sort by region to ensure consistent row order
    day_df = df[df['observation_date'] == target_date].sort_values('region')
    
    # Drop non-feature columns
    # We use errors='ignore' in case columns were already removed
    features = day_df.drop(columns=['observation_date', 'bird_count_binary', 'bird_type'], errors='ignore')
    
    return torch.tensor(features.values.astype(np.float32), dtype=torch.float32)

def instert_update_rolling(df: pd.DataFrame, predictions: np.ndarray, date: int = 0) -> pd.DataFrame:
    unique_dates = df['observation_date'].unique()
    binary_preds = (predictions >= 0.5).astype(int)
    
    # Get regions for the current date in the same order as df_to_tensor
    current_date = unique_dates[date]
    current_day_mask = df['observation_date'] == current_date
    regions = df[current_day_mask].sort_values('region')['region'].values
    
    # Map predictions to regions
    pred_map = dict(zip(regions, binary_preds))

    # Update current day bird_count_binary with the predictions
    df.loc[current_day_mask, 'bird_count_binary'] = df.loc[current_day_mask, 'region'].map(pred_map)
    
    # Propagate to future lags
    for i in range(1, 6):
        if date + i < len(unique_dates):
            future_date = unique_dates[date + i]
            # Update each region's lag column for the future date
            mask = df['observation_date'] == future_date
            df.loc[mask, f'sighted_lag_{i}'] = df.loc[mask, 'region'].map(pred_map)
            
    return df

def predict(bird_df: pd.DataFrame, model: nn.Module, sigmoid: float = 0.5) -> pd.DataFrame:
    region_mapping = {0: 'Dalsland', 1: 'Ångermanland', 2: 'Medelpad', 3: 'Jämtland', 4: 'Härjedalen', 5: 'Bohuslän', 6: 'Hälsingland', 7: 'Dalarna', 8: 'Norrbotten', 9: 'Blekinge', 10: 'Lappland', 11: 'Värmland', 12: 'Västmanland', 13: 'Gästrikland', 14: 'Småland', 15: 'Östergötland', 16: 'Närke', 17: 'Västerbotten', 18: 'Gotland', 19: 'Västergötland', 20: 'Halland', 21: 'Uppland', 22: 'Öland', 23: 'Södermanland', 24: 'Skåne'}
    
    bird_df = bird_df.copy()
    bird_df = bird_df.sort_values('observation_date')
    unique_dates = bird_df['observation_date'].unique()
    
    results = []
    
    for date_idx in range(len(unique_dates)):
        outputs = get_predictions_for_date(model, bird_df, date_idx)
        
        current_date = unique_dates[date_idx]
        current_day_df = bird_df[bird_df['observation_date'] == current_date].sort_values('region')
        region_ids = current_day_df['region'].values
        
        for r_id, prob in zip(region_ids, outputs):
            results.append({
                'observation_date': current_date,
                'region': region_mapping.get(r_id, r_id),
                'probability': float(prob),
                'prediction': 1 if prob >= sigmoid else 0 # Added thresholded value
            })
        
        bird_df = instert_update_rolling(bird_df, outputs, date_idx)
    
    return pd.DataFrame(results)




# Main inference function

def inference_daily():
    # get historical data: 
    hops_df = get_feature_group_as_df()
    hops_df = true_false_to_01(hops_df)
    hops_df = sort_by_date(hops_df)
    hops_df, REGION_MAPPING = encode_reigon(hops_df)
    hops_df = drop_unused_columns(hops_df, drop_date = False)
    hops_df = birdcount_binarization(hops_df)

    # get future features as weather df
    weather_df=features()
    weather_df = to_hopsworks_df(weather_df)
    weather_df = sort_by_date(weather_df)
    REGION_MAP = get_reigon_mapping()
    weather_df, __ = encode_reigon(weather_df,premade_mapping =REGION_MAP)
    weather_df = true_false_to_01(weather_df)
    weather_df = make_feature_compatible(weather_df, reference_df=hops_df)

    # merge historical and future features
    mergd = fill_sighted_lag(weather_df,format_observation_date(hops_df))
    goleag_ds, whteag_ds = divide_ds(mergd)
    
    # loading models 
    mr = project.get_model_registry()

    # 2. Retrieve model metadata from Hopsworks (e.g., version 1)
    hops_goldag_meta = mr.get_model("Goleag_model", version=1)
    hops_whteag_meta = mr.get_model("Whteag_model", version=1)
    # 3. Download the model files to a local directory
    g_path = hops_goldag_meta.download()
    w_path = hops_whteag_meta.download()

    # 4. Re-initialize your model classes (ensure in_features matches your training data)
    goldag_model_loaded = BirdPercentModel(in_features=22, hidden_layers=[32, 16, 1])
    whteag_model_loaded = BirdPercentModel(in_features=22, hidden_layers=[64, 32, 1])

    # 4b) LOAD WEIGHTS (this is what you’re missing)
    goldag_ckpt = os.path.join(g_path, "goldag_model")   # <-- match filename in Hopsworks
    whteag_ckpt = os.path.join(w_path, "whteag_model")   # <-- likely; verify via os.listdir

    goldag_state = torch.load(goldag_ckpt, map_location="cpu")
    whteag_state = torch.load(whteag_ckpt, map_location="cpu")

    # Handle both common checkpoint formats
    if isinstance(goldag_state, dict) and "model_state_dict" in goldag_state:
        goldag_state = goldag_state["model_state_dict"]
    if isinstance(whteag_state, dict) and "model_state_dict" in whteag_state:
        whteag_state = whteag_state["model_state_dict"]

    goldag_model_loaded.load_state_dict(goldag_state)
    whteag_model_loaded.load_state_dict(whteag_state)

    goldag_model_loaded.eval()
    whteag_model_loaded.eval()

    # 5. Inference 
    whteag_preds = predict(whteag_ds, whteag_model_loaded)
    whteag_preds["bird_type"] = "whteag"
    goleag_preds = predict(goleag_ds, goldag_model_loaded)
    goleag_preds["bird_type"] = "goleag"
    preds = pd.concat([whteag_preds, goleag_preds], ignore_index=True)
    preds = to_hopsworks_df(preds)

    # 6. uppload to hopsworks feature store
    # code for adding it to hopsworks feature store
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="bird_sighting_predictions",
        version=1,
        description="Predicted probabilities of bird sightings by region and date",
        primary_key=["observation_date", "region", "bird_type"])

    fg.insert(preds)
    print("Inference and upload completed.")


if __name__ == "__main__":
    inference_daily()
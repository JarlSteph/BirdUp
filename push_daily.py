from Features import df_functions
from hsfs.feature import Feature
import datetime
import requests
import pandas as pd
import hopsworks
import json
import re
import os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project="BirdUp")
fs = project.get_feature_store(name='birdup_featurestore')
def enforce_fg_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ints that MUST be ints
    for c in ["observation_count", "weathercode"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")

    # numeric continuous
    for c in ["temperature", "rain", "wind"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    # months: keep bool if your FG was created with bools; otherwise cast to int
    for c in [col for col in df.columns if col.startswith("month_")]:
        # if they are True/False already, keep them
        if df[c].dtype != "bool":
            df[c] = df[c].fillna(False).astype("bool")

    return df

def insert_daily_data():
    # retrive todays data: 
    birding_fg = fs.get_feature_group(name='birding',version=1,)
    daily_df = df_functions.daily()
    cols = [f.name for f in birding_fg.schema] 
    birding_fg.select(["region"]).read()
    df_prev = birding_fg.select(cols).read()
    daily_df = df_functions.to_hopsworks_df(daily_df)
    day_df_lag = df_functions.add_daily_lags_from_hopsworks_simple(daily_df, df_prev, k=5)
    # after day_df_lag is created, before insert
    day_df_lag["observation_count"] = (
        pd.to_numeric(day_df_lag["observation_count"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )

    day_df_lag["weathercode"] = (
        pd.to_numeric(day_df_lag["weathercode"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )
    day_df_lag = enforce_fg_types(day_df_lag)

    birding_fg.insert(day_df_lag, wait=True)
    return
insert_daily_data()
 
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

import hsfs, hopsworks, pandas as pd

print("Python:", __import__("sys").version)
print("hsfs:", hsfs.__version__)
print("hopsworks:", hopsworks.__version__)



birding_fg = fs.get_feature_group(name="birding", version=1)
print("\n--- birding fg ---")
print("name:", birding_fg.name, "version:", birding_fg.version)
print("storage type:", getattr(birding_fg, "storage_connector", None))
print("online_enabled:", getattr(birding_fg, "online_enabled", None))
print("time_travel_format:", getattr(birding_fg, "time_travel_format", None))
print("event_time:", getattr(birding_fg, "event_time", None))
print("primary_key:", getattr(birding_fg, "primary_key", None))
print("partition_key:", getattr(birding_fg, "partition_key", None))
print("precombine_key:", getattr(birding_fg, "hudi_precombine_key", None))

f0 = birding_fg.features[0]
print("\nfeature[0] type:", type(f0))
print("feature[0] name:", getattr(f0, "name", None))
print("feature[0] dict:", f0.to_dict() if hasattr(f0, "to_dict") else f0)

# Try reading a DIFFERENT feature group in same project (control test)
weather_fg = fs.get_feature_group(name="weather", version=1)
print("\n--- weather read test ---")
try:
    df_test = weather_fg.select([weather_fg.features[0].name]).read()
    print("weather read OK, shape:", df_test.shape)
except Exception as e:
    print("weather read FAILED:", repr(e))

print("\n--- birding read test ---")
try:
    df_test2 = birding_fg.select([f.name for f in birding_fg.features[:3]]).read()
    print("birding read OK, shape:", df_test2.shape)
except Exception as e:
    print("birding read FAILED:", repr(e))


def insert_daily_data():
    # retrive todays data: 
    birding_fg = fs.get_feature_group(name='birding',version=1,)
    daily_df = df_functions.daily()
    print(type(birding_fg.features[0]), birding_fg.features[0])

    features = birding_fg.features
    first = features[0]

    if hasattr(first, "name"):          # hsfs.feature.Feature
        cols = [f.name for f in features]
    else:                               # dict-like
        cols = [f["name"] for f in features]

    df_prev = birding_fg.select(cols).read()

    daily_df = df_functions.to_hopsworks_df(daily_df)
    day_df_lag = df_functions.add_daily_lags_from_hopsworks_simple(daily_df, df_prev, k=5)
    birding_fg.insert(day_df_lag, wait=True)
    return
insert_daily_data()
 
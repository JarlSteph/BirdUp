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


def insert_daily_data():
    # retrive todays data: 
    birding_fg = fs.get_feature_group(name='birding',version=1,)
    daily_df = df_functions.daily()
    print(type(birding_fg.features[0]), birding_fg.features[0])

    features = birding_fg.features
    first = features[0]

    cols = [f.name for f in birding_fg.schema] 
    print("all strings:", all(isinstance(c, str) for c in cols))
    print("hsfs:", hsfs.__version__)
    print("hopsworks:", hopsworks.__version__)

    print("types in cols:", sorted(set(type(c) for c in cols)))
    print("first 5 cols:", cols[:5])
    print("all strings:", all(isinstance(c, str) for c in cols))

    birding_fg.select(["region"]).read()



    df_prev = birding_fg.select(cols).read()

    daily_df = df_functions.to_hopsworks_df(daily_df)
    day_df_lag = df_functions.add_daily_lags_from_hopsworks_simple(daily_df, df_prev, k=5)
    birding_fg.insert(day_df_lag, wait=True)
    return
insert_daily_data()
 
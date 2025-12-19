from Features.df_functions import *
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
project = hopsworks.login()
fs = project.get_feature_store() 

def insert_daily_data():
    # retrive todays data: 
    birding_fg = fs.get_feature_group(
    name='birding',
    version=1,)
    daily_df = daily()
    birding_fg.insert(daily_df, wait=True)
    return
 

 
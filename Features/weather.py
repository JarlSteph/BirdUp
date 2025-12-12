"""
To extract the HISTORICAL weather data & to API call the new weather data
"""
import requests
from datetime import datetime, timedelta

def API_tomorrow_weather(lon:float, lat:float, days:int = 7) -> dict:
    """
    Function to get the weather data from Open-Meteo API for tomorrow's date
    Args:
        lon (float): Longitude of the location
        lat (float): Latitude of the location
        days (int): Number of days to forecast
    Returns:
        dict: Weather data for tomorrow 
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_mean,precipitation_sum,weather_code,wind_speed_10m_mean&timezone=Europe%2FBerlin&forecast_days={days}"
    response = requests.get(url)
    if response.status_code == 200: 
        data = response.json()
        return data
    else:
        raise Exception(f"API request failed with status code {response.status_code}")
    


def historical_weather_download(start_date:str, lon:float, lat:float) -> dict:
    TODATE = datetime.now().strftime("%Y-%m-%d")
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={TODATE}&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_mean,weather_code&timezone=Europe%2FBerlin"
    response = requests.get(url)
    if response.status_code == 200: 
        data = response.json()
        return data, ["OBSERVATION DATE", "TEMPERATURE", "RAIN", "WIND", "WEATHERCODE"]
    else:
        raise Exception(f"API request failed with status code {response.status_code}")

    
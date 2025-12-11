"""
To extract the HISTORICAL weather data & to API call the new weather data
"""
import requests


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
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_mean,precipitation_sum,visibility_mean,wind_speed_10m_mean&timezone=Europe%2FBerlin&forecast_days={days}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        raise Exception(f"API request failed with status code {response.status_code}")
    

    
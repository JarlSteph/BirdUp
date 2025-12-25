"""
To extract the HISTORICAL weather data & to API call the new weather data
"""
import requests
import pandas as pd
import time as pytime

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

def _session():
    s = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def _today_se_str() -> str:
    return datetime.now(ZoneInfo("Europe/Stockholm")).date().isoformat()


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
    #response = requests.get(url)
    s = _session()
    response = s.get(url, timeout=(10, 30))  # 30s read räcker för daily
    response.raise_for_status()
    if response.status_code == 200: 
        data = response.json()
        return data
    else:
        raise Exception(f"API request failed with status code {response.status_code}")
    


def historical_weather_download(start_date:str, lon:float, lat:float) -> dict:
    TODATE = _today_se_str() #CHANGED CHANGED for Git Actions
    #TODATE = datetime.now().strftime("%Y-%m-%d") OLD ONE
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={TODATE}&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_mean,weather_code&timezone=Europe%2FBerlin"
    
    print("WEATHER URL:", url)
    print("REQUEST START:", datetime.now())
    response = requests.get(url, timeout=(10, 60))  
    print("REQUEST END:", datetime.now(), "status", response.status_code)
    if response.status_code == 200: 
        data = response.json()
        return data, ["OBSERVATION DATE", "TEMPERATURE", "RAIN", "WIND", "WEATHERCODE"]
    else:
        raise Exception(f"API request failed with status code {response.status_code}")

COLS = ["OBSERVATION DATE", "TEMPERATURE", "RAIN", "WIND", "WEATHERCODE"]
_running_weather_means = {
    "temperature_2m_mean": [],
    "precipitation_sum": [],
    "wind_speed_10m_mean": [],
    "weather_code": [],
}
def _safe_daily_values(data: dict):
    """
    Plockar ut första dagens värden ur data["daily"] om det finns.
    Returnerar None om det saknas.
    """
    d = (data or {}).get("daily") or {}
    try:
        t = d.get("temperature_2m_mean")
        r = d.get("precipitation_sum")
        w = d.get("wind_speed_10m_mean")
        c = d.get("weather_code")
        if not (t and r and w and c):
            return None
        return float(t[0]), float(r[0]), float(w[0]), int(c[0])
    except Exception:
        return None

def _mean_fallback_as_full_json():
    """
    Skapar en 'data'-dict som liknar Open-Meteo-svaret (med daily),
    så din downstreamkod inte behöver ändras.
    """
    def mean_or(default, xs):
        return sum(xs) / len(xs) if xs else default

    t = mean_or(0.0, _running_weather_means["temperature_2m_mean"])
    r = mean_or(0.0, _running_weather_means["precipitation_sum"])
    w = mean_or(0.0, _running_weather_means["wind_speed_10m_mean"])
    c = int(round(mean_or(0, _running_weather_means["weather_code"])))

    return {
        "daily": {
            "temperature_2m_mean": [t],
            "precipitation_sum": [r],
            "wind_speed_10m_mean": [w],
            "weather_code": [c],
        }
    }

def historical_weather_download_actions(start_date: str, lon: float, lat: float,
                                        retries: int = 3, sleep_s: float = 2.0):
    """
    Drop-in ersättare: returnerar (data, COLS) exakt som din original-funktion.
    Archive -> Forecast -> mean fallback, utan att krascha.
    """
    TODATE = _today_se_str()

    archive_url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={TODATE}"
        "&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_mean,weather_code"
        "&timezone=Europe%2FBerlin"
    )

    # 1) ARCHIVE
    for attempt in range(1, retries + 1):
        try:
            print("WEATHER URL:", archive_url)
            print("REQUEST START:", datetime.now())
            resp = requests.get(archive_url, timeout=(10, 60))
            print("REQUEST END:", datetime.now(), "status", resp.status_code)
            resp.raise_for_status()

            data = resp.json()

            vals = _safe_daily_values(data)
            if vals is None:
                raise KeyError("daily missing/empty in archive response")

            t, r, w, c = vals
            _running_weather_means["temperature_2m_mean"].append(t)
            _running_weather_means["precipitation_sum"].append(r)
            _running_weather_means["wind_speed_10m_mean"].append(w)
            _running_weather_means["weather_code"].append(c)

            return data, COLS

        except Exception as e:
            print(f"ARCHIVE FAIL attempt {attempt}/{retries}: {type(e).__name__}: {e}")
            pytime.sleep(sleep_s * attempt)

    # 2) FORECAST
    forecast_url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_mean,weather_code"
        "&timezone=Europe%2FBerlin&forecast_days=7"
    )

    for attempt in range(1, retries + 1):
        try:
            print("FORECAST URL:", forecast_url)
            print("FORECAST START:", datetime.now())
            resp = requests.get(forecast_url, timeout=(10, 60))
            print("FORECAST END:", datetime.now(), "status", resp.status_code)
            resp.raise_for_status()

            data = resp.json()

            vals = _safe_daily_values(data)
            if vals is None:
                raise KeyError("daily missing/empty in forecast response")

            t, r, w, c = vals
            _running_weather_means["temperature_2m_mean"].append(t)
            _running_weather_means["precipitation_sum"].append(r)
            _running_weather_means["wind_speed_10m_mean"].append(w)
            _running_weather_means["weather_code"].append(c)

            # OBS: vi returnerar fortfarande hela data-jsonen
            return data, COLS

        except Exception as e:
            print(f"FORECAST FAIL attempt {attempt}/{retries}: {type(e).__name__}: {e}")
            pytime.sleep(sleep_s * attempt)

    # 3) Mean fallback: returnera en data-dict som har daily-nycklar
    print("WEATHER FALLBACK: returning running-mean defaults")
    data = _mean_fallback_as_full_json()
    return data, COLS
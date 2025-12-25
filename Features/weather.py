"""
To extract the HISTORICAL weather data & to API call the new weather data
"""
import requests
from datetime import datetime, timedelta
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


def _extract_daily_row(payload: dict, obs_date: str) -> dict:
    """
    Normaliserar Open-Meteo JSON (archive eller forecast) till en 1-rads dict.
    Returnerar {} om payload saknar expected struktur.
    """
    daily = payload.get("daily") or {}
    # Open-Meteo brukar ha listor där index 0 motsvarar start_date
    try:
        return {
            "OBSERVATION DATE": obs_date,
            "TEMPERATURE": (daily.get("temperature_2m_mean") or [None])[0],
            "RAIN": (daily.get("precipitation_sum") or [None])[0],
            "WIND": (daily.get("wind_speed_10m_mean") or [None])[0],
            "WEATHERCODE": (daily.get("weather_code") or [None])[0],
        }
    except Exception:
        return {}
def historical_weather_download_actions(
    start_date: str,
    lon: float,
    lat: float,
    last_known_row: dict | None = None,
    debug: bool = False,
):
    """
    Returnerar (row_dict, COLS) där row_dict alltid har nycklarna i COLS.
    Försöker: archive -> forecast -> last_known -> None.
    """
    s = _session()
    today = _today_se_str()
    COLS = ["OBSERVATION DATE", "TEMPERATURE", "RAIN", "WIND", "WEATHERCODE"]

    # 1) ARCHIVE
    archive_url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={today}"
        "&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_mean,weather_code"
        "&timezone=Europe%2FBerlin"
    )
    try:
        if debug:
            print("ARCHIVE URL:", archive_url)
        r = s.get(archive_url, timeout=(10, 30))
        r.raise_for_status()
        row = _extract_daily_row(r.json(), obs_date=start_date)
        return row, COLS
    except Exception as e:
        if debug:
            print("ARCHIVE FAILED:", repr(e))

    # 2) FORECAST (1 dag)
    forecast_url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_mean,precipitation_sum,weather_code,wind_speed_10m_mean"
        "&timezone=Europe%2FBerlin"
        "&forecast_days=1"
    )
    try:
        if debug:
            print("FORECAST URL:", forecast_url)
        r = s.get(forecast_url, timeout=(10, 30))
        r.raise_for_status()
        row = _extract_daily_row(r.json(), obs_date=start_date)
        return row, COLS
    except Exception as e:
        if debug:
            print("FORECAST FAILED:", repr(e))

    # 3) LAST KNOWN (från Hopsworks, skickas in)
    if last_known_row:
        row = {
            "OBSERVATION DATE": start_date,
            "TEMPERATURE": last_known_row.get("TEMPERATURE"),
            "RAIN": last_known_row.get("RAIN"),
            "WIND": last_known_row.get("WIND"),
            "WEATHERCODE": last_known_row.get("WEATHERCODE"),
        }
        return row, COLS

    # 4) Fallback None
    row = {"OBSERVATION DATE": start_date, "TEMPERATURE": None, "RAIN": None, "WIND": None, "WEATHERCODE": None}
    return row, COLS
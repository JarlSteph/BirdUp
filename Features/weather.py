"""
To extract the HISTORICAL weather data & to API call the new weather data
"""
import requests
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
def _mean_fallback():
    def mean_or_zero(x):
        return float(sum(x) / len(x)) if len(x) > 0 else 0.0

    return {
        "temperature_2m_mean": [mean_or_zero(_running_weather_means["temperature_2m_mean"])],
        "precipitation_sum":   [mean_or_zero(_running_weather_means["precipitation_sum"])],
        "wind_speed_10m_mean": [mean_or_zero(_running_weather_means["wind_speed_10m_mean"])],
        "weather_code":        [int(mean_or_zero(_running_weather_means["weather_code"]))],
    }


def _safe_weather_dict():
    # Minimal fallback så pd.DataFrame(weather_dict) alltid funkar
    # och merge_weather_data inte dör på NaN/None.
    return {
        "temperature_2m_mean": [0.0],
        "precipitation_sum": [0.0],
        "wind_speed_10m_mean": [0.0],
        "weather_code": [0],
    }

def historical_weather_download_actions(start_date: str, lon: float, lat: float,
                                        retries: int = 3, sleep_s: float = 2.0):
    """
    Drop-in ersättare för historical_weather_download.
    Returnerar samma tuple: (data, COLS)
    """

    # --- 1) ARCHIVE (samma som din) ---
    TODATE = _today_se_str()  # samma som du kör i Actions
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={TODATE}"
        "&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_mean,weather_code"
        "&timezone=Europe%2FBerlin"
    )

    for attempt in range(1, retries + 1):
        try:
            print("WEATHER URL:", url)
            print("REQUEST START:", datetime.now())
            response = requests.get(url, timeout=(10, 60))
            print("REQUEST END:", datetime.now(), "status", response.status_code)

            response.raise_for_status()
            data = response.json()
            _running_weather_means["temperature_2m_mean"].append(
                float(data["temperature_2m_mean"][0])
            )
            _running_weather_means["precipitation_sum"].append(
                float(data["precipitation_sum"][0])
            )
            _running_weather_means["wind_speed_10m_mean"].append(
                float(data["wind_speed_10m_mean"][0])
            )
            _running_weather_means["weather_code"].append(
                int(data["weather_code"][0])
            )

            return data, COLS

        except Exception as e:
            print(f"ARCHIVE FAIL attempt {attempt}/{retries}: {type(e).__name__}: {e}")
            time.sleep(sleep_s * attempt)

    # --- 2) FORECAST fallback (packas om till samma "enkla dict") ---
    f_url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_mean,weather_code"
        "&timezone=Europe%2FBerlin&forecast_days=7"
    )

    for attempt in range(1, retries + 1):
        try:
            print("FORECAST URL:", f_url)
            print("FORECAST START:", datetime.now())
            r = requests.get(f_url, timeout=(10, 60))
            print("FORECAST END:", datetime.now(), "status", r.status_code)

            r.raise_for_status()
            j = r.json()
            d = j.get("daily", {})

            # VIKTIGT: vi returnerar en dict som ser ut som archive-nycklarna,
            # inte hela forecast-jsonen.
            out = {
                "temperature_2m_mean": [ (d.get("temperature_2m_mean") or [0.0])[0] ],
                "precipitation_sum":   [ (d.get("precipitation_sum")   or [0.0])[0] ],
                "wind_speed_10m_mean": [ (d.get("wind_speed_10m_mean") or [0.0])[0] ],
                "weather_code":        [ (d.get("weather_code")        or [0])[0] ],
            }
            _running_weather_means["temperature_2m_mean"].append(
                    float(out["temperature_2m_mean"][0])
                )
            _running_weather_means["precipitation_sum"].append(
                float(out["precipitation_sum"][0])
            )
            _running_weather_means["wind_speed_10m_mean"].append(
                float(out["wind_speed_10m_mean"][0])
            )
            _running_weather_means["weather_code"].append(
                int(out["weather_code"][0])
            )
            return out, COLS

        except Exception as e:
            print(f"FORECAST FAIL attempt {attempt}/{retries}: {type(e).__name__}: {e}")
            time.sleep(sleep_s * attempt)

    # --- 3) sista fallback: krascha inte ---
    print("WEATHER FALLBACK: returning safe defaults")
    return _mean_fallback(), COLS
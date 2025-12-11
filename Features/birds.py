"""
To extract the HISTORICAL bird data & to API call the new bird data
"""

from dotenv import load_dotenv
import os
import requests



load_dotenv()
BIRD_API_KEY = os.getenv("BIRD_KEY")

def API_bird_data(bird_type: str = "goleag", location: str = "SE") -> dict:
    """
    Fetch recent bird observation data from the eBird API. (the birdcode is in the link eg):
    https://ebird.org/species/goleag --> goleag or whteag
    
    :param bird_type: Description
    :type bird_type: str
    :param location: Description
    :type location: str
    :return: Description
    :rtype: dict
    """



    url = f"https://api.ebird.org/v2/data/obs/{location}/recent/{bird_type}"

    payload={}
    headers = {
  'X-eBirdApiToken': BIRD_API_KEY
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    print(f"Bird data API response code: {response.status_code}")
    return response.json()





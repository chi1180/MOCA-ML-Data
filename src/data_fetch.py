import os

import pandas as pd
import requests

""" config """
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
CACHE_DATA_FILE = os.path.join(DATA_DIR, "points_cache.csv")
POINTS_API_ENDPOINT = "https://moca-jet.vercel.app/api/stops"


def load_data():
    # Load data from cache or fetch from API if cache is not available.
    if not os.path.exists(CACHE_DATA_FILE):
        print("Fetching data from API...")
        response = requests.get(POINTS_API_ENDPOINT)
        if response.status_code == 200:
            data = response.text
            with open(CACHE_DATA_FILE, "w") as f:
                f.write(data)
        else:
            raise Exception(f"Failed to fetch data from API: {response.status_code}")
    return pd.read_csv(CACHE_DATA_FILE)

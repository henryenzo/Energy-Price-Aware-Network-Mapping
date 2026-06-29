"""
    This is my first attempt at making a generator of stochastic parameters.
    For now, it just prints the thing but maybe later it'll save it directly into the json file or something like that.

    It will now also get the energy price from the ENTSO-E transparency platform API, and overall be a module for helper functions

    Created on June 23rd, 2026 by Enzo Henry
"""

import numpy as np
import json
import requests
import urllib3
import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError



def generate_random_energy_price(num_time_slots, num_nodes, min_price=1, max_price=5):
    """
    Generates a random energy price for each node at each time slot.\n
    The prices are uniformly distributed between min_price and max_price.
    """
    return json.dumps({f"i{node+1}": np.random.randint(min_price, max_price + 1, size=num_time_slots).tolist() for node in range(num_nodes)})

#print(generate_random_energy_price(10, 10))    # that's for model3


def get_api_token():
    """
    Fetches the API token from the 'api_token' file.
    """
    with open('api_token', 'r') as file:
        return file.read().strip()
    
TOKEN = get_api_token()

COUNTRIES = {
    "France": "FR",
    "Germany_Luxemburg": "DE_LU",
    "Belgium": "BE",
    "Spain": "ES",
    "Northern_Italy": "IT_NORD",
    "Netherlands": "NL",
}

def fetch_energy_prices(start=None, end=None, country_list=COUNTRIES.values(), csv_file_name="energy_prices.csv"):
    """
    Fetch day-ahead prices for a list of ENTSO-E countries/zones and export to CSV.\n 
 
    Arguments:\n
    - start, end : pd.Timestamp (tz-aware, ex: pd.Timestamp("2024-01-01", tz="Europe/Brussels"))
    - country_list : list[str] - zone codes for ENTSO-E, for example I took: ["FR", "DE_LU", "BE", "ES", "IT_NORD", "NL"]
    - csv_file_name : str - name of the output CSV file
 
    Returns:\n
    df: pd.DataFrame indexed by datetime, one column per country/bidding zone (for example Germany and Luxemburg are merged, Italy is separate)
    """
    # those three lines solve some SSL issues on MacOS, I haven't tried on Windows/Ubuntu
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    session = requests.Session()
    session.verify = False 
 
    client = EntsoePandasClient(api_key=TOKEN, session=session)
    data = {}

    if start or end is None:
        end = pd.Timestamp.now(tz="Europe/Brussels").floor("D")
        start = end - pd.Timedelta(days=1)

    for code in country_list:
        try:
            data[code] = client.query_day_ahead_prices(code, start=start, end=end)
        except NoMatchingDataError:
            print(f"No data available for {code}")
        except Exception as e:
            print(e)
 
    df = pd.DataFrame(data).sort_index()
    df.index.name = "datetime"
    df.to_csv(f"data/{csv_file_name}", sep=",", decimal=".", encoding="utf-8-sig")
    return df

def get_energy_prices_from_csv(csv_file_name="energy_prices.csv", time_slots=10, stride=1, country_list=None):
    """
    Fetches energy prices from a CSV file and returns a dictionary of prices for each country/zone.\n
 
    Arguments:\n
    - csv_file_name : str - name of the input CSV file
    - time_slots : int - number of time slots to return 
    - stride : int - step size for selecting time slots
    - country_list : list[str] - list of country/zone codes to include (default: all available in the CSV)
 
    Returns:\n
    prices_dict - keys are country/zone codes, values are lists of prices for the specified time slots.
    """
    try:
        df = pd.read_csv(f"data/{csv_file_name}", sep=",", decimal=".", encoding="utf-8-sig", index_col="datetime", parse_dates=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found, please run fetch_energy_prices(start, end, country_list) first")
    
    if country_list is None:
        country_list = df.columns.tolist()
    
    prices_dict = {}
    for code in country_list:
        assert code in df.columns, f"Column {code} not found"
        assert time_slots <= len(df)/4, f"Requested {time_slots} time slots, but only {len(df)/4} available"
        prices_dict[code] = df[code].iloc[::stride][:time_slots].tolist()
    
    return prices_dict
    




if __name__ == "__main__":
    start = pd.Timestamp("2026-06-25", tz="Europe/Brussels")
    end = pd.Timestamp("2026-06-26", tz="Europe/Brussels")
    country_list = list(COUNTRIES.values())
    #fetch_energy_prices(start, end, country_list)

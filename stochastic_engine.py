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
        if code == "GB":
            # Because of Brexit, we don't have access to the UK day-ahead prices for free anymore, so let's just consider a great Ireland :-)
            print(f"London and Glasgow are set to the same price as Dublin")
            code = "IE_SEM"
        try:
            data[code] = client.query_day_ahead_prices(code, start=start, end=end)
        except NoMatchingDataError:
            print(f"No data available for {code}")
        except Exception as e:
            print(e)
 
    df = pd.DataFrame(data).sort_index()
    df.index.name = "datetime"

    # Apparently Swiss (CH) and Great Ireland (IE_SEM) only have a resolution of 1 hour so let's just ffill
    df.ffill(inplace=True)

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
    
def json_parser(model_name: str, file_name = "test_models.json") -> dict: # same as in model_class.py
    """
        Helper function to parse the json file containing the test models and return the parameters of the model as a dict.\n
        Arguments:  
        - model_name: str, the name of the model to parse \n
        - file_name:  str, the name of the json file containing the test models (default: "test_models.json" is the one I used for tests) \n

        Returns: dict containing every paramters for the model (physGraph, sfc, availability, requirements etc)
    """

    with open(file_name, 'r') as file:
        test_models = json.load(file)["test_models"]
    #print(test_models["test_models"])
    for model in test_models:
        if model["model_name"] == model_name:
            return model["model_parameters"]
    raise ValueError(f"Model '{model_name}' does not exist")


def generate_virtual_graph(sfc_len=[7], mem_req=2, cpu_req=2, bw_req=2, req=None):
    # generates the virtual graph and the associated parameters, that are considered all the same first
    sfcs_num = len(sfc_len)
    if sfcs_num == 1:
        virtualGraph = {}
        for vnf in range(sfc_len[0]):
            virtualGraph[f"v{vnf+1}"] = [f"v{vnf+2}"]
        virtualGraph[f"v{vnf+1}"] = []
        json_entry = '"virtualGraph": ' + json.dumps(virtualGraph) + ',\n'

        # this sets all requirements to the same value if req is used
        if isinstance(req, int):
            mem_req = req
            cpu_req = req
            bw_req = req 
        elif req == "random" or "stochastic" or "rand":
            mem_req = np.random.randint(1, 5)
            cpu_req = np.random.randint(1, 5)
            bw_req = np.random.randint(1, 5)
        
        computing_req = {}
        memory_req = {}
        bandwidth_req = {} 
        for vnf in virtualGraph.keys():
            computing_req[vnf] = cpu_req
            memory_req[vnf] = mem_req
            bandwidth_req[vnf] = {virtualGraph[vnf][i]: bw_req for i in range(len(virtualGraph[vnf]))}
        json_entry = json_entry + '"computing_requirements": ' + json.dumps(computing_req) + ',\n' + '"memory_requirements": ' + json.dumps(memory_req) + ',\n' + '"bandwidth_requirements_dict": ' + json.dumps(bandwidth_req) + ',\n'
        return json_entry
    else:
        virtualGraph = {}
        for sfc in range(sfcs_num):
            for vnf in range(sfc_len[sfc]):
                virtualGraph[f"s{sfc+1}v{vnf+1}"] = [f"s{sfc+1}v{vnf+2}"]
            virtualGraph[f"s{sfc+1}v{vnf+1}"] = []
        json_entry = '"virtualGraph": ' + json.dumps(virtualGraph) + ',\n'

        # this sets all requirements to the same value if req is used
        if req:
            mem_req = req
            cpu_req = req
            bw_req = req 
        
        computing_req = {}
        memory_req = {}
        bandwidth_req = {} 
        for vnf in virtualGraph.keys():
            computing_req[vnf] = cpu_req
            memory_req[vnf] = mem_req
            bandwidth_req[vnf] = {virtualGraph[vnf][i]: bw_req for i in range(len(virtualGraph[vnf]))}
        json_entry = json_entry + '"computing_requirements": ' + json.dumps(computing_req) + ',\n' + '"memory_requirements": ' + json.dumps(memory_req) + ',\n' + '"bandwidth_requirements_dict": ' + json.dumps(bandwidth_req) + ',\n'
        return json_entry




if __name__ == "__main__":
    start = pd.Timestamp("2026-06-25", tz="Europe/Brussels")
    end = pd.Timestamp("2026-06-26", tz="Europe/Brussels")
    country_list = list(COUNTRIES.values())

    model = json_parser("nobel-eu")
    country_list = list(set(model["node_country"].values()))

    #fetch_energy_prices(country_list=country_list, csv_file_name="energy_prices.csv")

    #print(generate_virtual_graph([7]))

"""
    This is my first attempt at making a generator of stochastic parameters.
    For now, it just prints the thing but maybe later it'll save it directly into the json file or something like that.

    Created on June 23rd, 2026 by Enzo Henry
"""

import numpy as np
import json


def generate_energy_price(num_time_slots, num_nodes, min_price=1, max_price=5):
    """
    Generates a random energy price for each node at each time slot.\n
    The prices are uniformly distributed between min_price and max_price.
    """
    return json.dumps({f"i{node+1}": np.random.randint(min_price, max_price + 1, size=num_time_slots).tolist() for node in range(num_nodes)})

print(generate_energy_price(10, 10))    # that's for model3
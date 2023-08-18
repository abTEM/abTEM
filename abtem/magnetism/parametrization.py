from abtem.core.utils import get_data_path
import os
import json


def get_parameters():
    path = os.path.join(get_data_path(__file__), "lyon.json")

    with open(path, "r") as f:
        parameters = json.load(f)

    return parameters

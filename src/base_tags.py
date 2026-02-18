import os

import numpy as np
import pandas as pd

""" config """
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
BASE_TAGS_FILE = os.path.join(DATA_DIR, "base_tags.csv")


def load_base_tags():
    if not os.path.exists(BASE_TAGS_FILE):
        raise Exception(f"Base tags file not found: {BASE_TAGS_FILE}")
    df = pd.read_csv(BASE_TAGS_FILE)
    # Convert embedding column from string to numpy array
    df["embedding"] = df["embedding"].apply(
        lambda x: np.array(eval(x)) if isinstance(x, str) else x
    )
    return df

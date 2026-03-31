import os
import kagglehub
import pandas as pd
 
DATASET_HANDLE = "masud7866/solomon-vrptw-benchmark"
 
 
def get_solomon_path() -> str:
    return kagglehub.dataset_download(DATASET_HANDLE)
 
 
def load_instance(name: str) -> pd.DataFrame:
    path = get_solomon_path()
 
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.startswith(name) and f.endswith(".csv"):
                return pd.read_csv(os.path.join(root, f))
 
    raise FileNotFoundError(f"Instance '{name}' not found in {path}")
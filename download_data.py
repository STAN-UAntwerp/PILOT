import pathlib
import joblib

from benchmark_config import UCI_DATASET_IDS
from benchmark_util import load_data

DATAFOLDER = pathlib.Path(__file__).parent / "Data"
DATAFOLDER.mkdir(exist_ok=True)

for repo_id in UCI_DATASET_IDS:
    dataset = load_data(repo_id)
    joblib.dump(dataset, DATAFOLDER / f"{repo_id}.pkl")

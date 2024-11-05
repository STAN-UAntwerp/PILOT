import pathlib
import joblib

from benchmark_config import UCI_DATASET_IDS, IGNORE_COLUMNS, LOGTRANSFORM_TARGET
from benchmark_util import load_data

DATAFOLDER = pathlib.Path(__file__).parent / "Data"
DATAFOLDER.mkdir(exist_ok=True)

for repo_id in UCI_DATASET_IDS:
    dataset = load_data(
        repo_id,
        ignore_feat=IGNORE_COLUMNS.get(repo_id),
        logtransform_target=(repo_id in LOGTRANSFORM_TARGET),
    )
    joblib.dump(dataset, DATAFOLDER / f"{repo_id}.pkl")

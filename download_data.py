import pathlib
import joblib

from benchmark_config import UCI_DATASET_IDS, IGNORE_COLUMNS, LOGTRANSFORM_TARGET, IGNORE_PMLB
from benchmark_util import load_data

from pmlb import regression_dataset_names

DATAFOLDER = pathlib.Path(__file__).parent / "Data"
DATAFOLDER.mkdir(exist_ok=True)

for repo_id in UCI_DATASET_IDS:
    filename = DATAFOLDER / f"uci_{repo_id}.pkl"
    if filename.exists():
        print(f"{filename.name} already exists, skipping")
        continue
    dataset = load_data(
        repo_id,
        ignore_feat=IGNORE_COLUMNS.get(repo_id),
        logtransform_target=(repo_id in LOGTRANSFORM_TARGET),
        use_download=False
    )
    
    print(f"Storing data in {filename}")
    joblib.dump(dataset, filename)
    
for repo_id in regression_dataset_names:
    if repo_id in IGNORE_PMLB:
        print(f"Skipping {repo_id}")
        continue
    filename = DATAFOLDER / f"pmlb_{repo_id.split('_')[0]}.pkl"
    if filename.exists():
        print(f"{filename.name} already exists, skipping")
        continue
    dataset = load_data(
        repo_id,
        ignore_feat=IGNORE_COLUMNS.get(repo_id),
        logtransform_target=(repo_id in LOGTRANSFORM_TARGET),
        use_download=False,
        kind='pmlb'
    )
    
    print(f"Storing data in {filename}")
    joblib.dump(dataset, filename)

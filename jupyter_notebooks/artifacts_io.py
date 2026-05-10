
import os
import json
import glob
import time
from typing import Dict, Any
import pandas as pd

def _ts():
    return time.strftime("%Y%m%d_%H%M%S")

def save_artifacts(dfs: Dict[str, pd.DataFrame],
                   meta: Dict[str, Any] = None,
                   out_dir: str = "artifacts") -> str:
    """
    Save one or more DataFrames to a versioned folder as Parquet, plus a meta.json.
    Returns the path to the created folder.
    """
    if meta is None:
        meta = {}
    os.makedirs(out_dir, exist_ok=True)
    run_dir = os.path.join(out_dir, f"run_{_ts()}")
    os.makedirs(run_dir, exist_ok=True)

    meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_tables": len(dfs),
            **meta}

    # Save dataframes
    for name, df in dfs.items():
        safe = name.replace(" ", "_").replace("/", "_")
        df.to_parquet(os.path.join(run_dir, f"{safe}.parquet"), index=False)

    # Save metadata
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return run_dir

def _list_runs(out_dir: str = "artifacts"):
    paths = sorted(glob.glob(os.path.join(out_dir, "run_*")))
    return [p for p in paths if os.path.isdir(p)]

def latest_run(out_dir: str = "artifacts") -> str:
    runs = _list_runs(out_dir)
    if not runs:
        raise FileNotFoundError(f"No runs found in '{out_dir}'.")
    return runs[-1]

def load_artifacts(out_dir: str = "artifacts"):
    """
    Load all Parquet tables and meta.json from the latest run.
    Returns (tables: Dict[str, pd.DataFrame], meta: Dict[str, Any], run_dir: str)
    """
    run_dir = latest_run(out_dir)
    tables = {}
    for f in os.listdir(run_dir):
        if f.endswith(".parquet"):
            key = f[:-8]  # strip ".parquet"
            tables[key] = pd.read_parquet(os.path.join(run_dir, f))
    meta_path = os.path.join(run_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return tables, meta, run_dir

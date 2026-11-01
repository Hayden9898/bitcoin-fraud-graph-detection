# Loads the raw Elliptic CSVs into pandas: one row per transaction, plus the edge list.
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "elliptic" / "raw"

# Elliptic encodes illicit as "1", licit as "2"; we use 1 = illicit, 0 = licit, -1 = unlabeled
LABEL_MAP = {"1": 1, "2": 0, "unknown": -1}


def load_elliptic(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(raw_dir / "elliptic_txs_features.csv", header=None)
    features.columns = ["txId", "timestep"] + [f"f{i}" for i in range(features.shape[1] - 2)]

    classes = pd.read_csv(raw_dir / "elliptic_txs_classes.csv", dtype={"class": str})
    classes["label"] = classes["class"].map(LABEL_MAP)

    txs = features.merge(classes[["txId", "label"]], on="txId", validate="one_to_one")
    edges = pd.read_csv(raw_dir / "elliptic_txs_edgelist.csv")
    return txs, edges

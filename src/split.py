# Temporal split: train on the past, validate on the next weeks, test on the future.
import pandas as pd

TRAIN_LAST_STEP = 29
VAL_LAST_STEP = 34


def temporal_masks(txs: pd.DataFrame) -> dict[str, pd.Series]:
    labeled = txs["label"] != -1
    step = txs["timestep"]
    return {
        "train": labeled & (step <= TRAIN_LAST_STEP),
        "val": labeled & step.between(TRAIN_LAST_STEP + 1, VAL_LAST_STEP),
        "test": labeled & (step > VAL_LAST_STEP),
    }

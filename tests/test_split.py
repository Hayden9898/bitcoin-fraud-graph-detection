import pandas as pd

from src.split import temporal_masks


def test_masks_split_on_week_boundaries_and_skip_unlabeled():
    txs = pd.DataFrame(
        {
            "timestep": [28, 29, 29, 30, 34, 34, 35, 36],
            "label": [0, 1, -1, 1, 0, -1, 1, 0],
        }
    )
    masks = temporal_masks(txs)

    assert txs[masks["train"]].index.tolist() == [0, 1]
    assert txs[masks["val"]].index.tolist() == [3, 4]
    assert txs[masks["test"]].index.tolist() == [6, 7]

import numpy as np

from src.metrics import illicit_metrics


def test_counts_and_scores_treat_illicit_as_positive():
    y_true = np.array([1, 1, 0, 0, 0])
    prob = np.array([0.9, 0.2, 0.7, 0.1, 0.3])

    metrics = illicit_metrics(y_true, prob, threshold=0.5)

    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 1, 1)
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0.5

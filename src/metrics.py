# Illicit-class metrics shared by every model, plus the threshold sweep run on val.
import numpy as np
from sklearn.metrics import average_precision_score


def illicit_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = prob >= threshold
    tp = int(np.sum(pred & (y_true == 1)))
    fp = int(np.sum(pred & (y_true == 0)))
    fn = int(np.sum(~pred & (y_true == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pr_auc": average_precision_score(y_true, prob),
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    thresholds = np.arange(0.01, 1.0, 0.01)
    f1s = [illicit_metrics(y_true, prob, t)["f1"] for t in thresholds]
    return round(float(thresholds[np.argmax(f1s)]), 2)

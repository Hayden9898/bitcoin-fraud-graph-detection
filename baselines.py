# Step 3: logistic regression and XGBoost baselines on local vs all features.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data import load_elliptic
from src.metrics import best_f1_threshold, illicit_metrics
from src.split import temporal_masks

# f0-f92 describe the transaction itself; f93-f164 are Elliptic's one-hop neighbour
# summaries, so "all" already carries some graph information
FEATURE_SETS = {
    "local": [f"f{i}" for i in range(93)],
    "all": [f"f{i}" for i in range(165)],
}

txs, _ = load_elliptic()
masks = temporal_masks(txs)
# y and X are keyed by split: "train", "val", "test"
y = {split: txs.loc[mask, "label"].to_numpy() for split, mask in masks.items()}
licit_per_illicit = (y["train"] == 0).sum() / (y["train"] == 1).sum()

rows = []
for feature_set, columns in FEATURE_SETS.items():
    X = {split: txs.loc[mask, columns] for split, mask in masks.items()}

    # The scaler sits inside the pipeline so it is fit on train only
    logreg = make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
    )
    logreg.fit(X["train"], y["train"])

    xgb = XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=licit_per_illicit,
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        early_stopping_rounds=50,
    )
    # Trees are built from train only; val log loss just decides when to stop adding them
    xgb.fit(X["train"], y["train"], eval_set=[(X["val"], y["val"])], verbose=False)

    for model_name, model in {"LR": logreg, "XGB": xgb}.items():
        # Column 1 is P(illicit)
        val_prob = model.predict_proba(X["val"])[:, 1]
        test_prob = model.predict_proba(X["test"])[:, 1]
        # Val only chooses the threshold; test is scored once at each threshold
        for threshold in (0.5, best_f1_threshold(y["val"], val_prob)):
            metrics = illicit_metrics(y["test"], test_prob, threshold)
            rows.append({"model": model_name, "features": feature_set, **metrics})

print(f"XGB trees kept after early stopping (all features): {xgb.best_iteration + 1}")
print(pd.DataFrame(rows).round(3).to_string(index=False))

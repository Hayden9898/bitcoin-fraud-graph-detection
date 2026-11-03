# Step 5: GraphSAGE on the full transaction graph, local vs all features.
# Every node passes messages; only labeled train nodes are graded.
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch_geometric.nn import SAGEConv

from src.data import FEATURE_SETS, load_elliptic
from src.metrics import best_f1_threshold, illicit_metrics
from src.split import temporal_masks

torch.manual_seed(0)

txs, edges = load_elliptic()
masks = {split: torch.tensor(mask.to_numpy()) for split, mask in temporal_masks(txs).items()}
y = torch.tensor(txs["label"].to_numpy())

# The edge list names transactions by txId; PyG wants row positions in the feature table
row_of = pd.Series(np.arange(len(txs)), index=txs["txId"])
edge_index = torch.tensor(
    np.stack([row_of[edges["txId1"]].to_numpy(), row_of[edges["txId2"]].to_numpy()])
)
# Money flows one way, but suspicion should travel both ways
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        # Each SAGEConv: W1 · self + W2 · mean(neighbours)
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        # One raw score (logit) per node; sigmoid turns it into P(illicit)
        return self.conv2(h, edge_index).squeeze(-1)


def predict(model: GraphSAGE, x: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x, edge_index)).numpy()


def train(x: torch.Tensor, epochs: int = 200) -> GraphSAGE:
    model = GraphSAGE(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_y = y[masks["train"]].float()
    best_val_pr_auc, best_state = -1.0, None

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = F.binary_cross_entropy_with_logits(logits[masks["train"]], train_y)
        loss.backward()
        optimizer.step()

        # Early stopping: keep the weights from the epoch with the best val PR-AUC
        val_prob = predict(model, x)[masks["val"]]
        val_pr_auc = average_precision_score(y[masks["val"]], val_prob)
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc, best_state = val_pr_auc, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model


rows = []
for feature_set, columns in FEATURE_SETS.items():
    x = torch.tensor(txs[columns].to_numpy(), dtype=torch.float32)
    # Scale with train statistics only, same rule as the logreg pipeline
    mean, std = x[masks["train"]].mean(0), x[masks["train"]].std(0) + 1e-8
    x = (x - mean) / std

    prob = predict(train(x), x)
    val_prob, test_prob = prob[masks["val"]], prob[masks["test"]]
    y_val, y_test = y[masks["val"]].numpy(), y[masks["test"]].numpy()
    for threshold in (0.5, best_f1_threshold(y_val, val_prob)):
        metrics = illicit_metrics(y_test, test_prob, threshold)
        rows.append({"model": "GraphSAGE", "features": feature_set, **metrics})

print(pd.DataFrame(rows).round(3).to_string(index=False))

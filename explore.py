# Step 1: class balance, unlabeled share, and nodes per timestep.
from src.data import load_elliptic

txs, edges = load_elliptic()
print(f"{len(txs):,} transactions, {len(edges):,} edges, {txs.shape[1] - 3} features")

counts = txs["label"].map({1: "illicit", 0: "licit", -1: "unlabeled"}).value_counts()
print(counts.to_frame("count").assign(share=lambda c: (c["count"] / len(txs)).round(3)))

labeled = txs[txs["label"] != -1]
print(f"illicit share of labeled: {labeled['label'].mean():.3f}")

per_step = txs.groupby("timestep")["label"].agg(
    nodes="size",
    labeled=lambda s: (s != -1).sum(),
    illicit=lambda s: (s == 1).sum(),
)
print(per_step.to_string())

# bitcoin-fraud-graph-detection

GNN fraud benchmark on the Elliptic Bitcoin dataset. Train on timesteps 1–34, test on 35–49.

## Setup

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p data/elliptic/raw && cd data/elliptic/raw
for f in features classes edgelist; do curl -fLO https://data.pyg.org/datasets/elliptic/elliptic_txs_$f.csv.zip && unzip -q elliptic_txs_$f.csv.zip && rm elliptic_txs_$f.csv.zip; done
cd - && .venv/bin/python explore.py
```

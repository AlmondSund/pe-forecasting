#!/usr/bin/env python3
"""Train a scikit-learn logistic regression on PE features CSV.

Expected CSV columns: label, pe, wpe, mpe_tau1, ... optionally start_sec.

Usage:
    PYTHONPATH=src python3 bin/train_pe.py data/features.csv
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from permutation_entropy.models import LogisticConfig, evaluate, predict_proba, train_logistic


def load_features(path: Path):
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("Empty CSV.")
    feature_keys = [k for k in rows[0].keys() if k not in {"label", "start_sec"}]
    X, y, starts = [], [], []
    for row in rows:
        y.append(int(row["label"]))
        starts.append(float(row.get("start_sec", 0.0)))
        X.append([float(row[k]) for k in feature_keys])
    return X, y, starts, feature_keys


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    csv_path = Path(sys.argv[1])
    X, y, starts, feature_keys = load_features(csv_path)

    stratify = y if len(set(y)) > 1 else None
    X_tr, X_val, y_tr, y_val, starts_tr, starts_val = train_test_split(
        X, y, starts, test_size=0.2, random_state=42, stratify=stratify
    )
    model = train_logistic(X_tr, y_tr, LogisticConfig(class_weight="balanced"))
    acc, auc = evaluate(model, X_val, y_val)
    val_probs = predict_proba(model, X_val)
    full_probs = predict_proba(model, X)

    print(f"Features: {feature_keys}")
    print(f"Validation accuracy: {acc:.3f}, AUC: {auc:.3f}")
    print(f"Validation positives: {sum(y_val)}, negatives: {len(y_val) - sum(y_val)}")

    if any(starts):
        plt.figure(figsize=(10, 3))
        plt.plot(starts, full_probs, lw=0.8, label="p(alert)")
        plt.scatter(starts_val, val_probs, color="orange", s=12, label="val")
        plt.axhline(0.5, color="gray", ls="--", lw=0.8, label="threshold 0.5")
        plt.xlabel("start_sec")
        plt.ylabel("Probability")
        plt.title(f"Acc={acc:.3f}, AUC={auc:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig("pe_probs.png", dpi=150)
        print("Saved plot: pe_probs.png")


if __name__ == "__main__":
    main()

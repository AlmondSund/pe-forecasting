"""Ingest seismic MiniSEED and export permutation-entropy feature CSV.

This is a thin, documented script-like module that you can run directly:

    PYTHONPATH=src python -m data_features.ingest \
        --mseed data/raw.mseed \
        --window 30 --hop 5 \
        --output data/features.csv

Requirements: `obspy` for MiniSEED IO, plus `numpy`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

try:  # optional dependency
    import obspy  # type: ignore
except Exception as exc:  # pragma: no cover - optional
    raise ImportError("obspy is required to read MiniSEED. Install with `pip install obspy`." ) from exc

from permutation_entropy.features import multiscale_pe, permutation_entropy, sliding_windows, weighted_permutation_entropy


def load_mseed(path: Path) -> tuple[np.ndarray, float]:
    """Load MiniSEED into (samples, sample_rate)."""
    st = obspy.read(str(path))
    tr = st[0]
    return tr.data.astype(float), float(tr.stats.sampling_rate)


def window_series(series: Sequence[float], sample_rate: float, window_s: float, hop_s: float) -> List[np.ndarray]:
    """Slice a series into overlapping windows (seconds-based)."""
    window = int(window_s * sample_rate)
    hop = max(1, int(hop_s * sample_rate))
    return [np.asarray(series[start : start + window]) for start in range(0, max(0, len(series) - window + 1), hop)]


def features_for_window(win: Sequence[float], m: int = 4, taus: Iterable[int] = (1, 2, 3, 4)) -> dict:
    row = {
        "pe": permutation_entropy(win, m=m, tau=1, normalize=True),
        "wpe": weighted_permutation_entropy(win, m=m, tau=1, normalize=True),
    }
    for tau, val in multiscale_pe(win, m=m, taus=taus):
        row[f"mpe_tau{tau}"] = val
    return row


def save_csv(rows: List[dict], path: Path):
    if not rows:
        raise ValueError("No feature rows to save.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process(mseed_path: Path, out_csv: Path, window_s: float, hop_s: float, m: int, taus: Iterable[int]):
    samples, sr = load_mseed(mseed_path)
    windows = window_series(samples, sr, window_s, hop_s)
    rows = []
    for i, win in enumerate(windows):
        hop = max(1, int(hop_s * sr))
        start_sec = i * hop / sr  # actual window start in seconds
        row = {"start_sec": start_sec, **features_for_window(win, m=m, taus=taus)}
        rows.append(row)
    save_csv(rows, out_csv)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract PE features from MiniSEED and save to CSV.")
    p.add_argument("--mseed", required=True, type=Path, help="Path to MiniSEED file")
    p.add_argument("--output", required=True, type=Path, help="Output CSV path")
    p.add_argument("--window", type=float, default=30.0, help="Window length in seconds")
    p.add_argument("--hop", type=float, default=5.0, help="Hop length in seconds")
    p.add_argument("--m", type=int, default=4, help="Embedding dimension m")
    p.add_argument("--taus", type=int, nargs="+", default=[1, 2, 3, 4], help="List of delay values")
    return p.parse_args()


def main():
    args = parse_args()
    process(args.mseed, args.output, args.window, args.hop, args.m, args.taus)
    print(f"Saved features -> {args.output}")


if __name__ == "__main__":
    main()

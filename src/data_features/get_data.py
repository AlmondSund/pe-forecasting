"""Backward-compatible shim for feature extraction.

The original placeholder `get-data.py` is now implemented as a thin wrapper
around `data_features.ingest`. Import `process` or run `python -m data_features.ingest`
for the full CLI.
"""

from __future__ import annotations

from .ingest import load_mseed, window_series, features_for_window, save_csv, process, main

__all__ = [
    "load_mseed",
    "window_series",
    "features_for_window",
    "save_csv",
    "process",
    "main",
]

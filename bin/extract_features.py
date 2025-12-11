#!/usr/bin/env python3
"""Extract permutation-entropy features from a MiniSEED file to CSV.

Usage:
    PYTHONPATH=src python3 bin/extract_features.py \
        --mseed data/raw.mseed \
        --output data/features.csv \
        --window 30 --hop 5
"""

import sys
from data_features.ingest import main


if __name__ == "__main__":
    sys.exit(main())

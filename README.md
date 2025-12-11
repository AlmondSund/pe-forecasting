# pe-forecasting

From-scratch implementation of permutation-entropy feature extraction and a scikit-learn logistic forecaster for volcanic eruption (or tremor) alerting. Everything here is lightweight: pure-Python feature calculators plus a small training helper built on `scikit-learn`.

## Layout
- `src/permutation_entropy/` — permutation entropy (PE), weighted PE (WPE), multiscale PE (MPE), sliding-window helpers, and scikit-learn logistic model.
- `src/data_features/` — MiniSEED → CSV feature extraction (`ingest.py` CLI; `get_data.py` shim).
- `bin/train_pe.py` — train/evaluate on a CSV of features (`label,pe,wpe,mpe_tau1,...`).
- `bin/extract_features.py` — CLI wrapper over `data_features.ingest` to turn MiniSEED into feature CSVs.
- `docs/` — presentation slides.
- `data/` — place your MiniSEED/CSV feature files here (not versioned).

## Quickstart
```bash
cd /home/marti/Downloads/pe-forecasting
python3 -m venv .venv && source .venv/bin/activate
pip install scikit-learn numpy matplotlib obspy

# Extract features from MiniSEED
PYTHONPATH=src python3 bin/extract_features.py --mseed data/raw.mseed --output data/features.csv --window 30 --hop 5

# Train on a CSV with columns: label, pe, wpe, mpe_tau1, ...
PYTHONPATH=src python3 bin/train_pe.py data/features.csv
```

### Use the API directly
```python
from permutation_entropy.features import sliding_windows, multiscale_pe
from permutation_entropy.models import train_logistic, evaluate

series = [0.1, 0.3, 0.2, 0.4, 0.5, 0.7, 0.6, 0.2]
windows = sliding_windows(series, window=5, step=1)
rows = []
for w in windows:
    pe = multiscale_pe(w, m=4, taus=(1, 2, 3))
    rows.append([val for _, val in pe])

X = rows
y = [0, 0, 1, 1]  # toy labels
model = train_logistic(X, y)
acc, auc = evaluate(model, X, y)
print(acc, auc)
```

## Notes
- Deterministic tie-breaking in ordinal patterns (value, then index) to make PE stable on quantized data.
- Weighted PE uses window variance as weights; if a window is flat the entropy returns 0.0.
- `multiscale_pe` returns `[(tau, value), ...]` for convenient unpacking.
- The CLI uses a stratified 80/20 split and `class_weight='balanced'` to cope with imbalance.

## Next steps
- Add automated feature extraction from MiniSEED via `obspy`.
- Add notebooks with real seismic case studies.
- Package for pip and add CI/tests.

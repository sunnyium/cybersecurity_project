# Cybersecurity Project — Network Intrusion Detection (CICIDS2017)

Early-stage companion project to [ids-portfolio](../../Downloads/ids-portfolio),
built on the full CICIDS2017 dataset (8 raw daily capture CSVs). Currently
focused on data consolidation and preprocessing; modeling and evaluation
are still in progress.

## Status

- [x] Raw CICIDS2017 CSVs collected (`data/`)
- [x] Initial merge / label consolidation (`src/build_dataset.py`,
      `processing/preprocessing.ipynb`)
- [ ] Feature cleaning and EDA
- [ ] Model training
- [ ] Evaluation and writeup

## Repo layout

```
data/         # raw CICIDS2017 CSVs (gitignored) + merged parquet outputs
processing/   # exploratory / preprocessing notebooks
src/          # dataset-building scripts
requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

More details will be added as the project progresses.

# Network Intrusion Detection on CIC-IDS2017 — Multi-Class Attack Categorization

Classifying network flows into **seven categories** : BENIGN and six attack
families (DoS, DDoS, PortScan, BruteForce, WebAttack, Bot) from the full
CIC-IDS2017 dataset (~2.5M flows across a week of capture with no subsampling).
The emphasis is on justified data engineering and honest evaluation under a
class imbalance.

## Status

- [x] Raw CIC-IDS2017 CSVs collected (`data/`)
- [x] Schema verification, merge, and cleaning (`src/build_dataset.py`,
    `processing/01preprocessing.ipynb`)
- [x] Label grouping and EDA (`src/eda.py`, `processing/02eda.ipynb`)
- [x] Split protocol and model definitions (`src/split_dataset.py`,
    `src/models.py`)
- [x] Model training and persistence (`src/train.py` → `data/xgb_multiclass.json`)
- [x] Evaluation utilities (`src/evaluation.py`)
- [ ] **Modeling & diagnosis notebook** is the per-class report, confusion
    matrix, baseline ladder, feature importance and cross-validation that
    `src/evaluation.py` exposes are written but not yet narrated in a notebook
- [ ] Final writeup

## Pipeline

```
8 raw daily CSVs  (2,572,640 rows)
  src/build_dataset.py :: merge_files()
    per-file cleaning, then append
▼
data/merged_rawlabels.parquet
  src/build_dataset.py :: clean_merged()
    cross-file dedup, constant-column removal, CamelCase renaming
▼
data/merge_complete.parquet   (2,497,980 rows, 69 features + Label + SourceFile)
  src/split_dataset.py    stratified 80/20, BENIGN capped in train only
    src/train.py
▼
data/xgb_multiclass.json  +  data/xgb_multiclass_meta.json
```

## Notebooks

1. **`processing/01preprocessing.ipynb`** Asks is merging the 8 files valid?
  Schema verification (1 distinct schema across all files), per-file cleaning,
  cross-file deduplication (74,660 duplicates removed), and removal of 8
  constant columns.
2. **`processing/02eda.ipynb`** Addresses lass imbalance, feature scale and skew,    
  data quality, class separability, and feature redundancy each finding tied to a
  downstream modeling decision.

## Data decisions worth calling out

Recorded here because they shape every number below.

- **Label grouping.** 15 raw labels collapse into 7 modeling groups
  (`LABEL_MAP` in `src/build_dataset.py`). The four DoS variants become `DoS`,
  the two Patator attacks become `BruteForce`, the three Web-Attack variants
  become `WebAttack`.
- **Two classes excluded.** Infiltration (36 rows) and Heartbleed (11 rows) are
  too rare to train or evaluate on, and are tagged `EXCLUDED` rather than
  silently dropped.
- **Severe imbalance.** 1,064:1 between the largest and smallest class; BENIGN
  alone is 82.96% of rows. Predicting BENIGN for everything therefore scores
  ~83% accuracy, which is why accuracy is never the headline metric.
- **Sentinels vs. corruption.** `-1` fills 47.8% / 35.7% of the
  `InitWinBytes` columns and means *not observed*, not a measurement. Separately,
  2,910 rows (0.117%) carry impossible negative durations, rates or header
  lengths and counter overflow, handled distinctly from the sentinels.
- **`DestinationPort` is categorical**, not ordinal, despite being stored as a
  float across 53,791 distinct values.
- **Redundant features.** 72 feature pairs correlate above |r| = 0.95; 6 pairs
  were confirmed to be exact aliases on the full data (e.g. `TotalFwdPackets` /
  `SubflowFwdPackets`), while 2 sample-level candidates failed verification.

## Current results

From `data/xgb_multiclass_meta.json`, produced by `src/train.py` on the full
cleaned dataset:

| | |
|---|---|
| Training rows | 540,543 (BENIGN capped at 200,000 in the training fold only) |
| Test rows | 499,587 (true class distribution, never subsampled) |
| Accuracy | 0.9988 |
| Macro precision | 0.9442 |
| Macro recall | 0.9950 |
| Macro F1 | 0.9648 |
| Fit time | 33.7 s |

Macro recall sits well above macro precision, so the model **over-flags** at
least one attack family — it catches nearly everything but raises false alarms.
Which class, and how badly, is exactly what the per-class report and confusion
matrix in the pending modeling notebook are for. The aggregate numbers above
should not be read as the final result until that diagnosis is done.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The 8 raw CSVs are not committed — download CIC-IDS2017 from the Canadian
Institute for Cybersecurity and place the files in `data/` under their original
names (see `FILE_ORDER` in `src/build_dataset.py`). Then:

```bash
python src/build_dataset.py # 8 CSVs -> merge_complete.parquet
python src/split_dataset.py # print the split and class balance
python src/train.py # fit and persist the model
jupyter notebook # run processing/01 -> 02
```

`src/train.py` takes `--cap N` to change the BENIGN cap in the training fold
(`--cap 0` disables it) and `--no-score` to skip the post-fit sanity check.

## Repo layout

```
src/
  build_dataset.py   # 8 CSVs -> cleaned parquet; label grouping; dataset loading
  eda.py             # imbalance, scale/skew, data quality, separability, redundancy
  plotting.py        # shared figure styling; writes to data/results
  split_dataset.py   # stratified split + majority-class subsampling
  models.py          # majority / logistic baselines + XGBoost; chunked prediction
  train.py           # fit and persist the model, with a metadata sidecar
  evaluation.py      # imbalance-aware metrics, baseline ladder, confusion matrix,
   # feature importance, stratified-CV stability being written
processing/
  01preprocessing.ipynb   # schema verification, merge, cleaning
  02eda.ipynb             # imbalance, scale, data quality, separability, redundancy
data/                # raw CSVs + parquet + model (gitignored); results/ images kept
```

## Engineering notes

- **Memory-bounded consolidation.** The naive `pd.concat` of 8 files peaks at
  several GB. `build_dataset.py` streams one file at a time through a pyarrow
  `ParquetWriter` in 250K-row batches, and does the cross-file dedup and
  zero-variance pass in chunks, keeping peak memory near a single file.
- **Chunked prediction.** `models.predict_in_chunks` scores the 499K-row test
  set 250K rows at a time so the `(n, 7)` probability matrix never has to be
  materialised in one allocation.
- **Imbalance handled where it belongs.** BENIGN is capped in the *training*
  fold only; the test fold keeps the true distribution, so every reported
  metric reflects the real class mix.
- **Scaling only where it matters.** The logistic baseline is wrapped in a
  `StandardScaler` fit on training data alone; XGBoost is scale-invariant and
  gets the raw features.
- **Class order is persisted.** The saved model stores class *indices*, so
  `xgb_multiclass_meta.json` records the encoder order and feature order
  alongside it. Reading importances back without that mapping is how a silent
  class mix-up happens.

## Known limitations

- **Within-capture evaluation.** A random split shares each attack tool's
  fingerprint between train and test, so scores partly reflect tool recognition
  rather than proven generalization to novel attacks. The EDA already flags this:
  `InitWinBytesForward` tiers classes by OS/tool default, not by attack concept.
  Cross-dataset evaluation (e.g. CSE-CIC-IDS2018) is the honest next test.
- **Two attacks are out of scope** (Infiltration, Heartbleed) — deliberately
  excluded and documented above.
- **No hyperparameter search**; sensible defaults only.

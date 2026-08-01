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
- [x] Modelling and diagnosis (`processing/03modelling.ipynb`) covering the
    baseline ladder, per-class report, confusion matrix, feature importance
    and stratified cross-validation
- [ ] Cross-dataset evaluation against CSE-CIC-IDS2018, the one test that can
    show whether this generalises beyond a single week of capture

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
2. **`processing/02eda.ipynb`** Addresses class imbalance, feature scale and skew,
  data quality, class separability, and feature redundancy each finding tied to a
  downstream modeling decision.
3. **`processing/03modelling.ipynb`** Asks which attack families the model
  actually catches and which it over-flags. Baseline ladder (majority to
  logistic to XGBoost), per-class report, confusion matrix in both normalised
  and raw-count form, false-positive accounting, gain-based feature importance,
  and 5-fold stratified cross-validation.

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

## Results

Trained on 540,543 rows (BENIGN capped at 200,000 in the training fold only) and
scored on 499,587 held-out rows at the true class distribution.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| Always BENIGN | 0.8296 | 0.1185 | 0.1429 | 0.1296 |
| Logistic regression | 0.9604 | 0.6130 | 0.7092 | 0.6375 |
| **XGBoost** | **0.9988** | **0.9442** | **0.9950** | **0.9648** |

The control scores 0.83 accuracy while catching nothing, which is why accuracy
is reported only to show the trap.

### The macro-precision gap is one class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| BENIGN | 1.000 | 0.999 | 0.999 | 414,451 |
| DoS | 0.997 | 1.000 | 0.998 | 38,746 |
| DDoS | 0.999 | 1.000 | 1.000 | 25,603 |
| PortScan | 0.989 | 0.999 | 0.994 | 18,139 |
| BruteForce | 0.999 | 1.000 | 0.999 | 1,830 |
| WebAttack | 0.979 | 0.991 | 0.985 | 429 |
| **Bot** | **0.646** | 0.977 | **0.778** | 389 |

Every family except Bot holds precision at or above 0.979. Excluding Bot, macro
precision rises from 0.944 to 0.994, so the gap against macro recall (0.995) is
one class rather than a systemic bias.

The mechanism is worth stating plainly. Bot and PortScan absorb almost exactly
the same number of false positives out of BENIGN, 208 and 207. For PortScan,
with 18,139 true rows, that is noise and precision stays at 0.989. For Bot, with
389, it is fatal: 208 false alarms against 380 correct ones. Identical absolute
error, opposite consequence. Only 0.05% of BENIGN rows are misrouted to Bot, but
BENIGN outnumbers Bot by more than 1,000:1, so a rounding error on the majority
class becomes the minority class's dominant failure mode.

### Stability

5-fold stratified cross-validation gives macro-F1 **0.9916 ± 0.0011**. The tight
spread shows the fit is stable under resampling and the single split was not
lucky. The *level* is optimistic and is not the headline: the folds are drawn
from the capped training set, where the in-fold BENIGN:Bot ratio is roughly
129:1 instead of the 1,065:1 of real traffic, so Bot's precision recovers inside
a fold. **0.9648 on the uncapped test fold is the number that reflects real
traffic.**

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
jupyter notebook # run processing/01 -> 02 -> 03
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
   # feature importance, stratified-CV stability
processing/
  01preprocessing.ipynb   # schema verification, merge, cleaning
  02eda.ipynb             # imbalance, scale, data quality, separability, redundancy
  03modelling.ipynb       # baseline ladder, per-class diagnosis, importance, CV
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
  rather than proven generalization to novel attacks. The EDA flagged
  `InitWinBytesForward` as tiering classes by OS/tool default rather than by
  attack concept. It turns out to rank only 12th by gain at 1.9%, which weakens
  that one piece of evidence but does not clear the model, because the design
  cannot separate attack signal from tool signature either way. Cross-dataset
  evaluation (e.g. CSE-CIC-IDS2018) is the honest next test.
- **Bot detection is not deployable as it stands.** At 0.646 precision, roughly
  one in three Bot alerts is a benign flow. Recall is 0.977, so the signal is
  there; the problem is the false-alarm rate against a BENIGN pool three orders
  of magnitude larger. Per-class threshold tuning or a cost-sensitive objective
  is the obvious next experiment and is not attempted here.
- **Two attacks are out of scope** (Infiltration, Heartbleed), deliberately
  excluded and documented above.
- **The logistic baseline did not converge** at `max_iter=200`. It is a
  reference rung, not a tuned candidate, so this is recorded rather than fixed.
- **No hyperparameter search**; sensible defaults only.

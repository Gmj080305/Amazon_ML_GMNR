# Stage 5 — Feature Engineering

## What this stage does

For every `(S1, candidate)` pair in `candidate_pairs.tsv`, computes 17 numerical features that describe how similar the two records are. The output is a labeled feature matrix used to train the matching model in Stage 7.

---

## Features

### Name similarity (6)

All six metrics run on the `name_norm` field from `norm_s1.tsv` and `norm_s2/s3.tsv`.

| Feature | What it measures |
|---|---|
| `name_lev` | Levenshtein edit distance, normalised to 0–1 |
| `name_jaro` | Jaro-Winkler similarity — rewards shared prefix |
| `name_jaccard` | Token-set Jaccard on `name_tokens` |
| `name_tfidf` | Char 3-gram TF-IDF cosine — catches typos and partial matches |
| `name_tsr` | Token sort ratio — handles word reordering (`Acme Corp` vs `Corp Acme`) |
| `name_tset` | Token set ratio — handles one name being a subset of the other |

### Address similarity (6)

Same six metrics on `addr_norm`.

| Feature | What it measures |
|---|---|
| `addr_lev` | Levenshtein on normalised address |
| `addr_jaro` | Jaro-Winkler on normalised address |
| `addr_jaccard` | Token-set Jaccard on `addr_tokens` |
| `addr_tfidf` | Char 3-gram TF-IDF cosine on address |
| `addr_tsr` | Token sort ratio on address |
| `addr_tset` | Token set ratio on address |

### Structural (3)

| Feature | What it measures |
|---|---|
| `country_match` | 1 if both records have the same country, 0 otherwise |
| `name_tok_ratio` | min(name tokens) / max(name tokens) — penalises very different lengths |
| `addr_tok_ratio` | same ratio on address tokens |

### Rank (1)

| Feature | What it measures |
|---|---|
| `name_rank` | Rank of this candidate by `name_tsr` among all candidates for this S1 entity, normalised 0–1. 0 = best candidate, 1 = worst. Tells the model whether this is a strong or weak candidate relative to the alternatives. |

### Ambiguity (1)

| Feature | What it measures |
|---|---|
| `n_candidates` | Total number of candidates for this S1 entity. High count = ambiguous entity — the model should be more conservative about declaring a match. |

---

## Implementation notes

**Why chunks?** 110M pairs × 18 floats ≈ 15 GB if held in memory at once. The script processes 500k pairs at a time, saves each chunk as a parquet file, then merges at the end. Peak RAM usage stays under 4 GB.

**Chunk size default is 500,000 pairs.** With 110M total pairs that gives 220 chunks. On a 64 GB instance you can safely increase this to 1–2M pairs per chunk for fewer disk writes and a ~20% speedup:

```bash
python src/features.py --chunk-size 1000000
```

**Why chunking is safe for the rank feature.** `candidate_pairs.tsv` has one row per S1 entity, so all candidates for a given S1 are on the same line and get buffered into the same chunk. The `name_rank` groupby is always computed over a complete set of candidates for each entity.

**Why joblib Parallel?** Levenshtein, Jaro-Winkler, token sort/set ratios are per-pair operations with no shared state. rapidfuzz releases the GIL so threading is effective. At ~3M pairs/second/core, 8 cores processes 110M pairs in under 5 minutes for the string metrics alone.

**Why TF-IDF is chunk-level not pair-level?** The vectoriser needs all texts to compute IDF weights. Vectorising the whole chunk at once gives meaningful character frequency weights across that batch. This is the actual bottleneck, not the string metrics.

**No SageMaker Processing Job needed.** The `ml.r6i.2xlarge` notebook instance (64 GB RAM, 8 cores) is sufficient to run this directly. A Processing Job would add 5–10 minutes of startup overhead for no benefit on a single run. Just run it in a `tmux` session so it survives SSH disconnects:

```bash
tmux new -s features
python src/features.py
# Ctrl+B then D to detach, tmux attach -t features to return
```

---

## Output

`output/features/features_train.parquet`

| Column | Type | Description |
|---|---|---|
| `source1_entity_id` | str | S1 entity ID |
| `candidate_id` | str | S2 or S3 candidate ID |
| `label` | int | 1 = true match, 0 = not a match |
| `name_lev` … `n_candidates` | float | The 17 features above |

---

## How to run

```bash
pip install rapidfuzz scikit-learn joblib pyarrow --quiet

python src/features.py

# with explicit options
python src/features.py \
  --input-dir       data/normalized \
  --data-dir        data \
  --candidate-pairs output/candidate_pairs.tsv \
  --output-dir      output/features \
  --chunk-size      500000 \
  --workers         8
```

Expected time on `ml.r6i.2xlarge` (8 cores): **15–25 minutes** for 110M pairs.

Progress prints every 500k pairs:

```
chunk   1: 500,000 pairs, pos rate 0.0021
chunk   2: 1,000,000 pairs, pos rate 0.0019
...
Done. 110,806,114 rows, 17 features
Positive rate : 0.00089
Saved         : output/features/features_train.parquet
```

---

## Gate metric

```python
import pandas as pd

df = pd.read_parquet('output/features/features_train.parquet')

assert df.shape[1] == 20            # 17 features + source1_entity_id, candidate_id, label
assert df['label'].sum() > 0        # some positives exist
assert df.isna().sum().sum() == 0   # no NaNs

print(f'rows          : {len(df):,}')
print(f'positive rate : {df["label"].mean():.5f}')
```

Both name and address feature groups must have non-zero variance — if either is all-zero, the normalization pipeline for that field is broken.

---

## How this feeds into Stage 7

Stage 6 splits this parquet by S1 entity (not by row) into train/val sets. Stage 7 loads those splits, trains LightGBM on the feature columns, and uses the val set to tune the decision threshold against F₀.5.

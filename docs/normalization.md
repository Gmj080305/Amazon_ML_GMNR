# Normalization — How It Works

This document covers what we did in EDA, what the normalization pipeline does, and how to run it.

---

## EDA Findings

**Null rates** (from `null_rates.json`)

| Source | Missing names | Missing addresses |
|--------|--------------|-------------------|
| S1     | 0            | 0                 |
| S2     | 2            | ~169k             |
| S3     | 13           | ~176k             |

S2 and S3 have a large number of records with no address at all. The normalization pipeline handles this gracefully — missing fields produce empty normalized output, not errors.

**Singleton fraction**: ~5.6% of S1 entities have no matches in S2/S3. These are "singletons" and the model should predict an empty match list for them.

**Noise patterns found in the data**

- Legal suffix typos and garbled transliterations: `limirrad`, `limidhèdh`, `elaelapi` (garbled LLP from Indian script)
- Abbreviation inconsistency: `Pvt. Ltd.` vs `Private Limited` vs `Pvt Ltd`
- Road type abbreviations: `Rd` vs `Road`, `St` vs `Street`, `MG Rd` vs `Mahatma Gandhi Road`
- Landmark references in addresses: `Near SBI ATM`, `Opp. City Mall`
- Municipal number formats: `D.No. 4-5-6`, `Plot No. 45/B`, `123/4A`
- Colonial city names: `Bangalore` vs `Bengaluru`, `Bombay` vs `Mumbai`
- Non-Latin scripts: Devanagari, Tamil, Telugu, Kannada etc. found in some records

---

## Normalization Pipeline

`src/normalize.py` processes each record in five steps:

**1. Script detection + transliteration**
Detects if `business_name` or `business_address` is in a non-Latin script (Devanagari, Tamil etc.) and converts it to Latin using ITRANS transliteration. Records already in Latin pass through unchanged.

**2. Extraction** (runs on transliterated but otherwise raw text)
- `extract_pin()` — pulls out any 6-digit Indian PIN or 5-digit US/French ZIP already in the address. Returns `None` if absent; never guesses.
- `extract_landmark()` — pulls out phrases like `Near SBI ATM` → `SBI ATM`. Stored as its own column so it doesn't pollute address similarity.
- `extract_legal_suffix()` — identifies the canonical suffix (`pvt ltd`, `inc`, `co` etc.).

**3. Name normalization**
`normalize_name()`: ASCII-lower → `&` to `and` → strip punctuation → normalize legal suffixes → abbreviate street terms.

**4. Address normalization**
`normalize_address()`: remove landmark phrase → strip number prefixes (D.No. etc.) → normalize separators (123/4A → 123 4A) → ASCII-lower → strip punctuation → abbreviate street terms → normalize city variants.

**5. Suffix normalization direction**
Abbreviated form is canonical throughout. `road → rd`, `limited → ltd`, `corporation → co`. This ensures `MG Road` and `MG Rd` produce identical tokens.

---

## Output Columns

Each normalized TSV has these columns:

| Column | Description |
|--------|-------------|
| `entity_id` | Unchanged from source |
| `country` | Lowercased |
| `name_norm` | Fully normalized business name |
| `name_core` | Name with legal suffix stripped |
| `name_tokens` | Space-separated sorted unique tokens of `name_norm` |
| `legal_suffix` | Canonical suffix (`pvt ltd`, `inc`, `co` etc.) or empty |
| `addr_norm` | Fully normalized address (no landmark) |
| `addr_tokens` | Space-separated sorted unique tokens of `addr_norm` |
| `pin_zip` | Extracted PIN/ZIP or empty |
| `has_pin` | `True` / `False` |
| `landmark` | Extracted landmark phrase or empty |

---

## How to Run

### Option 1 — SageMaker Processing Job (recommended)

Run `notebooks/03_normalization_job.ipynb`. This launches the job, processes all 6 files, and writes the normalized TSVs to S3 under `data/normalized/`.

### Option 2 — Locally or in the notebook instance

```bash
python src/normalize.py \
  --input-dir  data/ \
  --output-dir output/
```

Expects files at `data/train/train_source1.tsv` etc. Writes to `output/train/norm_s1.tsv` etc.

### Option 3 — Import as a module

```python
from src.normalize import normalize_record, normalize_dataframe

norm_df = normalize_dataframe(training_s1)
```

---

## Files

```
src/
  normalize.py          processing job script + importable module

notebooks/
  01_eda.ipynb          null rates, singleton fraction, noise pattern discovery
  03_normalization_job.ipynb   launches the SageMaker Processing Job

data/
  train/                raw train TSVs
  test/                 raw test TSVs
  normalized/           output of the processing job (written to S3)
    train/
      norm_s1.tsv
      norm_s2.tsv
      norm_s3.tsv
    test/
      norm_test_s1.tsv
      norm_test_s2.tsv
      norm_test_s3.tsv
```

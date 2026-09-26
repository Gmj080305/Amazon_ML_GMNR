"""
Stage 5 - Pairwise NAME Feature Engineering
=============================================================
Six name-similarity features for (S1, candidate) pairs:
    1. Levenshtein ratio
    2. Jaro-Winkler similarity
    3. Token Jaccard
    4. Char 3-gram TF-IDF cosine similarity
    5. Token sort ratio (handles word reordering)
    6. Token set ratio

WHAT THIS IS BUILT ON (confirmed with teammates so far, nothing beyond this
is assumed):
    - Normalized name column so far: `business_name_translit` (string dtype,
      one row per source record; training data is a pandas DataFrame).
    - Transformations applied SO FAR upstream: script transliteration to
      Latin, lowercasing, French character -> Latin conversion.
    - NOT yet done upstream: abbreviation expansion, legal suffix
      extraction, punctuation stripping. Do not assume any string passed
      into this script has had those applied.
    - The real candidate_pairs file format (stage 3/4 output) is not
      finalized yet. This script does not assume a specific pair-file
      schema - it just needs a DataFrame with two name columns, one per
      side of the pair (i.e. already one row per (S1, candidate) pair,
      exploded out of any comma-separated candidate list if that's the
      real format). Column names are passed in as arguments so renaming
      later, once the real schema is confirmed, is a one-line change.

Nothing below depends on blocking (stage 3/4) being finished. These are
pure functions over two strings (except the TF-IDF step, see note below) -
point them at real data later with zero changes beyond column names and
file paths.

Note on "rapidfuzz is O(n^2), too slow for 1M+ pairs": that's true only for
a naive pure-Python string-matching loop. rapidfuzz's ratio/token functions
used below are C++-implemented and fast per call; the actual scale concern
is calling any per-pair function 1M+ times in a plain Python for-loop,
which is why this script uses joblib.Parallel. This has nothing to do with
whether Levenshtein/Jaro-Winkler themselves are usable as features - they
are, on the blocked candidate set (which stage 3/4 already caps at ~200
candidates/entity).
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed


# ---------------------------------------------------------------------------
# 1. Pure pairwise similarity functions (operate on two strings, return 0-1)
# ---------------------------------------------------------------------------

def levenshtein_ratio(a: str, b: str) -> float:
    """Normalized Levenshtein ratio, 0-1. rapidfuzz.fuzz.ratio is 0-100."""
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def jaro_winkler_sim(a: str, b: str) -> float:
    """Jaro-Winkler similarity - already 0-1 in rapidfuzz.distance."""
    if not a or not b:
        return 0.0
    return JaroWinkler.normalized_similarity(a, b)


def token_jaccard(a: str, b: str) -> float:
    """Jaccard similarity over whitespace-split tokens."""
    tokens_a, tokens_b = set(a.split()), set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def token_sort_ratio(a: str, b: str) -> float:
    """Sorts tokens in each string before comparing - robust to word order."""
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b) / 100.0


def token_set_ratio(a: str, b: str) -> float:
    """Compares token SETS (dedups tokens, ignores order and repeats)."""
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


# ---------------------------------------------------------------------------
# 2. Char 3-gram TF-IDF cosine - NOT a pure pairwise function like the 5
#    above. It needs a vectorizer fit on a corpus first.
# ---------------------------------------------------------------------------

def tfidf_char3gram_cosine(names_a: pd.Series, names_b: pd.Series) -> np.ndarray:
    """
    Row-wise cosine similarity between names_a[i] and names_b[i], using a
    char 3-gram TF-IDF space fit on the UNION of both sides (so both sides
    map into the same vocabulary).

    sklearn's TfidfVectorizer L2-normalizes rows by default, so cosine
    similarity between two rows is just their dot product - this avoids
    building a full n x n pairwise similarity matrix, which would not
    scale to 1M+ pairs.
    """
    names_a = names_a.fillna("").astype(str)
    names_b = names_b.fillna("").astype(str)

    corpus = pd.concat([names_a, names_b], ignore_index=True)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3))
    vectorizer.fit(corpus)

    vecs_a = vectorizer.transform(names_a)
    vecs_b = vectorizer.transform(names_b)

    sims = np.asarray(vecs_a.multiply(vecs_b).sum(axis=1)).ravel()
    return sims


# ---------------------------------------------------------------------------
# 3. Parallel row-wise computation of the 5 pure pairwise functions
# ---------------------------------------------------------------------------

def _row_features(name_a: str, name_b: str) -> dict:
    name_a = name_a if isinstance(name_a, str) else ""
    name_b = name_b if isinstance(name_b, str) else ""
    return {
        "name_levenshtein_ratio": levenshtein_ratio(name_a, name_b),
        "name_jaro_winkler": jaro_winkler_sim(name_a, name_b),
        "name_token_jaccard": token_jaccard(name_a, name_b),
        "name_token_sort_ratio": token_sort_ratio(name_a, name_b),
        "name_token_set_ratio": token_set_ratio(name_a, name_b),
    }


def compute_name_features(
    df: pd.DataFrame,
    name_col_s1: str = "business_name_translit_s1",
    name_col_cand: str = "business_name_translit_cand",
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Computes all 6 name features for every row (= one S1/candidate pair)
    in df.

    Parameters
    ----------
    df : DataFrame with at least [name_col_s1, name_col_cand]. One row per
         (S1, candidate) pair.
    name_col_s1, name_col_cand : column names holding the normalized name
         string for each side. CHANGE THESE once the real candidate-pairs
         schema is confirmed - everything else stays the same.
    n_jobs : passed to joblib.Parallel. -1 uses all available cores.

    Returns
    -------
    A copy of df with 6 new feature columns appended.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(_row_features)(a, b)
        for a, b in zip(df[name_col_s1], df[name_col_cand])
    )
    feat_df = pd.DataFrame(results, index=df.index)

    feat_df["name_tfidf_char3gram_cosine"] = tfidf_char3gram_cosine(
        df[name_col_s1], df[name_col_cand]
    )

    return pd.concat([df, feat_df], axis=1)


# ---------------------------------------------------------------------------
# 4. Save / reload - matches the Stage 5 gate: "feature matrix saves and
#    reloads from S3, shape preserved". Works with a local path now; swap
#    for an s3://... URI later (needs s3fs installed) - no other changes.
# ---------------------------------------------------------------------------

def save_features(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False)


def load_features(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# 5. Mock data - stand-in until real normalized data + real candidate pairs
#    land. Mimics the confirmed schema: strings, lowercased, Latin script.
#    DELETE this section once real data is available.
# ---------------------------------------------------------------------------

def make_mock_pairs() -> pd.DataFrame:
    """
    Hand-built pairs covering the cases that matter for sanity-checking:
    identical, reordered tokens, typo, abbreviation-style difference,
    unrelated business, and a missing name.
    """
    data = [
        ("sharma traders pvt ltd", "sharma traders pvt ltd", "identical"),
        ("sharma traders pvt ltd", "traders sharma pvt ltd", "word reorder"),
        ("sharma traders pvt ltd", "sharma traders private limited", "abbreviation variant (not expanded upstream yet)"),
        ("sharma traders pvt ltd", "sharme tradres pvt ltd", "typo"),
        ("sharma traders pvt ltd", "green valley bakery", "unrelated business"),
        ("sharma traders pvt ltd", "", "missing candidate name"),
        ("global tech solutions inc", "global technology solutions inc", "partial word match"),
        ("global tech solutions inc", "globale tech solutionz inc", "transliteration-style noise"),
    ]
    df = pd.DataFrame(data, columns=[
        "business_name_translit_s1",
        "business_name_translit_cand",
        "case_description",
    ])
    df.insert(0, "candidate_entity_id", [f"S2-{i:05d}" for i in range(len(df))])
    df.insert(0, "s1_entity_id", [f"S1-{i:05d}" for i in range(len(df))])
    return df


# ---------------------------------------------------------------------------
# 6. Demo / smoke test - run this in Jupyter to confirm everything works
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mock_df = make_mock_pairs()
    result = compute_name_features(mock_df)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)

    cols_to_show = [
        "case_description",
        "name_levenshtein_ratio", "name_jaro_winkler", "name_token_jaccard",
        "name_tfidf_char3gram_cosine", "name_token_sort_ratio", "name_token_set_ratio",
    ]
    print(result[cols_to_show].round(3))

    # Parquet round-trip check (Stage 5 gate: shape preserved)
    save_features(result, "name_features_mock.parquet")
    reloaded = load_features("name_features_mock.parquet")
    assert reloaded.shape == result.shape, "shape mismatch after parquet round-trip!"
    print("\nParquet round-trip OK - shape preserved:", reloaded.shape)

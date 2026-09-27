"""
src/features.py
Stage 5 — Pairwise feature engineering

Computes features for every (S1, candidate) pair in candidate_pairs.tsv and
writes them as chunk parquet files. Works for both splits: train pairs get a
label from the ground truth, test pairs get label -1.

Candidate pairs are read in whole S1 rows, so an entity's candidates never
straddle two chunks. The group features below depend on seeing every
candidate of an entity at once.

Feature groups:
  similarity  - six string metrics each on name and address
  rarity      - name overlap weighted by token rarity; generic names like
                "general trading" share tokens without meaning much
  address     - house/plot number overlap, PIN match, missing-address flag
  group       - how this candidate compares with the entity's other
                candidates; the pair-level metrics cannot tell the clear
                match from the least-bad option
  candidate   - how many entities list this candidate, and whether this
                entity is its best; needs a second pass over all chunks

Run from the repo root:
  python src/features.py
  python src/features.py --split test
"""

import os
import gc
import glob
import math
import argparse
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein, JaroWinkler
from rapidfuzz.fuzz import token_sort_ratio, token_set_ratio
from sklearn.feature_extraction.text import TfidfVectorizer


FEATURE_COLS = [
    'name_lev', 'name_jaro', 'name_jaccard', 'name_tfidf', 'name_tsr', 'name_tset',
    'addr_lev', 'addr_jaro', 'addr_jaccard', 'addr_tfidf', 'addr_tsr', 'addr_tset',
    'name_tok_ratio', 'addr_tok_ratio', 'name_rank', 'n_candidates',
    'name_idf_jac', 'name_max_shared_idf', 'addr_num_jac',
    'pin_match', 'addr_missing', 'from_s3',
    'pair_sim', 'grp_max_sim', 'grp_gap_sim', 'grp_margin', 'grp_rank_sim', 'n_strong',
    'cand_n_s1', 'cand_gap_sim', 'mutual_best',
]

RECORD_COLS = ['entity_id', 'name_norm', 'name_tokens',
               'addr_norm', 'addr_tokens', 'pin_zip']

FILES = {
    'train': ('train/norm_s1.tsv', 'train/norm_s2.tsv', 'train/norm_s3.tsv'),
    'test':  ('test/norm_test_s1.tsv', 'test/norm_test_s2.tsv', 'test/norm_test_s3.tsv'),
}

STRONG_SIM = 0.8


# ── Loading ────────────────────────────────────────────────────────────────

def read_records(path):
    return pd.read_csv(path, sep='\t', dtype=str, usecols=RECORD_COLS).fillna('')


def load_records(input_dir, split):
    f1, f2, f3 = FILES[split]
    s1 = read_records(f'{input_dir}/{f1}').set_index('entity_id')
    pool = pd.concat([read_records(f'{input_dir}/{f2}'),
                      read_records(f'{input_dir}/{f3}')], ignore_index=True)
    return s1, pool.set_index('entity_id')


def load_ground_truth(path):
    gt = pd.read_csv(path, sep='\t', dtype=str).fillna('')
    return {s1: set(m.split(',')) - {''}
            for s1, m in zip(gt['source1_entity_id'], gt['matched_entity_ids'])}


def token_idf(pool):
    counts = Counter()
    for tokens in pool['name_tokens']:
        counts.update(set(tokens.split()))
    n = len(pool)
    return {t: math.log(n / c) for t, c in counts.items()}


# ── Pair table ─────────────────────────────────────────────────────────────

def explode(block, ground_truth):
    s1_ids, cand_ids, labels = [], [], []
    for s1_id, cands in zip(block['source1_entity_id'], block['candidate_entity_ids']):
        true_ids = ground_truth.get(s1_id, ()) if ground_truth is not None else None
        for cand in cands.split(','):
            if not cand:
                continue
            s1_ids.append(s1_id)
            cand_ids.append(cand)
            labels.append(-1 if true_ids is None else int(cand in true_ids))
    return pd.DataFrame({'source1_entity_id': s1_ids, 'candidate_id': cand_ids,
                         'label': np.array(labels, dtype=np.int8)})


# ── Pair-level features ────────────────────────────────────────────────────

def string_metrics(a, b, prefix):
    metrics = {
        'lev':  process.cpdist(a, b, scorer=Levenshtein.normalized_similarity, workers=-1),
        'jaro': process.cpdist(a, b, scorer=JaroWinkler.similarity, workers=-1),
        'tsr':  process.cpdist(a, b, scorer=token_sort_ratio, workers=-1) / 100.0,
        'tset': process.cpdist(a, b, scorer=token_set_ratio, workers=-1) / 100.0,
    }
    return {f'{prefix}_{k}': v for k, v in metrics.items()}


def tfidf_cosine(a, b):
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3))
    try:
        mat = vec.fit_transform(list(a) + list(b))   # rows are L2-normalised
    except ValueError:
        # Every text in the chunk is empty, e.g. a chunk of records with no address.
        return np.zeros(len(a))
    n = len(a)
    return np.asarray(mat[:n].multiply(mat[n:]).sum(axis=1)).ravel()


def name_token_features(a_tokens, b_tokens, idf):
    n = len(a_tokens)
    jaccard, tok_ratio = np.zeros(n), np.zeros(n)
    idf_jac, max_shared = np.zeros(n), np.zeros(n)
    for i, (x, y) in enumerate(zip(a_tokens, b_tokens)):
        a, b = set(x.split()), set(y.split())
        union = a | b
        if not union:
            continue
        shared = a & b
        jaccard[i] = len(shared) / len(union)
        tok_ratio[i] = min(len(a), len(b)) / max(len(a), len(b))
        union_w = sum(idf.get(t, 0.0) for t in union)
        if shared:
            shared_w = [idf.get(t, 0.0) for t in shared]
            idf_jac[i] = sum(shared_w) / union_w if union_w else 0.0
            max_shared[i] = max(shared_w)
    return {'name_jaccard': jaccard, 'name_tok_ratio': tok_ratio,
            'name_idf_jac': idf_jac, 'name_max_shared_idf': max_shared}


def addr_token_features(a_tokens, b_tokens):
    n = len(a_tokens)
    jaccard, tok_ratio, num_jac = np.zeros(n), np.zeros(n), np.zeros(n)
    for i, (x, y) in enumerate(zip(a_tokens, b_tokens)):
        a, b = set(x.split()), set(y.split())
        union = a | b
        if not union:
            continue
        jaccard[i] = len(a & b) / len(union)
        tok_ratio[i] = min(len(a), len(b)) / max(len(a), len(b))
        a_num = {t for t in a if any(c.isdigit() for c in t)}
        b_num = {t for t in b if any(c.isdigit() for c in t)}
        if a_num | b_num:
            num_jac[i] = len(a_num & b_num) / len(a_num | b_num)
    return {'addr_jaccard': jaccard, 'addr_tok_ratio': tok_ratio, 'addr_num_jac': num_jac}


def check_ids(pairs, s1, pool, split):
    known = pairs['source1_entity_id'].isin(s1.index) & pairs['candidate_id'].isin(pool.index)
    if not known.any():
        raise ValueError(
            f'none of the ids in the candidate pairs exist in the {split} records. '
            f'candidate_pairs.tsv was probably built from the other split; rebuild it '
            f'with: python src/block.py --split {split}')
    return pairs[known].reset_index(drop=True)


def pair_features(pairs, s1, pool, idf):
    a = s1.reindex(pairs['source1_entity_id']).fillna('')
    b = pool.reindex(pairs['candidate_id']).fillna('')

    features = {}
    features.update(string_metrics(a['name_norm'].values, b['name_norm'].values, 'name'))
    features.update(string_metrics(a['addr_norm'].values, b['addr_norm'].values, 'addr'))
    features['name_tfidf'] = tfidf_cosine(a['name_norm'].values, b['name_norm'].values)
    features['addr_tfidf'] = tfidf_cosine(a['addr_norm'].values, b['addr_norm'].values)
    features.update(name_token_features(a['name_tokens'].values, b['name_tokens'].values, idf))
    features.update(addr_token_features(a['addr_tokens'].values, b['addr_tokens'].values))

    a_pin, b_pin = a['pin_zip'].values, b['pin_zip'].values
    features['pin_match'] = (a_pin == b_pin) & ~np.isin(a_pin, ['', 'nan', 'None'])
    features['addr_missing'] = (a['addr_norm'].values == '') | (b['addr_norm'].values == '')
    features['from_s3'] = pairs['candidate_id'].str.startswith('S3-').values

    for name, values in features.items():
        pairs[name] = np.asarray(values, dtype=np.float32)
    return pairs


# ── Group features (within one S1 entity's candidates) ─────────────────────

def group_features(df):
    # A single similarity summary to compare candidates on; keeps the group
    # features few and readable instead of repeating them per metric.
    df['pair_sim'] = ((df['name_tset'] + df['addr_tset'] + df['name_idf_jac']) / 3
                      ).astype(np.float32)

    by_entity = df.groupby('source1_entity_id')
    size = by_entity['pair_sim'].transform('size')
    df['n_candidates'] = size.astype(np.float32)
    df['name_rank'] = ((by_entity['name_tsr'].rank(ascending=False, method='min') - 1)
                       / size).astype(np.float32)

    df['grp_max_sim'] = by_entity['pair_sim'].transform('max')
    df['grp_gap_sim'] = df['grp_max_sim'] - df['pair_sim']
    df['grp_rank_sim'] = ((by_entity['pair_sim'].rank(ascending=False, method='min') - 1)
                          / size).astype(np.float32)
    df['n_strong'] = (df['pair_sim'] >= STRONG_SIM).groupby(
        df['source1_entity_id']).transform('sum').astype(np.float32)

    ordered = df.sort_values(['source1_entity_id', 'pair_sim'], ascending=[True, False])
    position = ordered.groupby('source1_entity_id').cumcount()
    second = ordered.loc[position == 1].set_index('source1_entity_id')['pair_sim']
    df['grp_margin'] = df['grp_max_sim'] - df['source1_entity_id'].map(second).fillna(0)
    return df


# ── Candidate features (across all entities, second pass) ──────────────────

def candidate_features(chunk_files):
    """How many entities list each candidate, and which scores it best. A pair
    that is each side's best option is the strongest match signal available."""
    both = pd.concat([pd.read_parquet(f, columns=['candidate_id', 'pair_sim'])
                      for f in chunk_files], ignore_index=True)
    stats = both.groupby('candidate_id')['pair_sim'].agg(['max', 'size'])
    del both
    gc.collect()

    for path in chunk_files:
        df = pd.read_parquet(path)
        df['cand_n_s1'] = df['candidate_id'].map(stats['size']).astype(np.float32)
        df['cand_gap_sim'] = (df['candidate_id'].map(stats['max'])
                              - df['pair_sim']).astype(np.float32)
        df['mutual_best'] = ((df['grp_gap_sim'] == 0)
                             & (df['cand_gap_sim'] == 0)).astype(np.float32)
        df.to_parquet(path, index=False)


# ── Main ───────────────────────────────────────────────────────────────────

def main(args):
    s1, pool = load_records(args.input_dir, args.split)
    idf = token_idf(pool)
    ground_truth = load_ground_truth(args.ground_truth) if args.split == 'train' else None

    os.makedirs(args.output_dir, exist_ok=True)
    for old in glob.glob(os.path.join(args.output_dir, 'chunk_*.parquet')):
        os.remove(old)

    chunk_files, total = [], 0
    blocks = pd.read_csv(args.candidate_pairs, sep='\t', dtype=str,
                         chunksize=args.s1_per_chunk)
    for i, block in enumerate(blocks):
        pairs = explode(block.fillna(''), ground_truth)
        if pairs.empty:
            continue
        pairs = check_ids(pairs, s1, pool, args.split)
        pairs = group_features(pair_features(pairs, s1, pool, idf))

        path = os.path.join(args.output_dir, f'chunk_{i:04d}.parquet')
        pairs.to_parquet(path, index=False)
        chunk_files.append(path)
        total += len(pairs)
        print(f'chunk {i + 1}: {total:,} pairs', flush=True)
        del pairs
        gc.collect()

    print('candidate features...', flush=True)
    candidate_features(chunk_files)
    print(f'done: {total:,} pairs, {len(FEATURE_COLS)} features, {len(chunk_files)} chunks '
          f'in {args.output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',        default='train', choices=['train', 'test'])
    parser.add_argument('--input-dir',    default='data/normalized')
    parser.add_argument('--ground-truth', default='data/train/train_ground_truth.tsv')
    parser.add_argument('--candidate-pairs')
    parser.add_argument('--output-dir')
    parser.add_argument('--s1-per-chunk', type=int, default=40_000)
    args = parser.parse_args()

    if args.candidate_pairs is None:
        args.candidate_pairs = ('output/candidate_pairs.tsv' if args.split == 'train'
                                else 'output/test/candidate_pairs.tsv')
    if args.output_dir is None:
        args.output_dir = f'output/features_v2/{args.split}'

    main(args)
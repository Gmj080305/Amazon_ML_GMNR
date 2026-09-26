"""
src/features.py
Stage 5 — Pairwise Feature Engineering

Computes 18 features for every (S1, candidate) pair in candidate_pairs.tsv
and saves the result as a parquet file.

Run:
  python src/features.py
  python src/features.py --workers 8 --chunk-size 500000
"""

import os
import gc
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz.fuzz import token_sort_ratio, token_set_ratio
from rapidfuzz.distance import JaroWinkler, Levenshtein


FEATURE_COLS = [
    'name_lev', 'name_jaro', 'name_jaccard', 'name_tfidf',
    'name_tsr',  'name_tset',
    'addr_lev', 'addr_jaro', 'addr_jaccard', 'addr_tfidf',
    'addr_tsr',  'addr_tset',
    'country_match',
    'name_tok_ratio', 'addr_tok_ratio',
    'name_rank', 'n_candidates',
]


# ── Individual similarity helpers ──────────────────────────────────────────

#levenshtein distance = min no of edits required to change 1 string to another, edits = insert, delete, substitute
def lev_sim(a, b):
    a, b = str(a), str(b)
    return 1.0 - Levenshtein.distance(a, b) / max(len(a), len(b), 1)

#jaro distance = measure of similarity between 2 strings, 0 = no similarity, 1 = exact match, complicated formula, but basically it looks at the number of matching characters and transpositions
def jaro(a, b):
    return JaroWinkler.similarity(str(a), str(b))

#jaccard similarity = size of intersection / size of union, 0 = no similarity, 1 = exact match
def jaccard(a, b):
    a, b = set(str(a).split()), set(str(b).split())
    return len(a & b) / len(a | b) if a | b else 0.0

#token_sort_ratio = sorts the tokens in the strings and then computes the similarity ratio
def tsr(a, b):
    return token_sort_ratio(str(a), str(b)) / 100.0

#token_set_ratio = token_set_ratio = sorts the tokens in the strings, removes duplicates, and then computes the similarity ratio
def tset(a, b):
    return token_set_ratio(str(a), str(b)) / 100.0

#token count ratio = min(len(a), len(b)) / max(len(a), len(b)), 0 = no similarity, 1 = exact match
def tok_count_ratio(a, b):
    na, nb = len(str(a).split()), len(str(b).split())
    return min(na, nb) / max(na, nb, 1)


# ── TF-IDF cosine for a list of text pairs ─────────────────────────────────

#tfidf_cosine = cosine similarity between two vectors, which is the dot product of the vectors divided by the product of their magnitudes. 
#It measures the cosine of the angle between two vectors, which is a measure of how similar they are. 
#The TfidfVectorizer converts a collection of raw documents to a matrix of TF-IDF features, which is a numerical representation of the importance of each word in the document relative to the entire corpus.

def tfidf_cosine(texts_a, texts_b):
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=1)
    mat = vec.fit_transform(texts_a + texts_b)
    n   = len(texts_a)
    a, b = mat[:n], mat[n:]
    scores  = np.array(a.multiply(b).sum(axis=1)).flatten()
    norms   = (np.sqrt(np.array(a.power(2).sum(axis=1)).flatten()) *
               np.sqrt(np.array(b.power(2).sum(axis=1)).flatten()))
    norms[norms == 0] = 1.0
    return scores / norms


# ── Features for one pair ──────────────────────────────────────────────────
def pair_features(row):
    return {
        'name_lev':       lev_sim(row['s1_name'], row['c_name']),
        'name_jaro':      jaro(row['s1_name'], row['c_name']),
        'name_jaccard':   jaccard(row['s1_name_tok'], row['c_name_tok']),
        'name_tsr':       tsr(row['s1_name'], row['c_name']),
        'name_tset':      tset(row['s1_name'], row['c_name']),
        'name_tfidf':     0.0,  # filled chunk-level
        'addr_lev':       lev_sim(row['s1_addr'], row['c_addr']),
        'addr_jaro':      jaro(row['s1_addr'], row['c_addr']),
        'addr_jaccard':   jaccard(row['s1_addr_tok'], row['c_addr_tok']),
        'addr_tsr':       tsr(row['s1_addr'], row['c_addr']),
        'addr_tset':      tset(row['s1_addr'], row['c_addr']),
        'addr_tfidf':     0.0,  # filled chunk-level
        'country_match':  int(row['s1_country'] == row['c_country']),

        'name_tok_ratio': tok_count_ratio(row['s1_name_tok'], row['c_name_tok']),
        'addr_tok_ratio': tok_count_ratio(row['s1_addr_tok'], row['c_addr_tok']),
        'name_rank':      0.0,  # filled chunk-level
        'n_candidates':   row['n_candidates'],
        'source1_entity_id': row['s1_id'],
        'candidate_id':      row['c_id'],
        'label':             row['label'],
    }

#chunking logic to process large datasets in smaller chunks, using parallel processing to speed up the computation of features for each pair of entities. The function computes pairwise features for each row in the chunk and then calculates TF-IDF cosine similarity for names and addresses, as well as ranking based on name similarity scores.
def process_chunk(rows, n_jobs):
    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(pair_features)(r) for r in rows
    )
    df = pd.DataFrame(results)

    df['name_tfidf'] = tfidf_cosine(
        [r['s1_name'] for r in rows], [r['c_name'] for r in rows])
    df['addr_tfidf'] = tfidf_cosine(
        [r['s1_addr'] for r in rows], [r['c_addr'] for r in rows])

    rank = df.groupby('source1_entity_id')['name_tsr'].rank(
        ascending=False, method='min') - 1
    max_rank = df.groupby('source1_entity_id')['name_tsr'].transform('max').rank(
        ascending=False, method='min')
    df['name_rank'] = rank / df.groupby(
        'source1_entity_id')['name_rank'].transform('count').clip(lower=1)

    return df


# ── Main ───────────────────────────────────────────────────────────────────

def main(args):
    norm_cols = ['entity_id', 'country', 'name_norm', 'name_tokens',
                 'addr_norm', 'addr_tokens']

    s1  = pd.read_csv(f'{args.input_dir}/train/norm_s1.tsv',
                      sep='\t', dtype=str, usecols=norm_cols).fillna('')
    s23 = pd.concat([
        pd.read_csv(f'{args.input_dir}/train/norm_s2.tsv',
                    sep='\t', dtype=str, usecols=norm_cols).fillna(''),
        pd.read_csv(f'{args.input_dir}/train/norm_s3.tsv',
                    sep='\t', dtype=str, usecols=norm_cols).fillna(''),
    ], ignore_index=True)

    s1_idx  = s1.set_index('entity_id')
    s23_idx = s23.set_index('entity_id')

    gt = pd.read_csv(f'{args.data_dir}/train/train_ground_truth.tsv',
                     sep='\t', dtype=str).fillna('')
    gt_labels = {
        s1_id: {x.strip() for x in str(m).split(',') if x.strip()}
        for s1_id, m in zip(gt['source1_entity_id'], gt['matched_entity_ids'])
    }

    cand = pd.read_csv(args.candidate_pairs, sep='\t', dtype=str).fillna('')
    n_cands = {
        s1_id: len([c for c in str(cids).split(',') if c.strip()])
        for s1_id, cids in zip(cand['source1_entity_id'],
                                cand['candidate_entity_ids'])
    }

    os.makedirs(args.output_dir, exist_ok=True)
    chunk_files, chunk_num, total = [], 0, 0
    buffer = []

    for s1_id, cids_str in zip(cand['source1_entity_id'],
                                cand['candidate_entity_ids']):
        if s1_id not in s1_idx.index:
            continue
        s1r = s1_idx.loc[s1_id]
        true_matches = gt_labels.get(s1_id, set())

        for cid in str(cids_str).split(','):
            cid = cid.strip()
            if not cid or cid not in s23_idx.index:
                continue
            cr = s23_idx.loc[cid]
            buffer.append({
                's1_id': s1_id, 'c_id': cid,
                's1_name': s1r['name_norm'], 'c_name': cr['name_norm'],
                's1_name_tok': s1r['name_tokens'], 'c_name_tok': cr['name_tokens'],
                's1_addr': s1r['addr_norm'], 'c_addr': cr['addr_norm'],
                's1_addr_tok': s1r['addr_tokens'], 'c_addr_tok': cr['addr_tokens'],
                's1_country': s1r['country'], 'c_country': cr['country'],
                'n_candidates': n_cands.get(s1_id, 1),
                'label': int(cid in true_matches),
            })

            if len(buffer) >= args.chunk_size:
                df = process_chunk(buffer, args.workers)
                path = os.path.join(args.output_dir, f'chunk_{chunk_num:04d}.parquet')
                df.to_parquet(path, index=False)
                chunk_files.append(path)
                total += len(df)
                chunk_num += 1
                print(f'chunk {chunk_num:3d}: {total:,} pairs, '
                      f'pos rate {df["label"].mean():.4f}', flush=True)
                buffer = []
                del df
                gc.collect()

    if buffer:
        df = process_chunk(buffer, args.workers)
        path = os.path.join(args.output_dir, f'chunk_{chunk_num:04d}.parquet')
        df.to_parquet(path, index=False)
        chunk_files.append(path)
        total += len(df)
        print(f'chunk {chunk_num + 1:3d}: {total:,} pairs total', flush=True)
        del df
        gc.collect()

    print('merging...', flush=True)
    out_path = os.path.join(args.output_dir, 'features_train.parquet')
    pd.concat([pd.read_parquet(f) for f in chunk_files],
              ignore_index=True).to_parquet(out_path, index=False)

    for f in chunk_files:
        os.remove(f)

    check = pd.read_parquet(out_path)
    print(f'\nDone. {len(check):,} rows, {len(FEATURE_COLS)} features')
    print(f'Positive rate : {check["label"].mean():.5f}')
    print(f'Saved         : {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir',       default='data/normalized')
    parser.add_argument('--data-dir',        default='data')
    parser.add_argument('--candidate-pairs', default='output/candidate_pairs.tsv')
    parser.add_argument('--output-dir',      default='output/features')
    parser.add_argument('--chunk-size',      type=int, default=500_000)
    parser.add_argument('--workers',         type=int, default=8)
    args = parser.parse_args()

    main(args)

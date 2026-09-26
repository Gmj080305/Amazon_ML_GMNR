"""
src/block.py
Stage 3 — Blocking / Candidate Generation

Composite-key blocking. Each record emits keys built from its rarest tokens
and character q-grams; two records become candidates only if they share one.
Selectivity comes from the keys themselves, so candidate sets stay small
without truncating a ranked list.

Why this shape:

  Scales. The step is an exact equality join on an integer key, so it maps
  straight onto a distributed hash join. Nothing compares a record against
  every other record, and nothing holds an N x M structure.

  Small candidate sets. A record with a distinctive name yields a handful of
  candidates rather than a fixed quota.

  Memory. Tokens are factorised to ints once and each key is packed into a
  single int64, so the join runs over two numpy columns. An earlier
  inverted-index version stored postings as Python tuples of ID strings and
  exhausted 64 GB: forked workers copy-on-write those objects as CPython
  touches their refcounts, so each worker duplicated the index.

Key design. Pair-only token keys recovered 82% of true pairs. Diagnosing the
misses gave four gaps, each with a key type here:

  common tokens - 78% of misses shared name tokens, but none rare enough to
                  survive the bucket filter: a pair like (general, electronics)
                  sits in a huge bucket and gets dropped. Lengthening the key
                  restores selectivity, so triples and the full token set are
                  emitted alongside pairs. Raising max_bucket instead would
                  keep the pairs but inflate every candidate set.
  typos         - "traders" and "tarrness" share no token, so no token key can
                  connect them. Character q-grams can.
  short names   - a one-token name emits no pair, so single-token keys are
                  emitted too, while the token is rare enough to stay selective.
  unstable rank - an extra word in one source displaces shared tokens out of
                  the rarest-k window, so k is wide and all pairs within it
                  are emitted.

Country is only a partition key: the groups are whatever distinct values
appear, so an unseen country in the test set partitions like any other.
Nothing branches on a country name.

Usage:
  python src/block.py
  python src/block.py --max-bucket 100 --max-candidates 80
  python src/block.py --split test
"""

import os
import gc
import argparse
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd


# A key is two ids packed into one int64, tagged by key type:
#   key = (type << 58) | (lo << 29) | hi
ID_BITS = 29
ID_MASK = (1 << ID_BITS) - 1
PAYLOAD_MASK = (1 << 58) - 1

# Id spaces are offset so the same spelling in different fields never
# collides inside a key.
NAME_BASE = 0
ADDR_BASE = 1 << 26
PIN_BASE  = 2 << 26
QGRAM_BASE = 3 << 26

# How many of the rarest items per field feed key construction. Wider windows
# cost keys but survive one source carrying an extra word.
N_RARE_NAME = 4
N_RARE_ADDR = 3
N_RARE_QGRAM = 3

QGRAM_SIZE = 4

# A single-token key is only worth emitting while the token is rare enough to
# stay selective on its own; the bucket filter enforces the rest.
SINGLE_TOKEN_MAX_DF = 200

# Cap on the full-token-set key so a very long name does not make it unique to
# one record and therefore useless as a key.
MAX_FULL_NAME_TOKENS = 8


# ── Vocabulary ─────────────────────────────────────────────────────────────

def qgrams(text, q=QGRAM_SIZE):
    text = str(text).replace(' ', '')
    return {text[i:i + q] for i in range(len(text) - q + 1)}


def build_vocabulary(pool):
    """Item -> id and item -> document frequency, over the pool being searched.
    Frequencies decide which items count as rare."""
    name_df, addr_df, qgram_df = Counter(), Counter(), Counter()

    for nt, at, nn in zip(pool['name_tokens'], pool['addr_tokens'],
                          pool['name_norm']):
        name_df.update(t for t in set(str(nt).split()) if len(t) >= 3)
        addr_df.update(t for t in set(str(at).split()) if len(t) >= 3)
        qgram_df.update(qgrams(nn))

    pins = {p for p in pool['pin_zip'].astype(str).unique()
            if p not in ('', 'nan', 'None')}

    return {
        'name':  ({t: i for i, t in enumerate(name_df)},  name_df),
        'addr':  ({t: i for i, t in enumerate(addr_df)},  addr_df),
        'qgram': ({g: i for i, g in enumerate(qgram_df)}, qgram_df),
        'pin':   ({p: i for i, p in enumerate(pins)},     None),
    }


def rarest(items, vocab, df, base, n):
    """The n least frequent items of a record, as offset ids, rarest first.
    Items absent from the pool vocabulary are skipped: they cannot match
    anything, so a key built on one would only ever be a singleton."""
    scored = [(df[x], vocab[x]) for x in items if x in vocab]
    if not scored:
        return [], []
    scored.sort()
    return ([base + i for _, i in scored[:n]],
            [d for d, _ in scored[:n]])


# ── Key generation ─────────────────────────────────────────────────────────

def pack(key_type, a, b=0):
    lo, hi = (a, b) if a <= b else (b, a)
    return (key_type << 58) | ((lo & ID_MASK) << ID_BITS) | (hi & ID_MASK)


def pack_multi(key_type, ids):
    """Key over three or more ids, which will not fit in two 29-bit slots.
    Hashing a sorted tuple of ints is deterministic across processes (only
    str and bytes hashing is randomised), and a collision at this width costs
    at most one spurious candidate."""
    return (key_type << 58) | (hash(tuple(sorted(ids))) & PAYLOAD_MASK)


def record_keys(name_tokens, addr_tokens, name_norm, pin, vocabs):
    """Every blocking key this record carries. Pair ids are sorted before
    packing, so word-order variants between sources land on the same key."""
    name_vocab, name_df = vocabs['name']
    addr_vocab, addr_df = vocabs['addr']
    qgram_vocab, qgram_df = vocabs['qgram']
    pin_vocab, _ = vocabs['pin']

    names, name_dfs = rarest(
        {t for t in str(name_tokens).split() if len(t) >= 3},
        name_vocab, name_df, NAME_BASE, N_RARE_NAME)
    addrs, _ = rarest(
        {t for t in str(addr_tokens).split() if len(t) >= 3},
        addr_vocab, addr_df, ADDR_BASE, N_RARE_ADDR)
    grams, _ = rarest(
        qgrams(name_norm), qgram_vocab, qgram_df, QGRAM_BASE, N_RARE_QGRAM)

    keys = set()

    for tid, df in zip(names, name_dfs):
        if df <= SINGLE_TOKEN_MAX_DF:
            keys.add(pack(0, tid))

    for a, b in combinations(names, 2):
        keys.add(pack(1, a, b))

    for a in names[:2]:
        for b in addrs[:2]:
            keys.add(pack(2, a, b))

    if pin:
        pid = pin_vocab.get(pin)
        if pid is not None:
            for a in names[:2]:
                keys.add(pack(3, a, PIN_BASE + pid))

    for a, b in combinations(addrs, 2):
        keys.add(pack(4, a, b))

    for a, b in combinations(grams, 2):
        keys.add(pack(5, a, b))

    # Triples and the full token set carry the pairs that are individually
    # common. Three ordinary words co-occurring is rare even when each is not.
    for combo in combinations(names, 3):
        keys.add(pack_multi(6, combo))

    all_names, _ = rarest(
        {t for t in str(name_tokens).split() if len(t) >= 3},
        name_vocab, name_df, NAME_BASE, MAX_FULL_NAME_TOKENS)
    if len(all_names) >= 2:
        keys.add(pack_multi(7, all_names))

    for a, b in combinations(names[:3], 2):
        if addrs:
            keys.add(pack_multi(8, (a, b, addrs[0])))

    for combo in combinations(addrs, 3):
        keys.add(pack_multi(9, combo))

    return keys


def build_key_table(df, vocabs):
    """Long-form (row index, key) table for a set of records."""
    rows, keys = [], []
    for i, (nt, at, nn, pin) in enumerate(zip(df['name_tokens'],
                                              df['addr_tokens'],
                                              df['name_norm'],
                                              df['pin_zip'])):
        pin = str(pin)
        if pin in ('', 'nan', 'None'):
            pin = None
        for k in record_keys(nt, at, nn, pin, vocabs):
            rows.append(i)
            keys.append(k)
    return pd.DataFrame({'row': np.asarray(rows, dtype=np.int32),
                         'key': np.asarray(keys, dtype=np.int64)})


# ── Join ───────────────────────────────────────────────────────────────────

def drop_common_keys(key_table, max_bucket):
    """Discard keys held by more than max_bucket pool records. A key that
    common is not evidence of anything and would dominate both join size and
    candidate count; the other key types still cover genuine pairs."""
    sizes = key_table.groupby('key', sort=False).size()
    keep = sizes.index[sizes.values <= max_bucket]
    return key_table[key_table['key'].isin(keep)]


def join_chunked(s1_keys, pool_keys, s1_ids, pool_ids, chunk_rows=2_000_000):
    """Inner join on key, chunked over the Source 1 side so the intermediate
    result never has to fit in memory whole."""
    candidates = {}
    for start in range(0, len(s1_keys), chunk_rows):
        merged = s1_keys.iloc[start:start + chunk_rows].merge(
            pool_keys, on='key', how='inner', suffixes=('_s1', '_pool'))
        if merged.empty:
            continue
        for s1_row, pool_row in zip(merged['row_s1'].values,
                                    merged['row_pool'].values):
            candidates.setdefault(s1_ids[s1_row], set()).add(pool_ids[pool_row])
        del merged
    return candidates


# ── Refinement ─────────────────────────────────────────────────────────────

def cap_by_overlap(candidates, s1_lookup, pool_lookup, max_candidates):
    """Trim entities whose keys proved less selective than expected, ranking
    by shared-token count over name and address. Needs no extra index."""
    capped = {}
    for s1_id, cands in candidates.items():
        if len(cands) <= max_candidates:
            capped[s1_id] = cands
            continue
        name_a, addr_a = s1_lookup[s1_id]
        name_a, addr_a = set(str(name_a).split()), set(str(addr_a).split())
        scored = []
        for cid in cands:
            name_b, addr_b = pool_lookup[cid]
            scored.append((
                len(name_a & set(str(name_b).split())) * 2
                + len(addr_a & set(str(addr_b).split())),
                cid))
        scored.sort(reverse=True)
        capped[s1_id] = {cid for _, cid in scored[:max_candidates]}
    return capped


# ── Main ───────────────────────────────────────────────────────────────────

def build_candidates(s1, pool, max_bucket=50, max_candidates=100):
    """Partition by country, then block within each partition. Country is only
    a partition key, so a value never seen in training needs no handling."""
    all_candidates = {}

    for country in sorted(s1['country'].unique()):
        s1_c   = s1[s1['country'] == country].reset_index(drop=True)
        pool_c = pool[pool['country'] == country].reset_index(drop=True)
        if pool_c.empty or s1_c.empty:
            print(f'\n{country}: {len(s1_c):,} S1, {len(pool_c):,} pool  (skipped)')
            continue

        print(f'\n{country}: {len(s1_c):,} S1, {len(pool_c):,} pool')

        print('  vocabulary...', flush=True)
        vocabs = build_vocabulary(pool_c)
        print(f'    {len(vocabs["name"][0]):,} name tokens, '
              f'{len(vocabs["addr"][0]):,} addr tokens, '
              f'{len(vocabs["qgram"][0]):,} q-grams, '
              f'{len(vocabs["pin"][0]):,} pins')

        print('  keys...', flush=True)
        pool_keys = build_key_table(pool_c, vocabs)
        s1_keys   = build_key_table(s1_c, vocabs)
        print(f'    pool {len(pool_keys):,}, S1 {len(s1_keys):,}')

        before = pool_keys['key'].nunique()
        pool_keys = drop_common_keys(pool_keys, max_bucket)
        print(f'    {before:,} distinct -> {pool_keys["key"].nunique():,} '
              f'after dropping buckets over {max_bucket}')

        print('  joining...', flush=True)
        found = join_chunked(s1_keys, pool_keys,
                             s1_c['entity_id'].values,
                             pool_c['entity_id'].values)

        if max_candidates:
            found = cap_by_overlap(
                found,
                dict(zip(s1_c['entity_id'],
                         zip(s1_c['name_tokens'], s1_c['addr_tokens']))),
                dict(zip(pool_c['entity_id'],
                         zip(pool_c['name_tokens'], pool_c['addr_tokens']))),
                max_candidates)

        pairs = sum(len(v) for v in found.values())
        print(f'    {len(found):,} entities with candidates, {pairs:,} pairs, '
              f'mean {pairs / max(len(found), 1):.1f}')

        all_candidates.update(found)
        del pool_keys, s1_keys, vocabs
        gc.collect()

    return all_candidates


# ── Metrics ────────────────────────────────────────────────────────────────

def print_gate_metrics(candidates, ground_truth, n_s1):
    gt = {}
    for s1_id, matched in zip(ground_truth['source1_entity_id'],
                              ground_truth['matched_entity_ids']):
        ids = {x.strip() for x in str(matched).split(',') if x.strip()}
        if ids:
            gt[s1_id] = ids

    total = sum(len(v) for v in gt.values())
    found = sum(len(v & candidates.get(s1_id, set())) for s1_id, v in gt.items())
    recall = found / total if total else 0.0

    counts = [len(v) for v in candidates.values()] or [0]
    mean_c = sum(counts) / n_s1

    print('\n── Gate metrics ──────────────────────────────')
    print(f'{"PASS" if recall >= 0.985 else "FAIL"}  recall          {recall:.4f}   ({found:,}/{total:,})   gate >= 0.985')
    print(f'{"ok  " if mean_c <= 200 else "warn"}  mean candidates {mean_c:.1f}   gate <= 200, lower is better')
    print(f'{"ok  " if max(counts) <= 600 else "warn"}  max candidates  {max(counts):,}   gate <= 600')
    print(f'      total pairs     {sum(counts):,}')
    print(f'      S1 with 0 cands {n_s1 - len(candidates):,}')


def save_candidate_pairs(candidates, all_s1_ids, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for s1_id in all_s1_ids:
            f.write(f'{s1_id}\t{",".join(sorted(candidates.get(s1_id, ())))}\n')
    print(f'saved {path}  ({len(all_s1_ids):,} rows)')


# ── Entry point ────────────────────────────────────────────────────────────

COLS = ['entity_id', 'country', 'name_norm', 'name_tokens',
        'addr_tokens', 'pin_zip']

FILES = {
    'train': ('train/norm_s1.tsv', 'train/norm_s2.tsv', 'train/norm_s3.tsv'),
    'test':  ('test/norm_test_s1.tsv', 'test/norm_test_s2.tsv',
              'test/norm_test_s3.tsv'),
}


def load(path):
    return pd.read_csv(path, sep='\t', dtype=str, usecols=COLS).fillna('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir',      default='data/normalized')
    parser.add_argument('--output-dir',     default='output')
    parser.add_argument('--data-dir',       default='data')
    parser.add_argument('--split',          default='train',
                        choices=['train', 'test'])
    parser.add_argument('--max-bucket',     type=int, default=50)
    parser.add_argument('--max-candidates', type=int, default=100)
    args = parser.parse_args()

    f1, f2, f3 = FILES[args.split]
    s1   = load(f'{args.input_dir}/{f1}')
    pool = pd.concat([load(f'{args.input_dir}/{f2}'),
                      load(f'{args.input_dir}/{f3}')], ignore_index=True)

    candidates = build_candidates(s1, pool,
                                  max_bucket=args.max_bucket,
                                  max_candidates=args.max_candidates)

    if args.split == 'train':
        gt = pd.read_csv(f'{args.data_dir}/train/train_ground_truth.tsv',
                         sep='\t', dtype=str).fillna('')
        print_gate_metrics(candidates, gt, len(s1))

    save_candidate_pairs(candidates, s1['entity_id'].tolist(),
                         os.path.join(args.output_dir, 'candidate_pairs.tsv'))
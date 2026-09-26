"""
diagnose_blocking.py — why do true pairs fail to share a blocking key?

Samples true-match pairs that blocking missed and asks what the two records
actually have in common. The answer decides the fix:

  shares rare tokens        -> key construction is too narrow, add key types
  shares only common tokens -> need a selectivity-preserving fallback
  shares q-grams not tokens -> typo noise, token keys cannot help
  shares nothing            -> unrecoverable without a different signal

Run:  python src/diagnose_blocking.py
"""

import random
import pandas as pd
from collections import Counter

COLS = ['entity_id', 'country', 'name_norm', 'name_tokens',
        'addr_tokens', 'pin_zip']
SAMPLE = 2000
RARE_DF = 50


def load(path):
    return pd.read_csv(path, sep='\t', dtype=str, usecols=COLS).fillna('')


def qgrams(text, q=4):
    text = str(text).replace(' ', '')
    return {text[i:i + q] for i in range(len(text) - q + 1)}


print('loading...', flush=True)
s1 = load('data/normalized/train/norm_s1.tsv')
pool = pd.concat([load('data/normalized/train/norm_s2.tsv'),
                  load('data/normalized/train/norm_s3.tsv')],
                 ignore_index=True)
gt = pd.read_csv('data/train/train_ground_truth.tsv', sep='\t',
                 dtype=str).fillna('')
cand = pd.read_csv('output/candidate_pairs.tsv', sep='\t',
                   dtype=str).fillna('')

candidates = {
    r: set(c.split(',')) if c else set()
    for r, c in zip(cand['source1_entity_id'], cand['candidate_entity_ids'])
}

s1_lookup = s1.set_index('entity_id')[['name_tokens', 'addr_tokens',
                                        'name_norm', 'pin_zip']].to_dict('index')
pool_lookup = pool.set_index('entity_id')[['name_tokens', 'addr_tokens',
                                            'name_norm', 'pin_zip']].to_dict('index')

print('document frequencies...', flush=True)
name_df = Counter()
addr_df = Counter()
for nt, at in zip(pool['name_tokens'], pool['addr_tokens']):
    name_df.update(set(str(nt).split()))
    addr_df.update(set(str(at).split()))

missed = []
for s1_id, matched in zip(gt['source1_entity_id'], gt['matched_entity_ids']):
    got = candidates.get(s1_id, set())
    for cid in str(matched).split(','):
        cid = cid.strip()
        if cid and cid not in got and s1_id in s1_lookup and cid in pool_lookup:
            missed.append((s1_id, cid))

print(f'{len(missed):,} missed true pairs, sampling {SAMPLE:,}\n')
random.seed(42)
sample = random.sample(missed, min(SAMPLE, len(missed)))

stats = Counter()
examples = []

for s1_id, cid in sample:
    a, b = s1_lookup[s1_id], pool_lookup[cid]

    a_name = set(str(a['name_tokens']).split())
    b_name = set(str(b['name_tokens']).split())
    a_addr = set(str(a['addr_tokens']).split())
    b_addr = set(str(b['addr_tokens']).split())

    shared_name = a_name & b_name
    shared_addr = a_addr & b_addr
    rare_shared = {t for t in shared_name if name_df[t] <= RARE_DF}
    rare_addr   = {t for t in shared_addr if addr_df[t] <= RARE_DF}
    shared_q    = qgrams(a['name_norm']) & qgrams(b['name_norm'])

    if len(rare_shared) >= 2:
        stats['2+ rare name tokens shared'] += 1
    elif len(rare_shared) == 1 and rare_addr:
        stats['1 rare name + rare addr'] += 1
    elif len(rare_shared) == 1:
        stats['exactly 1 rare name token'] += 1
    elif shared_name:
        stats['only common name tokens'] += 1
    elif len(shared_q) >= 3:
        stats['no tokens, but q-grams'] += 1
    elif shared_addr:
        stats['no name overlap, addr only'] += 1
    else:
        stats['nothing shared'] += 1
        if len(examples) < 8:
            examples.append((a['name_norm'], b['name_norm']))

    if len(a_name) < 2 or len(b_name) < 2:
        stats['(one side has <2 name tokens)'] += 1

total = len(sample)
for k, v in stats.most_common():
    print(f'  {k:32s} {v:>6,}  ({v / total:.1%})')

if examples:
    print('\n  pairs sharing nothing:')
    for a, b in examples:
        print(f'    {a[:45]:47s} | {b[:45]}')

print("""
  reading this:
    "1 rare name token" or "1 rare + rare addr" dominant
        -> add single-token and name-plus-address keys
    "only common name tokens" dominant
        -> rarest-token selection is unstable, widen to more tokens
    "no tokens, but q-grams" dominant
        -> typo noise, add character q-gram keys
    "(one side has <2 name tokens)" high
        -> pair-only keys miss short names, single-token keys needed
""")
"""
src/analyze.py
Where is F_0.5 being lost?

Splits the gap to 1.0 into:
  blocking loss - what a perfect classifier on the candidate set still misses
  model loss    - the difference between that perfect classifier and ours

Also measures the one-to-one constraint: Source 1 is deduplicated, so each
S2/S3 record matches at most one S1 entity. Keeping only the most probable
assignment per candidate removes merges that cannot all be right.

Run from the repo root:
  python src/analyze.py
  python src/analyze.py --threshold 0.6
"""

import argparse
import pandas as pd

from evaluate import macro_f05
from train import val_ground_truth, predictions_at


def oracle_predictions(val):
    hits = val[val['label'] == 1]
    return hits.groupby('source1_entity_id')['candidate_id'].agg(set).to_dict()


def one_to_one(val):
    """Keep each candidate only on the S1 entity that scores it highest."""
    best = val.sort_values('prob', ascending=False).drop_duplicates('candidate_id')
    return best


def error_breakdown(gt, preds):
    counts = {'singleton false merge': 0, 'entity with a false positive': 0,
              'entity with a missed match': 0}
    for s1, true_ids in gt.items():
        pred_ids = preds.get(s1, set())
        if not true_ids:
            if pred_ids:
                counts['singleton false merge'] += 1
            continue
        if pred_ids - true_ids:
            counts['entity with a false positive'] += 1
        if true_ids - pred_ids:
            counts['entity with a missed match'] += 1
    return counts


def main(args):
    val = pd.read_parquet(args.val_predictions)
    gt = val_ground_truth(args.ground_truth, args.val_fraction)

    oracle = macro_f05(gt, oracle_predictions(val))
    current = macro_f05(gt, predictions_at(val, args.threshold))
    deduped = macro_f05(gt, predictions_at(one_to_one(val), args.threshold))

    print(f'\n── F_0.5 breakdown (threshold {args.threshold}) ─────────────')
    print(f'  ceiling (perfect model on candidates) : {oracle:.4f}')
    print(f'  current model                         : {current:.4f}')
    print(f'  current + one-to-one constraint       : {deduped:.4f}')
    print(f'\n  lost to blocking : {1 - oracle:.4f}')
    print(f'  lost to model    : {oracle - current:.4f}')

    print(f'\n  error counts at this threshold ({len(gt):,} val entities):')
    for name, n in error_breakdown(gt, predictions_at(val, args.threshold)).items():
        print(f'    {name:30s} {n:>8,}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-predictions', default='output/model/val_predictions.parquet')
    parser.add_argument('--ground-truth',    default='data/train/train_ground_truth.tsv')
    parser.add_argument('--val-fraction',    type=float, default=0.2)
    parser.add_argument('--threshold',       type=float, default=0.5)
    args = parser.parse_args()

    main(args)
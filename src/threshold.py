"""
src/threshold.py
Stage 8 — Threshold optimisation

Sweeps the decision threshold over the saved val predictions and picks the one
that maximises macro F_0.5. Tried with and without the one-to-one constraint:
Source 1 is deduplicated, so each S2/S3 record belongs to at most one S1 entity,
and only its highest-scoring assignment can be right.

Scoring is vectorised: entities are encoded as integers once, and each threshold
is two bincounts, rather than rebuilding prediction sets per threshold.

Run from the repo root:
  python src/threshold.py
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train import val_ground_truth


def one_to_one(val):
    return val.sort_values('prob', ascending=False).drop_duplicates('candidate_id')


def encode(val, gt):
    """Integer entity codes for every val pair, aligned with the gt entity order.
    Entities blocking gave no candidates still exist in gt, so they still count."""
    ids = pd.Index(list(gt))
    codes = ids.get_indexer(val['source1_entity_id'])
    keep = codes >= 0
    true_count = np.array([len(gt[s1]) for s1 in ids])
    return {
        'codes':      codes[keep],
        'prob':       val['prob'].values[keep],
        'label':      val['label'].values[keep].astype(bool),
        'true_count': true_count,
        'n':          len(ids),
    }


def score_at(enc, threshold):
    n = enc['n']
    selected = enc['prob'] >= threshold
    pred = np.bincount(enc['codes'][selected], minlength=n)
    tp   = np.bincount(enc['codes'][selected & enc['label']], minlength=n)
    true = enc['true_count']

    precision = np.divide(tp, pred, out=np.zeros(n), where=pred > 0)
    recall    = np.divide(tp, true, out=np.zeros(n), where=true > 0)
    denom     = 0.25 * precision + recall
    f05 = np.divide(1.25 * precision * recall, denom, out=np.zeros(n), where=denom > 0)

    singleton = true == 0
    f05[singleton] = (pred[singleton] == 0).astype(float)

    return {
        'threshold':          round(threshold, 2),
        'f05':                f05.mean(),
        'precision':          tp.sum() / max(pred.sum(), 1),
        'recall':             tp.sum() / max(true.sum(), 1),
        'singleton_accuracy': (pred[singleton] == 0).mean(),
        'pred': pred, 'tp': tp,
    }


def sweep(enc, thresholds):
    return [score_at(enc, t) for t in thresholds]


def as_table(results, label):
    rows = [{k: r[k] for k in ('threshold', 'f05', 'precision', 'recall',
                               'singleton_accuracy')} for r in results]
    table = pd.DataFrame(rows)
    table['one_to_one'] = label
    return table


def error_breakdown(enc, best):
    """Counts per entity. Blocking misses are separated out, since no threshold
    can recover a pair that was never a candidate."""
    true, pred, tp = enc['true_count'], best['pred'], best['tp']
    reachable = np.bincount(enc['codes'][enc['label']], minlength=enc['n'])
    singleton = true == 0
    return {
        'singleton false merge':          int(((pred > 0) & singleton).sum()),
        'entity with a false positive':   int(((pred - tp > 0) & ~singleton).sum()),
        'entity missing a reachable match': int((reachable - tp > 0).sum()),
        'entity missing a blocked match': int((true - reachable > 0).sum()),
    }


def plot(table, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, style in ((False, '--'), (True, '-')):
        part = table[table['one_to_one'] == label]
        suffix = ' (1-to-1)' if label else ''
        ax.plot(part['threshold'], part['f05'], style, label=f'F0.5{suffix}')
        if label:
            ax.plot(part['threshold'], part['precision'], ':', label='precision (1-to-1)')
            ax.plot(part['threshold'], part['recall'], ':', label='recall (1-to-1)')
    ax.set_xlabel('threshold')
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=120, bbox_inches='tight')


def main(args):
    val = pd.read_parquet(args.val_predictions)
    gt = val_ground_truth(args.ground_truth, args.val_fraction)
    thresholds = np.arange(0.05, 0.975, 0.01)

    enc_raw = encode(val, gt)
    enc_1to1 = encode(one_to_one(val), gt)

    raw = sweep(enc_raw, thresholds)
    constrained = sweep(enc_1to1, thresholds)

    table = pd.concat([as_table(raw, False), as_table(constrained, True)],
                      ignore_index=True)
    best_row = table.loc[table['f05'].idxmax()]
    use_1to1 = bool(best_row['one_to_one'])
    best_enc = enc_1to1 if use_1to1 else enc_raw
    best = score_at(best_enc, best_row['threshold'])
    at_half = score_at(best_enc, 0.5)

    blocking_recall = enc_raw['label'].sum() / enc_raw['true_count'].sum()
    utilisation = best['recall'] / blocking_recall

    os.makedirs(args.output_dir, exist_ok=True)
    table.to_csv(os.path.join(args.output_dir, 'threshold_sweep.csv'), index=False)
    plot(table, os.path.join(args.output_dir, 'threshold_sweep.png'))
    with open(os.path.join(args.output_dir, 'threshold.json'), 'w') as f:
        json.dump({'threshold': float(best['threshold']),
                   'one_to_one': use_1to1,
                   'val_f05': round(float(best['f05']), 4)}, f, indent=2)

    t = best['threshold']
    print('\n── Stage 8 gates ─────────────────────────────')
    print(f'{"PASS" if best["f05"] >= 0.87 else "FAIL"}  val F_0.5 at best      {best["f05"]:.4f}   gate >= 0.87')
    print(f'      threshold / 1-to-1   {t:.2f} / {use_1to1}')
    print(f'{"PASS" if best["singleton_accuracy"] >= 0.96 else "FAIL"}  singleton accuracy     {best["singleton_accuracy"]:.1%}   gate >= 96%')
    print(f'{"PASS" if utilisation >= 0.82 else "FAIL"}  ceiling utilisation    {utilisation:.1%}   gate >= 82%')
    gain = best['f05'] - at_half['f05']
    print(f'{"PASS" if gain >= 0.02 else "WARN"}  gain over 0.50         {gain:+.4f}   gate >= +0.02')
    print(f'{"PASS" if 0.18 <= t <= 0.90 else "WARN"}  threshold in range     {t:.2f}   gate 0.18 – 0.90')
    print(f'      precision / recall   {best["precision"]:.4f} / {best["recall"]:.4f}')

    print(f'\n  errors at {t:.2f} ({enc_raw["n"]:,} val entities):')
    for name, count in error_breakdown(best_enc, best).items():
        print(f'    {name:34s} {count:>8,}')

    print(f'\n  saved to {args.output_dir}/: threshold.json, threshold_sweep.csv, threshold_sweep.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-predictions', default='output/model/val_predictions.parquet')
    parser.add_argument('--ground-truth',    default='data/train/train_ground_truth.tsv')
    parser.add_argument('--output-dir',      default='output/model')
    parser.add_argument('--val-fraction',    type=float, default=0.2)
    args = parser.parse_args()

    main(args)
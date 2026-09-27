"""
src/evaluate.py
F_0.5 evaluation — macro-averaged per Source 1 entity.

Singletons (S1 entities with no true matches) score 1.0 for an empty
prediction and 0.0 for any prediction.

Usage:
  python src/evaluate.py
  python src/evaluate.py --predictions output/matching_results.tsv
  python src/evaluate.py --predictions output/matching_results.tsv --split val
"""

import argparse
import pandas as pd


def f05_entity(true_ids, pred_ids):
    """F_0.5 for a single S1 entity."""
    true_ids = set(true_ids)
    pred_ids = set(pred_ids)

    # singleton: no true matches
    if not true_ids:
        return 1.0 if not pred_ids else 0.0

    if not pred_ids:
        return 0.0

    tp = len(true_ids & pred_ids)
    precision = tp / len(pred_ids)
    recall    = tp / len(true_ids)

    if precision == 0 and recall == 0:
        return 0.0

    return (1.25 * precision * recall) / (0.25 * precision + recall)


def macro_f05(ground_truth, predictions):
    """
    Macro-averaged F_0.5 across all S1 entities in ground_truth.

    ground_truth : dict  s1_id -> set of true matched ids (empty set for singletons)
    predictions  : dict  s1_id -> set of predicted matched ids
    """
    scores = []
    for s1_id, true_ids in ground_truth.items():
        pred_ids = predictions.get(s1_id, set())
        scores.append(f05_entity(true_ids, pred_ids))
    return sum(scores) / len(scores) if scores else 0.0


def load_ground_truth(path):
    gt = pd.read_csv(path, sep='\t', dtype=str).fillna('')
    result = {}
    for s1_id, matched in zip(gt['source1_entity_id'], gt['matched_entity_ids']):
        ids = {x.strip() for x in str(matched).split(',') if x.strip()}
        result[s1_id] = ids
    return result


def load_predictions(path):
    pred = pd.read_csv(path, sep='\t', dtype=str).fillna('')
    result = {}
    for s1_id, matched in zip(pred['source1_entity_id'], pred['matched_entity_ids']):
        ids = {x.strip() for x in str(matched).split(',') if x.strip()}
        result[s1_id] = ids
    return result


def print_report(ground_truth, predictions):
    scores = []
    tp_total = fp_total = fn_total = 0

    for s1_id, true_ids in ground_truth.items():
        pred_ids = predictions.get(s1_id, set())
        scores.append(f05_entity(true_ids, pred_ids))
        tp_total += len(true_ids & pred_ids)
        fp_total += len(pred_ids - true_ids)
        fn_total += len(true_ids - pred_ids)

    n_singletons   = sum(1 for v in ground_truth.values() if not v)
    n_correct_sing = sum(
        1 for s1_id, v in ground_truth.items()
        if not v and not predictions.get(s1_id)
    )

    macro = sum(scores) / len(scores)
    micro_p = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0
    micro_r = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0

    print(f'\n── Evaluation results ────────────────────────')
    print(f'  Macro F_0.5         : {macro:.4f}')
    print(f'  Micro precision     : {micro_p:.4f}')
    print(f'  Micro recall        : {micro_r:.4f}')
    print(f'  True positives      : {tp_total:,}')
    print(f'  False positives     : {fp_total:,}')
    print(f'  False negatives     : {fn_total:,}')
    print(f'  Singletons correct  : {n_correct_sing:,} / {n_singletons:,}  '
          f'({n_correct_sing / max(n_singletons, 1):.1%})')
    print(f'  S1 entities scored  : {len(scores):,}')

    return macro


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground-truth', default='data/train/train_ground_truth.tsv')
    parser.add_argument('--predictions',  default='output/matching_results.tsv')
    args = parser.parse_args()

    gt   = load_ground_truth(args.ground_truth)
    pred = load_predictions(args.predictions)

    print_report(gt, pred)
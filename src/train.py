"""
src/train.py
Stage 7 — Model training

Trains a LightGBM classifier on the train split and scores every val pair.
Val probabilities are saved so Stage 8 can sweep thresholds without retraining.

No class weighting: training negatives are already downsampled to 10:1, and
upweighting positives would push the model toward recall, the opposite of
what F_0.5 rewards. Stage 8 corrects the threshold for the shifted base rate.

Run from the repo root:
  python src/train.py
  python src/train.py --train-sample 0.25     # faster iteration
"""

import os
import argparse

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from split import FEATURE_COLS, is_val
from evaluate import macro_f05, load_ground_truth


PARAMS = {
    'objective':        'binary',
    'metric':           ['binary_logloss', 'auc'],
    'learning_rate':    0.05,
    'num_leaves':       127,
    'min_data_in_leaf': 200,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq':     1,
    'lambda_l2':        1.0,
    'verbose':          -1,
}


def load_split(path, extra_cols=(), sample=1.0, seed=42):
    df = pd.read_parquet(path, columns=FEATURE_COLS + ['label', *extra_cols])
    if sample < 1.0:
        df = df.sample(frac=sample, random_state=seed)
    return df


def train_model(train, val_sample, rounds, threads):
    params = {**PARAMS, 'num_threads': threads}
    train_set = lgb.Dataset(train[FEATURE_COLS], train['label'],
                            free_raw_data=True)
    val_set = lgb.Dataset(val_sample[FEATURE_COLS], val_sample['label'],
                          reference=train_set)
    return lgb.train(
        params, train_set,
        num_boost_round=rounds,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )


def predictions_at(val, threshold):
    hits = val[val['prob'] >= threshold]
    return hits.groupby('source1_entity_id')['candidate_id'].agg(set).to_dict()


def val_ground_truth(path, val_fraction):
    # Includes val entities that blocking gave no candidates, so they still count.
    gt = load_ground_truth(path)
    ids = list(gt)
    mask = is_val(ids, val_fraction)
    return {s1: gt[s1] for s1, keep in zip(ids, mask) if keep}


def importance_table(model):
    gain = model.feature_importance(importance_type='gain')
    table = pd.Series(gain, index=FEATURE_COLS).sort_values(ascending=False)
    return table / table.sum()


def main(args):
    train = load_split(args.train_dir, sample=args.train_sample)
    val = load_split(args.val_dir, extra_cols=['source1_entity_id', 'candidate_id'])
    print(f'train {len(train):,} pairs, val {len(val):,} pairs')

    # Early stopping on a val sample; scoring all 22M rows every round is slow.
    val_sample = val.sample(n=min(2_000_000, len(val)), random_state=42)

    train_auc_sample = train.sample(n=min(2_000_000, len(train)), random_state=42)
    model = train_model(train, val_sample, args.rounds, args.threads)
    del train

    val['prob'] = model.predict(val[FEATURE_COLS], num_threads=args.threads)
    train_auc = roc_auc_score(
        train_auc_sample['label'],
        model.predict(train_auc_sample[FEATURE_COLS], num_threads=args.threads))
    val_auc = roc_auc_score(val['label'], val['prob'])

    gt_val = val_ground_truth(args.ground_truth, args.val_fraction)
    f05 = macro_f05(gt_val, predictions_at(val, 0.5))
    baseline = sum(1 for v in gt_val.values() if not v) / len(gt_val)

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    model.save_model(args.model_path)
    val[['source1_entity_id', 'candidate_id', 'label', 'prob']].to_parquet(
        args.val_predictions, index=False)

    importance = importance_table(model)
    top3 = importance.head(3).sum()

    print('\n── Stage 7 gates ─────────────────────────────')
    print(f'{"PASS" if f05 >= 0.83 else "FAIL"}  val F_0.5 @ 0.50  {f05:.4f}   gate >= 0.83')
    print(f'      singleton base   {baseline:.4f}')
    print(f'{"PASS" if train_auc - val_auc < 0.05 else "WARN"}  AUC train / val   {train_auc:.4f} / {val_auc:.4f}   gap < 0.05')
    print(f'{"PASS" if top3 < 0.70 else "WARN"}  top-3 importance  {top3:.1%}   gate < 70%')
    print(f'      best iteration   {model.best_iteration}')
    print('\n  feature importance (gain share):')
    for name, share in importance.items():
        print(f'    {name:16s} {share:6.1%}')
    print(f'\n  model           -> {args.model_path}')
    print(f'  val predictions -> {args.val_predictions}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir',       default='output/features/train')
    parser.add_argument('--val-dir',         default='output/features/val')
    parser.add_argument('--ground-truth',    default='data/train/train_ground_truth.tsv')
    parser.add_argument('--model-path',      default='output/model/model.lgb')
    parser.add_argument('--val-predictions', default='output/model/val_predictions.parquet')
    parser.add_argument('--val-fraction',    type=float, default=0.2)
    parser.add_argument('--train-sample',    type=float, default=1.0)
    parser.add_argument('--rounds',          type=int,   default=2000)
    parser.add_argument('--threads',         type=int,   default=8)
    args = parser.parse_args()

    main(args)
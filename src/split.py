"""
src/split.py
Stage 6 — Train / validation split

Splits the feature matrix by S1 entity, never by pair: every pair for a given
S1 entity lands in the same side. A pair-level split would put near-identical
rows on both sides and make validation optimistic.

Assignment is by hash of the entity id rather than a shuffled list, so the
split is reproducible, needs no global list of ids in memory, and stays stable
if chunks are regenerated.

Training negatives are downsampled to at most NEG_PER_POS per positive.
Validation keeps every pair, because F_0.5 there has to reflect the real
candidate distribution the model will face at inference.

Reads the merged features_train.parquet if present, otherwise the chunk_*.parquet
files directly. Writes parts rather than one file, so nothing has to hold the
whole matrix at once.

Usage:
  python src/split.py
  python src/split.py --val-fraction 0.2 --neg-per-pos 10
"""

import os
import glob
import hashlib
import argparse

import numpy as np
import pandas as pd

from features import FEATURE_COLS

ID_COLS = ['source1_entity_id', 'candidate_id']


def val_bucket(entity_id, n_buckets=100):
    """Stable bucket for an entity id, independent of Python's hash seed."""
    digest = hashlib.md5(str(entity_id).encode()).digest()
    return int.from_bytes(digest[:4], 'big') % n_buckets


def is_val(entity_ids, val_fraction):
    cutoff = int(val_fraction * 100)
    return np.array([val_bucket(e) < cutoff for e in entity_ids])


def downsample_negatives(df, neg_per_pos, rng):
    """Keep every positive, cap negatives at neg_per_pos per positive."""
    pos = df[df['label'] == 1]
    neg = df[df['label'] == 0]
    keep = min(len(neg), len(pos) * neg_per_pos)
    if keep < len(neg):
        neg = neg.sample(n=keep, random_state=rng)
    return pd.concat([pos, neg], ignore_index=True)


def shrink(df):
    """float64 features to float32 — halves memory, no meaningful precision loss."""
    for col in FEATURE_COLS:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    return df


def input_files(features_dir):
    merged = os.path.join(features_dir, 'features_train.parquet')
    if os.path.exists(merged):
        return [merged]
    chunks = sorted(glob.glob(os.path.join(features_dir, 'chunk_*.parquet')))
    if not chunks:
        raise FileNotFoundError(f'no feature files in {features_dir}')
    return chunks


def singleton_baseline(ground_truth_path, val_fraction):
    """Macro F_0.5 on validation from predicting an empty list for every entity.

    Each singleton scores 1.0 and every other entity scores 0.0, so the value is
    just the singleton fraction. Any trained model has to beat this."""
    gt = pd.read_csv(ground_truth_path, sep='\t', dtype=str).fillna('')
    mask = is_val(gt['source1_entity_id'].values, val_fraction)
    val_gt = gt[mask]
    singletons = (val_gt['matched_entity_ids'].str.strip() == '').sum()
    return singletons / len(val_gt), singletons, len(val_gt)


def main(args):
    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.val_dir, exist_ok=True)

    files = input_files(args.features_dir)
    print(f'reading {len(files)} file(s) from {args.features_dir}')

    rng = args.seed
    train_rows = val_rows = 0
    train_pos = val_pos = 0

    for i, path in enumerate(files):
        df = shrink(pd.read_parquet(path))

        mask = is_val(df['source1_entity_id'].values, args.val_fraction)
        val_part = df[mask]
        train_part = downsample_negatives(df[~mask], args.neg_per_pos, rng + i)

        train_part.to_parquet(
            os.path.join(args.train_dir, f'part_{i:04d}.parquet'), index=False)
        val_part.to_parquet(
            os.path.join(args.val_dir, f'part_{i:04d}.parquet'), index=False)

        train_rows += len(train_part)
        val_rows   += len(val_part)
        train_pos  += int(train_part['label'].sum())
        val_pos    += int(val_part['label'].sum())

        print(f'  part {i + 1}/{len(files)}: '
              f'train {len(train_part):,}, val {len(val_part):,}', flush=True)

        del df, train_part, val_part

    baseline, n_single, n_val_entities = singleton_baseline(
        args.ground_truth, args.val_fraction)

    print(f'\n── Split summary ─────────────────────────────')
    print(f'  train pairs        : {train_rows:,}')
    print(f'  train positives    : {train_pos:,}  ({train_pos / max(train_rows, 1):.4f})')
    print(f'  train neg:pos      : {(train_rows - train_pos) / max(train_pos, 1):.1f}:1')
    print(f'  val pairs          : {val_rows:,}')
    print(f'  val positives      : {val_pos:,}  ({val_pos / max(val_rows, 1):.4f})')
    print(f'  val S1 entities    : {n_val_entities:,}')
    print(f'  val singletons     : {n_single:,}  ({n_single / n_val_entities:.4f})')
    print(f'\n  singleton baseline F_0.5 on val : {baseline:.4f}')
    print(f'  (the model must beat this)')
    print(f'\n  train -> {args.train_dir}/')
    print(f'  val   -> {args.val_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-dir',  default='output/features')
    parser.add_argument('--train-dir',     default='output/features/train')
    parser.add_argument('--val-dir',       default='output/features/val')
    parser.add_argument('--ground-truth',  default='data/train/train_ground_truth.tsv')
    parser.add_argument('--val-fraction',  type=float, default=0.2)
    parser.add_argument('--neg-per-pos',   type=int,   default=10)
    parser.add_argument('--seed',          type=int,   default=42)
    args = parser.parse_args()

    main(args)
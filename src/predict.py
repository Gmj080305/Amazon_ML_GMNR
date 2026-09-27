"""
src/predict.py
Final step — score test candidate pairs and write matching_results.tsv

Reads the model and threshold.json from one model directory, so the same
script produces a submission from any trained model version. The feature list
comes from the saved model itself, which keeps the columns and their order
exactly as the model was trained on.

Run from the repo root:
  python src/predict.py
  python src/predict.py --model-dir output/model_v2
"""

import os
import json
import glob
import argparse

import pandas as pd
import lightgbm as lgb


# Blocking only pairs records within one country, so country_match was 1 on
# every training pair. Models trained with it still expect the column.
CONSTANT_FEATURES = {'country_match': 1.0}


def load_model(model_dir):
    model = lgb.Booster(model_file=os.path.join(model_dir, 'model.lgb'))
    with open(os.path.join(model_dir, 'threshold.json')) as f:
        config = json.load(f)
    return model, config['threshold'], config['one_to_one']


def feature_frame(df, names):
    for col, value in CONSTANT_FEATURES.items():
        if col in names and col not in df:
            df[col] = value
    missing = [c for c in names if c not in df]
    if missing:
        raise ValueError(f'test features lack {missing}; compute them with the '
                         f'same features.py version the model was trained on')
    return df[names]


def score(files, model, threshold):
    """Keep only pairs at or above the threshold; everything else is discarded
    per chunk, so memory holds matches rather than all candidate pairs."""
    names = model.feature_name()
    kept = []
    for path in files:
        df = pd.read_parquet(path)
        prob = model.predict(feature_frame(df, names))
        hits = df.loc[prob >= threshold, ['source1_entity_id', 'candidate_id']].copy()
        hits['prob'] = prob[prob >= threshold]
        kept.append(hits)
    return pd.concat(kept, ignore_index=True)


def one_to_one(hits):
    # Source 1 is deduplicated, so a candidate can belong to one entity at most.
    return hits.sort_values('prob', ascending=False).drop_duplicates('candidate_id')


def write_matches(hits, s1_ids, path):
    matched = hits.groupby('source1_entity_id')['candidate_id'].agg(
        lambda ids: ','.join(sorted(ids)))
    out = pd.DataFrame({'source1_entity_id': s1_ids})
    out['matched_entity_ids'] = out['source1_entity_id'].map(matched).fillna('')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    out.to_csv(path, sep='\t', index=False)
    return out


def main(args):
    model, threshold, use_1to1 = load_model(args.model_dir)
    files = sorted(glob.glob(os.path.join(args.features_dir, 'chunk_*.parquet')))
    if not files:
        raise FileNotFoundError(f'no feature chunks in {args.features_dir}')

    hits = score(files, model, threshold)
    if use_1to1:
        hits = one_to_one(hits)

    # Every test S1 entity needs a row; candidate_pairs.tsv has exactly one each.
    s1_ids = pd.read_csv(args.candidate_pairs, sep='\t', dtype=str,
                         usecols=['source1_entity_id'])['source1_entity_id']
    out = write_matches(hits, s1_ids, args.output)

    matched = (out['matched_entity_ids'] != '').sum()
    print(f'model           : {args.model_dir}  (threshold {threshold}, one-to-one {use_1to1})')
    print(f'S1 entities     : {len(out):,}')
    print(f'with matches    : {matched:,}  ({matched / len(out):.1%})')
    print(f'total matches   : {len(hits):,}')
    print(f'file size       : {os.path.getsize(args.output) / 1e6:.1f} MB')
    print(f'saved           : {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir',       default='output/model')
    parser.add_argument('--features-dir',    default='output/features_v2/test')
    parser.add_argument('--candidate-pairs', default='output/test/candidate_pairs.tsv')
    parser.add_argument('--output',          default='output/test/matching_results.tsv')
    args = parser.parse_args()

    main(args)
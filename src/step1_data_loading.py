"""
step1_data_loading.py
---------------------
Loads and preprocesses the MovieLens 20M dataset for the CTR pipeline.

Outputs (saved to results/):
  - train_matrix.npz    : sparse binary user-item matrix (train)
  - test_matrix.npz     : sparse binary user-item matrix (test)
  - content_matrix.npy  : item content matrix from genome scores
  - cold_test.npz       : cold-start test matrix
  - cold_idx.npy        : indices of cold items
  - cold_content.npy    : genome scores for cold items
  - meta.json           : dataset statistics

Usage:
  python src/step1_data_loading.py --data_dir ./ml-20m
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

RATING_THRESHOLD = 4.0
MIN_USER_RATINGS = 5
MIN_ITEM_RATINGS = 5
COLD_FRAC        = 0.10
RANDOM_STATE     = 42
MAX_USERS        = 5000   # set None for full dataset
MAX_ITEMS        = 3000   # set None for full dataset


def filter_interactions(df):
    while True:
        before = len(df)
        uc = df['userId'].value_counts()
        ic = df['movieId'].value_counts()
        df = df[df['userId'].isin(uc[uc >= MIN_USER_RATINGS].index)]
        df = df[df['movieId'].isin(ic[ic >= MIN_ITEM_RATINGS].index)]
        if len(df) == before:
            break
    return df


def rows_to_csr(rows, n_users, n_items):
    if not rows:
        return csr_matrix((n_users, n_items), dtype=np.float32)
    u, v = zip(*rows)
    return csr_matrix((np.ones(len(rows), np.float32), (u, v)),
                      shape=(n_users, n_items))


def main(data_dir, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)

    # ── Load ratings ──────────────────────────────────────────
    logger.info('Loading ratings.csv ...')
    df = pd.read_csv(os.path.join(data_dir, 'ratings.csv'),
                     usecols=['userId', 'movieId', 'rating'])
    df = df[df['rating'] >= RATING_THRESHOLD][['userId', 'movieId']].drop_duplicates()

    # ── Load genome scores ────────────────────────────────────
    logger.info('Loading genome-scores.csv ...')
    genome_df = pd.read_csv(os.path.join(data_dir, 'genome-scores.csv'))
    genome = genome_df.pivot(index='movieId', columns='tagId',
                             values='relevance').fillna(0.0)

    # ── Filter to items with genome data ──────────────────────
    common_items = sorted(set(df['movieId']) & set(genome.index))
    if MAX_ITEMS:
        common_items = common_items[:MAX_ITEMS]
    df = df[df['movieId'].isin(common_items)]
    df = filter_interactions(df)

    if MAX_USERS:
        df = df[df['userId'].isin(sorted(df['userId'].unique())[:MAX_USERS])]
        df = filter_interactions(df)

    user_ids = sorted(df['userId'].unique())
    item_ids = sorted(df['movieId'].unique())
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {m: i for i, m in enumerate(item_ids)}
    n_users, n_items = len(user_ids), len(item_ids)
    logger.info(f'Users: {n_users}  Items: {n_items}  Interactions: {len(df)}')

    # ── Train / test split ─────────────────────────────────────
    train_rows, test_rows = [], []
    for uid, group in df.groupby('userId'):
        u_idx = user2idx[uid]
        idxs = [item2idx[m] for m in group['movieId']]
        if len(idxs) < 2:
            train_rows.extend([(u_idx, i) for i in idxs])
            continue
        tr, te = train_test_split(idxs, test_size=0.2, random_state=RANDOM_STATE)
        train_rows.extend([(u_idx, i) for i in tr])
        test_rows.extend([(u_idx, i) for i in te])

    train_matrix = rows_to_csr(train_rows, n_users, n_items)
    test_matrix  = rows_to_csr(test_rows,  n_users, n_items)

    # ── Content matrix ─────────────────────────────────────────
    content_matrix = genome.loc[item_ids].values.astype(np.float32)
    rs = content_matrix.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    content_matrix /= rs

    # ── Out-of-matrix split ────────────────────────────────────
    rng = np.random.default_rng(RANDOM_STATE)
    cold_idx = rng.choice(np.arange(n_items),
                          size=int(n_items * COLD_FRAC), replace=False)
    cold_test    = test_matrix[:, cold_idx]
    cold_content = content_matrix[cold_idx]

    # ── Save ───────────────────────────────────────────────────
    save_npz(os.path.join(output_dir, 'train_matrix.npz'), train_matrix)
    save_npz(os.path.join(output_dir, 'test_matrix.npz'),  test_matrix)
    save_npz(os.path.join(output_dir, 'cold_test.npz'),    cold_test)
    np.save(os.path.join(output_dir, 'content_matrix.npy'), content_matrix)
    np.save(os.path.join(output_dir, 'cold_idx.npy'),       cold_idx)
    np.save(os.path.join(output_dir, 'cold_content.npy'),   cold_content)

    meta = dict(n_users=n_users, n_items=n_items,
                train_nnz=int(train_matrix.nnz),
                test_nnz=int(test_matrix.nnz),
                n_cold=int(len(cold_idx)),
                content_shape=list(content_matrix.shape))
    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info('Data saved to results/. Summary:')
    for k, v in meta.items():
        logger.info(f'  {k}: {v}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   default='./ml-20m')
    parser.add_argument('--output_dir', default='results')
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)

"""
step4_evaluation.py
-------------------
Evaluates PMF, LDA, and CTR using Recall@M.

Two evaluation protocols (matching Wang & Blei 2011):
  1. In-matrix  : warm items seen during training
  2. Out-of-matrix : cold items never seen during training
                     CTR predicts using r*_ij = u_i^T θ_j (ε=0)
                     PMF cannot participate here.

Outputs (saved to results/):
  - recall_results.json : all Recall@M scores

Usage:
  python src/step4_evaluation.py
"""

import json
import logging
import os

import numpy as np
from scipy.sparse import load_npz

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

M_LIST = [20, 50, 100, 200, 300]


def recall_at_m(scores, test_matrix, M_list, exclude_matrix=None):
    """
    Recall@M averaged over all users with at least one test item.
    Recall@M = |relevant items in top-M| / |all relevant items|
    """
    results = {M: [] for M in M_list}
    test_csr = test_matrix.tocsr()
    excl_csr = exclude_matrix.tocsr() if exclude_matrix is not None else None

    for i in range(scores.shape[0]):
        rel = set(test_csr.getrow(i).indices)
        if not rel:
            continue
        s = scores[i].copy()
        if excl_csr is not None:
            s[excl_csr.getrow(i).indices] = -np.inf
        for M in M_list:
            top_m = np.argpartition(s, -M)[-M:]
            results[M].append(len(rel & set(top_m)) / len(rel))

    return {M: float(np.mean(v)) for M, v in results.items() if v}


def print_results(results_dict):
    all_M = sorted({m for res in results_dict.values() for m in res})
    header = f"{'Model':<20}" + ''.join(f'  R@{M:<6}' for M in all_M)
    print(header)
    print('-' * len(header))
    for name, res in results_dict.items():
        print(f'{name:<20}' +
              ''.join(f'  {res.get(M, float("nan")):.4f}' for M in all_M))


def main(results_dir='results'):
    # ── Load data ─────────────────────────────────────────────
    train_matrix = load_npz(os.path.join(results_dir, 'train_matrix.npz'))
    test_matrix  = load_npz(os.path.join(results_dir, 'test_matrix.npz'))
    cold_test    = load_npz(os.path.join(results_dir, 'cold_test.npz'))
    theta        = np.load(os.path.join(results_dir, 'theta.npy'))
    cold_theta   = np.load(os.path.join(results_dir, 'cold_theta.npy'))
    pmf_U        = np.load(os.path.join(results_dir, 'pmf_U.npy'))
    pmf_V        = np.load(os.path.join(results_dir, 'pmf_V.npy'))
    ctr_U        = np.load(os.path.join(results_dir, 'ctr_U.npy'))
    ctr_V        = np.load(os.path.join(results_dir, 'ctr_V.npy'))
    n_users      = train_matrix.shape[0]

    # ── LDA content-only baseline ─────────────────────────────
    # User profile = mean θ of training items
    train_csr = train_matrix.tocsr()
    user_profiles = np.zeros((n_users, theta.shape[1]), dtype=np.float32)
    for i in range(n_users):
        idx = train_csr.getrow(i).indices
        if len(idx) > 0:
            user_profiles[i] = theta[idx].mean(axis=0)

    # ── In-matrix evaluation ──────────────────────────────────
    logger.info('Evaluating in-matrix (warm items) ...')
    pmf_in  = recall_at_m(pmf_U @ pmf_V.T, test_matrix, M_LIST,
                           exclude_matrix=train_matrix)
    lda_in  = recall_at_m(user_profiles @ theta.T, test_matrix, M_LIST,
                           exclude_matrix=train_matrix)
    ctr_in  = recall_at_m(ctr_U @ ctr_V.T, test_matrix, M_LIST,
                           exclude_matrix=train_matrix)

    print('\n--- In-Matrix Recall@M ---')
    print_results({'PMF': pmf_in, 'LDA': lda_in, 'CTR': ctr_in})

    # ── Out-of-matrix evaluation (cold-start) ─────────────────
    logger.info('Evaluating out-of-matrix (cold-start) ...')
    # CTR: r*_ij = u_i^T θ_j  (ε = 0 for cold items)
    ctr_oom = recall_at_m(ctr_U @ cold_theta.T, cold_test, M_LIST)
    lda_oom = recall_at_m(user_profiles @ cold_theta.T, cold_test, M_LIST)

    print('\n--- Out-of-Matrix Recall@M (Cold-Start) ---')
    print_results({'LDA (cold)': lda_oom, 'CTR (cold)': ctr_oom})
    print('(PMF cannot participate — no item vectors for unseen items)\n')

    # ── Save results ──────────────────────────────────────────
    all_results = {
        'in_matrix'    : {'PMF': pmf_in, 'LDA': lda_in, 'CTR': ctr_in},
        'out_of_matrix': {'LDA': lda_oom, 'CTR': ctr_oom},
    }
    with open(os.path.join(results_dir, 'recall_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info('Results saved to results/recall_results.json')

    return all_results


if __name__ == '__main__':
    main()

"""
step2_lda.py
------------
Fits LDA on item genome-score content to produce per-item topic proportions θ_j.

In CTR, θ_j serves as the prior mean for item latent vectors:
    v_j = θ_j + ε_j

Outputs (saved to results/):
  - theta.npy       : (n_items x K) topic proportions for all items
  - cold_theta.npy  : (n_cold x K) topic proportions for cold items

Usage:
  python src/step2_lda.py
"""

import logging
import os

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

K            = 50
MAX_ITER     = 20
RANDOM_STATE = 42


def main(results_dir='results'):
    content_matrix = np.load(os.path.join(results_dir, 'content_matrix.npy'))
    cold_content   = np.load(os.path.join(results_dir, 'cold_content.npy'))

    logger.info(f'Fitting LDA: {content_matrix.shape[0]} items, '
                f'{content_matrix.shape[1]} features, K={K}')

    lda = LatentDirichletAllocation(
        n_components=K, max_iter=MAX_ITER,
        learning_method='batch', random_state=RANDOM_STATE, n_jobs=-1)
    lda.fit(content_matrix)

    theta      = lda.transform(content_matrix)   # (n_items x K)
    cold_theta = lda.transform(cold_content)      # (n_cold x K)

    logger.info(f'theta shape      : {theta.shape}')
    logger.info(f'cold_theta shape : {cold_theta.shape}')
    logger.info(f'Row sum mean     : {theta.sum(axis=1).mean():.4f}  (should be ~1.0)')

    np.save(os.path.join(results_dir, 'theta.npy'),      theta)
    np.save(os.path.join(results_dir, 'cold_theta.npy'), cold_theta)
    logger.info('LDA complete. theta and cold_theta saved to results/')


if __name__ == '__main__':
    main()

"""
step3_model_training.py
-----------------------
Trains PMF baseline and CTR main model using coordinate ascent.

CTR model (Wang & Blei, KDD 2011):
    v_j = θ_j + ε_j
    r_ij ~ N(u_i^T v_j, c_ij^-1)

Coordinate ascent updates (Eq. 9-11):
    u_i ← (Σ_j c_ij v_j v_j^T + λ_u I)^{-1}  Σ_j c_ij r_ij v_j
    v_j ← (Σ_i c_ij u_i u_i^T + λ_v I)^{-1} (Σ_i c_ij r_ij u_i + λ_v θ_j)

Outputs (saved to results/):
  - pmf_U.npy, pmf_V.npy  : PMF user and item factor matrices
  - ctr_U.npy, ctr_V.npy  : CTR user and item factor matrices

Usage:
  python src/step3_model_training.py
"""

import logging
import os

import numpy as np
from scipy.sparse import load_npz

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

K            = 50
LAMBDA_U     = 0.01
LAMBDA_V     = 0.01
CONF_A       = 1.0
CONF_B       = 0.01
N_EPOCHS     = 20
RANDOM_STATE = 42


# ── ALS closed-form solver ────────────────────────────────────────────────

def als_solve(F, FtF, obs, a, b, lam, K):
    """
    Closed-form ALS update for one user or item.
    Exploits: F C F^T = b*FtF + (a-b)*F_obs^T F_obs
    """
    A = b * FtF
    if len(obs) > 0:
        A = A + (a - b) * (F[obs].T @ F[obs])
    A += lam * np.eye(K, dtype=np.float32)
    rhs = ((a - b) * F[obs].sum(axis=0) if len(obs) > 0
           else np.zeros(K, dtype=np.float32))
    return np.linalg.solve(A, rhs)


# ── PMF ───────────────────────────────────────────────────────────────────

def train_pmf(train_matrix, K, lu, lv, a, b, n_epochs, seed):
    n_users, n_items = train_matrix.shape
    rng = np.random.default_rng(seed)
    U = (0.01 * rng.standard_normal((n_users, K))).astype(np.float32)
    V = (0.01 * rng.standard_normal((n_items, K))).astype(np.float32)
    csr = train_matrix.tocsr()
    csc = train_matrix.tocsc()

    for ep in range(n_epochs):
        VtV = V.T @ V
        for i in range(n_users):
            U[i] = als_solve(V, VtV, csr.getrow(i).indices, a, b, lu, K)
        UtU = U.T @ U
        for j in range(n_items):
            V[j] = als_solve(U, UtU, csc.getcol(j).indices, a, b, lv, K)
        s = U @ V.T; r, c = csr.nonzero(); ps = s[r, c]
        loss = (a*((ps-1)**2).sum() + b*((s**2).sum()-(ps**2).sum()) +
                lu*(U**2).sum() + lv*(V**2).sum())
        logger.info(f'PMF epoch {ep+1}/{n_epochs}  loss={loss:.4f}')

    return U, V


# ── CTR ───────────────────────────────────────────────────────────────────

def train_ctr(train_matrix, theta, pmf_U, pmf_V, K, lu, lv, a, b, n_epochs):
    """
    CTR coordinate ascent.
    Warm-started from PMF for faster convergence.
    The λ_v θ_j term in the v_j update couples LDA content with collaborative signals.
    """
    n_users, n_items = train_matrix.shape
    U = pmf_U.copy().astype(np.float32)
    V = pmf_V.copy().astype(np.float32)
    theta = theta.astype(np.float32)
    csr = train_matrix.tocsr()
    csc = train_matrix.tocsc()

    for ep in range(n_epochs):
        # ── Update U ──────────────────────────────────────────────
        VtV_b = b * (V.T @ V)
        for i in range(n_users):
            obs_j = csr.getrow(i).indices
            A   = VtV_b.copy() + lu * np.eye(K, dtype=np.float32)
            rhs = np.zeros(K, dtype=np.float32)
            if len(obs_j) > 0:
                V_obs = V[obs_j]
                A   += (a - b) * (V_obs.T @ V_obs)
                rhs += (a - b) * V_obs.sum(axis=0)
            U[i] = np.linalg.solve(A, rhs)

        # ── Update V ──────────────────────────────────────────────
        UtU_b = b * (U.T @ U)
        for j in range(n_items):
            obs_i = csc.getcol(j).indices
            A   = UtU_b.copy() + lv * np.eye(K, dtype=np.float32)
            rhs = lv * theta[j]   # <── LDA coupling: λ_v θ_j
            if len(obs_i) > 0:
                U_obs = U[obs_i]
                A   += (a - b) * (U_obs.T @ U_obs)
                rhs += (a - b) * U_obs.sum(axis=0)
            V[j] = np.linalg.solve(A, rhs)

        eps = V - theta; s = U @ V.T; r, c = csr.nonzero(); ps = s[r, c]
        loss = (b*(s**2).sum() + (a-b)*((ps-1)**2).sum() - b*(ps**2).sum() +
                lu*(U**2).sum() + lv*(eps**2).sum())
        logger.info(f'CTR epoch {ep+1}/{n_epochs}  loss={loss:.4f}')

    return U, V


# ── Main ──────────────────────────────────────────────────────────────────

def main(results_dir='results'):
    train_matrix = load_npz(os.path.join(results_dir, 'train_matrix.npz'))
    theta        = np.load(os.path.join(results_dir, 'theta.npy'))

    logger.info('Training PMF baseline ...')
    pmf_U, pmf_V = train_pmf(train_matrix, K, LAMBDA_U, LAMBDA_V,
                              CONF_A, CONF_B, N_EPOCHS, RANDOM_STATE)
    np.save(os.path.join(results_dir, 'pmf_U.npy'), pmf_U)
    np.save(os.path.join(results_dir, 'pmf_V.npy'), pmf_V)
    logger.info('PMF saved.')

    logger.info('Training CTR (warm-started from PMF) ...')
    ctr_U, ctr_V = train_ctr(train_matrix, theta, pmf_U, pmf_V,
                              K, LAMBDA_U, LAMBDA_V, CONF_A, CONF_B, N_EPOCHS)
    np.save(os.path.join(results_dir, 'ctr_U.npy'), ctr_U)
    np.save(os.path.join(results_dir, 'ctr_V.npy'), ctr_V)
    logger.info('CTR saved. Training complete.')


if __name__ == '__main__':
    main()

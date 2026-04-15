"""
step5_visualisations.py
-----------------------
Generates all plots for the CTR paper replication project.

Plots saved to plots/:
  - plot1_lda_topics.png           : LDA topic distribution
  - plot2_recall_in_matrix.png     : In-matrix Recall@M (PMF vs LDA vs CTR)
  - plot3_recall_out_of_matrix.png : Out-of-matrix Recall@M (cold-start)
  - plot4_combined_recall.png      : Both recall plots side by side
  - plot5_ctr_offset_analysis.png  : ||ε_j|| vs ||θ_j|| scatter
  - plot6_svd_explainability.png   : Part A SVD + Ridge results

Usage:
  python src/step5_visualisations.py
"""

import json
import logging
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.sparse import load_npz

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')

STYLES = {
    'PMF': dict(marker='o', ls='--', color='#4C72B0', label='PMF (CF only)'),
    'LDA': dict(marker='s', ls='--', color='#DD8452', label='LDA (content only)'),
    'CTR': dict(marker='^', ls='-',  color='#55A868', label='CTR (joint)', lw=2.5),
}


def plot_lda_topics(theta, plots_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    im = axes[0].imshow(theta[:50].T, aspect='auto', cmap='YlOrRd')
    axes[0].set_xlabel('Item (first 50)'); axes[0].set_ylabel('Topic')
    axes[0].set_title(r'LDA topic proportions $\theta_j$ (first 50 items)')
    plt.colorbar(im, ax=axes[0])
    axes[1].hist(theta.argmax(axis=1), bins=theta.shape[1],
                 color='steelblue', edgecolor='white')
    axes[1].set_xlabel('Dominant topic'); axes[1].set_ylabel('Number of items')
    axes[1].set_title('Items per dominant topic')
    plt.suptitle('LDA Output — Item Topic Proportions', fontsize=13)
    plt.tight_layout()
    path = os.path.join(plots_dir, 'plot1_lda_topics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    logger.info(f'Saved: {path}')


def plot_recall_in_matrix(pmf_in, lda_in, ctr_in, plots_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, res in [('PMF', pmf_in), ('LDA', lda_in), ('CTR', ctr_in)]:
        M_vals = sorted(res.keys())
        ax.plot(M_vals, [res[m] for m in M_vals], **STYLES[name])
    ax.set_xlabel('M', fontsize=12); ax.set_ylabel('Recall@M', fontsize=12)
    ax.set_title('In-Matrix Recall@M — PMF vs LDA vs CTR', fontsize=13)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    plt.tight_layout()
    path = os.path.join(plots_dir, 'plot2_recall_in_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    logger.info(f'Saved: {path}')


def plot_recall_out_of_matrix(lda_oom, ctr_oom, plots_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    M_vals = sorted(ctr_oom.keys())
    ax.plot(M_vals, [lda_oom[m] for m in M_vals],
            marker='s', ls='--', color='#DD8452', label='LDA (cold)')
    ax.plot(M_vals, [ctr_oom[m] for m in M_vals],
            marker='^', ls='-', color='#55A868', label='CTR (cold)', lw=2.5)
    ax.text(0.97, 0.05, 'PMF: N/A\n(no cold-start)',
            transform=ax.transAxes, ha='right', fontsize=9, color='gray',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_xlabel('M', fontsize=12); ax.set_ylabel('Recall@M', fontsize=12)
    ax.set_title('Out-of-Matrix Recall@M — Cold-Start', fontsize=13)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    plt.tight_layout()
    path = os.path.join(plots_dir, 'plot3_recall_out_of_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    logger.info(f'Saved: {path}')


def plot_combined(pmf_in, lda_in, ctr_in, lda_oom, ctr_oom, plots_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    M_vals = sorted(ctr_in.keys())

    ax = axes[0]
    for name, res in [('PMF', pmf_in), ('LDA', lda_in), ('CTR', ctr_in)]:
        ax.plot(M_vals, [res[m] for m in M_vals], **STYLES[name])
    ax.set_xlabel('M'); ax.set_ylabel('Recall@M')
    ax.set_title('In-Matrix Recall@M'); ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(M_vals, [lda_oom[m] for m in M_vals],
            marker='s', ls='--', color='#DD8452', label='LDA (cold)')
    ax.plot(M_vals, [ctr_oom[m] for m in M_vals],
            marker='^', ls='-', color='#55A868', label='CTR (cold)', lw=2.5)
    ax.set_xlabel('M'); ax.set_ylabel('Recall@M')
    ax.set_title('Out-of-Matrix Recall@M (Cold-Start)'); ax.legend(fontsize=9)

    plt.suptitle('CTR vs Baselines — MovieLens 20M', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(plots_dir, 'plot4_combined_recall.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    logger.info(f'Saved: {path}')


def plot_ctr_offset(ctr_U, ctr_V, theta, plots_dir):
    epsilon    = ctr_V - theta
    eps_norm   = np.linalg.norm(epsilon, axis=1)
    theta_norm = np.linalg.norm(theta,   axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(theta_norm, eps_norm, s=5, alpha=0.3, color='steelblue')
    axes[0].set_xlabel(r'$||\theta_j||$ (LDA prior)')
    axes[0].set_ylabel(r'$||\epsilon_j||$ (learned offset)')
    axes[0].set_title('CTR: Content prior vs collaborative offset')

    axes[1].hist(eps_norm, bins=40, color='coral', edgecolor='white', alpha=0.8)
    axes[1].set_xlabel(r'$||\epsilon_j||$'); axes[1].set_ylabel('Count')
    axes[1].set_title(r'Distribution of offset magnitude $||\epsilon_j||$')

    mean_eps   = eps_norm.mean()
    mean_theta = theta_norm.mean()
    fig.suptitle(f'CTR Offset Analysis  —  Mean ||ε||={mean_eps:.4f}  '
                 f'Mean ||θ||={mean_theta:.4f}', fontsize=12)
    plt.tight_layout()
    path = os.path.join(plots_dir, 'plot5_ctr_offset_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    logger.info(f'Saved: {path}')
    return float(mean_eps), float(mean_theta)


def main(results_dir='results', plots_dir='plots'):
    os.makedirs(plots_dir, exist_ok=True)

    theta      = np.load(os.path.join(results_dir, 'theta.npy'))
    ctr_U      = np.load(os.path.join(results_dir, 'ctr_U.npy'))
    ctr_V      = np.load(os.path.join(results_dir, 'ctr_V.npy'))

    with open(os.path.join(results_dir, 'recall_results.json')) as f:
        recall = json.load(f)

    pmf_in  = {int(k): v for k, v in recall['in_matrix']['PMF'].items()}
    lda_in  = {int(k): v for k, v in recall['in_matrix']['LDA'].items()}
    ctr_in  = {int(k): v for k, v in recall['in_matrix']['CTR'].items()}
    lda_oom = {int(k): v for k, v in recall['out_of_matrix']['LDA'].items()}
    ctr_oom = {int(k): v for k, v in recall['out_of_matrix']['CTR'].items()}

    plot_lda_topics(theta, plots_dir)
    plot_recall_in_matrix(pmf_in, lda_in, ctr_in, plots_dir)
    plot_recall_out_of_matrix(lda_oom, ctr_oom, plots_dir)
    plot_combined(pmf_in, lda_in, ctr_in, lda_oom, ctr_oom, plots_dir)
    mean_eps, mean_theta = plot_ctr_offset(ctr_U, ctr_V, theta, plots_dir)

    logger.info(f'All plots saved to {plots_dir}/')
    logger.info(f'Mean ||epsilon_j|| = {mean_eps:.4f}')
    logger.info(f'Mean ||theta_j||   = {mean_theta:.4f}')


if __name__ == '__main__':
    main()

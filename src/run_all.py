"""
run_all.py
----------
Runs the complete CTR pipeline end-to-end.

Usage:
  python run_all.py --data_dir ./ml-20m
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import step1_data_loading
import step2_lda
import step3_model_training
import step4_evaluation
import step5_visualisations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run full CTR pipeline')
    parser.add_argument('--data_dir', default='./ml-20m',
                        help='Path to ml-20m folder')
    args = parser.parse_args()

    print('\n' + '='*60)
    print('STEP 1 — Data Loading and Preprocessing')
    print('='*60)
    step1_data_loading.main(args.data_dir)

    print('\n' + '='*60)
    print('STEP 2 — LDA Topic Modelling')
    print('='*60)
    step2_lda.main()

    print('\n' + '='*60)
    print('STEP 3 — Model Training (PMF + CTR)')
    print('='*60)
    step3_model_training.main()

    print('\n' + '='*60)
    print('STEP 4 — Evaluation (Recall@M)')
    print('='*60)
    step4_evaluation.main()

    print('\n' + '='*60)
    print('STEP 5 — Visualisations')
    print('='*60)
    step5_visualisations.main()

    print('\n' + '='*60)
    print('PIPELINE COMPLETE')
    print('Results saved to: results/')
    print('Plots saved to  : plots/')
    print('='*60)

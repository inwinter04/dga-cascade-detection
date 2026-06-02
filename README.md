# Lightweight DGA Detection with Cascade Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight, fully CPU-deployable two-stage cascade system for detecting Domain Generation Algorithm (DGA) domains at the network edge. Designed for resource-constrained environments such as home routers, enterprise firewalls, and IoT gateways.

## Overview

The system combines two complementary machine learning classifiers:

- **Stage 1 (Fast Screening):** Logistic Regression on 14 handcrafted string features (49.8μs/domain, ~20K domains/second/core)
- **Stage 2 (Precise Classification):** Random Forest on 8 traffic features (5-minute sliding windows)

The cascade achieves **86.19% domain-level recall** with a **0.51% false positive rate** — all on CPU only.

## Key Results

| Metric | Stage 1 | Stage 2 | Cascade |
|:-------|:-------:|:-------:|:-------:|
| Recall | 94.75% | 89.57% | **86.19%** |
| TNR / Specificity | 61.66% | 95.15% | — |
| FPR | 38.34% | 4.85% | **0.51%** |
| Precision | — | 82.05% | **94.35%** |
| AUC | 0.9246 | 0.9848 | — |

## Repository Structure

```
├── src/
│   ├── features/          # Feature extraction (string + traffic)
│   ├── training/          # Training scripts + cascade pipeline
│   └── models/            # Trained model files (.pkl)
├── data/
│   ├── raw/
│   │   ├── baderj_dga/        # 52 DGA family implementations (third-party)
│   │   └── english_words.txt  # English word dictionary
│   └── processed/             # Precomputed feature statistics
├── experiments/           # Experiment reports (markdown)
├── output/stats/          # Experiment results (JSON)
├── tools/                 # Validation utilities
├── requirements.txt
└── run.sh                 # End-to-end pipeline
```

## Requirements

- Python 3.10+
- scikit-learn, numpy, pandas, scipy (see `requirements.txt`)

## Quick Start

```bash
# 1. Train Stage 1 (String feature LR)
python src/training/train_stage1.py

# 2. Train Stage 2 (Traffic feature RF)
python src/training/train_stage2_v2.py

# 3. Run end-to-end cascade evaluation
python src/training/cascade_pipeline.py

# 4. Run ablation study
python src/training/ablation_study.py

# 5. Compare with baseline (Sivaguru et al.)
python src/training/baseline_sivaguru.py
```

## Data

- **Benign domains:** [Tranco Top 200K](https://tranco-list.eu/)
- **DGA domains:** Generated using [baderj/domain\_generation\_algorithms](https://github.com/baderj/domain_generation_algorithms) (52 families)
- **Real URL validation:** DeepURLBench dataset (see [paper](https://github.com/deepinstinct-algo/DeepURLBench))

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

The DGA family implementations in `data/raw/baderj_dga/` are from [baderj/domain\_generation\_algorithms](https://github.com/baderj/domain_generation_algorithms) and are subject to their original license.

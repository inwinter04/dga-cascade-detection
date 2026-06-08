"""Classifier Comparison for Reviewer #1 — LR vs NB vs Shallow DT

Runs on the same 14 string features, same 60/20/20 split, same threshold tuning.
Reports: latency, AUC, Recall@95%, TNR@95%, Precision, F1.
"""
import sys, json, time, csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, precision_recall_curve,
                             confusion_matrix, precision_score, recall_score, f1_score)

FEATURE_NAMES = [
    'domain_length', 'sld_length', 'digit_ratio', 'vowel_ratio',
    'consonant_ratio', 'unique_char_ratio', 'normalized_entropy',
    'max_consonant_run', 'max_digit_run', 'vc_alternation',
    'subdomain_count', 'has_hyphen', 'bigram_anomaly', 'meaningful_ratio',
]

BENIGN = ROOT / 'data' / 'processed' / 'benign_features.csv'
DGA = ROOT / 'data' / 'processed' / 'dga_features.csv'
OUT = ROOT / 'output' / 'stats' / 'classifier_comparison.json'


def load_data():
    ben = np.loadtxt(BENIGN, delimiter=',', skiprows=1)
    Xb = ben[:, 1:]
    yb = np.zeros(len(Xb))
    dga_feats = []
    with open(DGA, 'r') as f:
        for row in csv.DictReader(f):
            dga_feats.append([float(row[n]) for n in FEATURE_NAMES])
    Xd = np.array(dga_feats)
    yd = np.ones(len(Xd))
    X = np.vstack([Xb, Xd])
    y = np.concatenate([yb, yd])
    return X, y


def find_threshold(y_true, y_prob, target=0.95):
    p, r, t = precision_recall_curve(y_true, y_prob)
    for thr, rec in zip(reversed(t), reversed(r[:-1])):
        if rec >= target:
            return float(thr)
    return float(t[0]) if len(t) > 0 else 0.0


def bench_latency(clf, X_sample, n_runs=10_000):
    """Measure per-sample inference latency in microseconds."""
    # Warmup
    for _ in range(100):
        clf.predict_proba(X_sample[:1])
    # Timed runs
    times = []
    for _ in range(n_runs):
        idx = _ % len(X_sample)
        t0 = time.perf_counter()
        clf.predict_proba(X_sample[idx:idx+1])
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1e6  # microseconds


def main():
    print("=" * 60)
    print("Classifier Comparison: LR vs NB vs Shallow DT")
    print("=" * 60)

    X, y = load_data()
    print(f"\nData: {len(X):,} samples ({FEATURE_NAMES.__len__()} features)")

    # Train/val/test split
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=42)
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    classifiers = {
        "Logistic Regression": LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Decision Tree (max_depth=5)": DecisionTreeClassifier(
            max_depth=5, class_weight='balanced', random_state=42),
    }

    results = {}

    for name, clf in classifiers.items():
        print(f"\n{'='*40}")
        print(f"Training: {name}")
        print(f"{'='*40}")

        # Train
        clf.fit(X_train, y_train)

        # Threshold tuning on validation set
        y_prob_val = clf.predict_proba(X_val)[:, 1]
        best_thr = find_threshold(y_val, y_prob_val, target=0.95)

        # Evaluate on test set
        y_prob_test = clf.predict_proba(X_test)[:, 1]
        y_pred_tuned = (y_prob_test >= best_thr).astype(int)
        y_pred_default = clf.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_tuned).ravel()

        auc = roc_auc_score(y_test, y_prob_test)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = f1_score(y_test, y_pred_tuned)

        # Latency benchmark
        latency_us = bench_latency(clf, X_test)

        results[name] = {
            'auc': round(auc, 4),
            'recall': round(recall, 4),
            'tnr': round(tnr, 4),
            'precision': round(precision, 4),
            'f1': round(f1, 4),
            'threshold': round(best_thr, 4),
            'latency_us': round(latency_us, 2),
            'n_test': len(X_test),
        }

        print(f"  AUC:        {auc:.4f}")
        print(f"  Recall@95%: {recall:.4f}")
        print(f"  TNR@95%:    {tnr:.4f}")
        print(f"  Precision:  {precision:.4f}")
        print(f"  F1:         {f1:.4f}")
        print(f"  Threshold:  {best_thr:.4f}")
        print(f"  Latency:    {latency_us:.1f} us/query")
        print(f"  Test_size:  {len(X_test):,}")

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"{'Classifier':<28} {'AUC':>6} {'Recall':>7} {'TNR':>7} {'F1':>6} {'Lat(us)':>8}")
    print(f"{'-'*70}")
    for name in ["Logistic Regression", "Naive Bayes", "Decision Tree (max_depth=5)"]:
        r = results[name]
        print(f"{name:<28} {r['auc']:>6.4f} {r['recall']:>7.4f} {r['tnr']:>7.4f} "
              f"{r['f1']:>6.4f} {r['latency_us']:>8.1f}")

    # Save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT}")

    # Summary for paper
    lr = results["Logistic Regression"]
    nb = results["Naive Bayes"]
    dt = results["Decision Tree (max_depth=5)"]
    print(f"\n{'='*70}")
    print("SUMMARY FOR PAPER:")
    print(f"  LR:  AUC={lr['auc']}, Recall={lr['recall']}, TNR={lr['tnr']}, "
          f"Latency={lr['latency_us']}us")
    print(f"  NB:  AUC={nb['auc']}, Recall={nb['recall']}, TNR={nb['tnr']}, "
          f"Latency={nb['latency_us']}us")
    print(f"  DT5: AUC={dt['auc']}, Recall={dt['recall']}, TNR={dt['tnr']}, "
          f"Latency={dt['latency_us']}us")
    print(f"\n  → LR vs NB: AUC diff={lr['auc']-nb['auc']:.4f}, "
          f"TNR diff={lr['tnr']-nb['tnr']:.4f}")
    print(f"  → LR vs DT: AUC diff={lr['auc']-dt['auc']:.4f}, "
          f"TNR diff={lr['tnr']-dt['tnr']:.4f}")
    print(f"  → LR latency advantage vs DT: {dt['latency_us']/lr['latency_us']:.1f}x faster")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

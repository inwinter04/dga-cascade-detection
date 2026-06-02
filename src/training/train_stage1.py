"""
Stage 1 分类器训练 — Logistic Regression

实验设计（基于 research_plan_v2）：
  Version A: 全14维特征
  Version B: 仅前6维强分离特征
  评估: 分别报告随机型DGA / 词典型DGA / 总体的Recall
  阈值调优: 找到 Recall≥95% 的阈值
  输出:
    - models/stage1_lr_full.pkl
    - models/stage1_lr_top6.pkl
    - 对比实验表格 → output/stats/stage1_results.json
"""

import sys
import json
import csv
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix,
                             precision_recall_curve, roc_curve)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ── 配置 ──
BENIGN_FEATURES = PROJECT_ROOT / "data" / "processed" / "benign_features.csv"
DGA_FEATURES = PROJECT_ROOT / "data" / "processed" / "dga_features.csv"
DGA_DOMAINS = PROJECT_ROOT / "data" / "processed" / "dga_domains.csv"
OUT_FULL = PROJECT_ROOT / "src" / "models" / "stage1_lr_full.pkl"
OUT_TOP6 = PROJECT_ROOT / "src" / "models" / "stage1_lr_top6.pkl"
OUT_RESULTS = PROJECT_ROOT / "output" / "stats" / "stage1_results.json"
OUT_FIG_DIR = PROJECT_ROOT / "output" / "figures"

FEATURE_NAMES = [
    'domain_length', 'sld_length', 'digit_ratio', 'vowel_ratio',
    'consonant_ratio', 'unique_char_ratio', 'normalized_entropy',
    'max_consonant_run', 'max_digit_run', 'vc_alternation',
    'subdomain_count', 'has_hyphen', 'bigram_anomaly', 'meaningful_ratio',
]

# 前6维强分离特征（sep > 0.8）
TOP6_FEATURES = [
    'bigram_anomaly', 'max_consonant_run', 'domain_length',
    'sld_length', 'vc_alternation', 'meaningful_ratio',
]
TOP6_INDICES = [FEATURE_NAMES.index(f) for f in TOP6_FEATURES]

# 词典型 DGA 家族（需要单独评估）
DICT_DGA_FAMILIES = {'suppobox', 'nymaim', 'proslikefan', 'pushdo',
                     'bumblebee', 'charbot', 'chinad', 'gozi'}


def load_data():
    """加载良性和 DGA 特征"""
    # 良性
    benign_data = np.loadtxt(BENIGN_FEATURES, delimiter=',', skiprows=1)
    X_benign = benign_data[:, 1:]  # skip rank col
    y_benign = np.zeros(len(X_benign))

    # DGA
    dga_features = []
    dga_families = []
    with open(DGA_FEATURES, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            feats = [float(row[name]) for name in FEATURE_NAMES]
            dga_features.append(feats)
            dga_families.append(row['family'])

    X_dga = np.array(dga_features)
    y_dga = np.ones(len(X_dga))

    # 合并
    X = np.vstack([X_benign, X_dga])
    y = np.concatenate([y_benign, y_dga])

    # 家族标签（用于分类型评估）
    families = ['benign'] * len(X_benign) + dga_families

    return X, y, families, X_benign, X_dga, y_benign, y_dga


def evaluate(y_true, y_pred, y_prob, label=""):
    """单次评估"""
    # 如果只有一类样本，跳过混淆矩阵
    if len(set(y_true)) < 2:
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        results = {
            'accuracy': round((y_true == y_pred).mean(), 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'auc': 0.0,
            'false_positive_rate': 0.0,
            'true_negative_rate': 0.0,
            'threshold': 0.5,
            'note': 'single_class',
        }
        return results

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    results = {
        'accuracy': round((tp + tn) / (tp + tn + fp + fn), 4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'auc': round(roc_auc_score(y_true, y_prob), 4),
        'false_positive_rate': round(fp / (fp + tn), 4),
        'true_negative_rate': round(tn / (tn + fp), 4),
        'threshold': 0.5,
    }
    return results


def find_threshold_for_recall(y_true, y_prob, target_recall=0.95):
    """找到能达到目标 Recall 的最高阈值（最大程度过滤良性）"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds: 升序排列（低→高）
    # recalls: 从高到低
    # 从高阈值向低阈值查找
    for t, r in zip(reversed(thresholds), reversed(recalls[:-1])):
        if r >= target_recall:
            return float(t)
    return float(thresholds[0]) if len(thresholds) > 0 else 0.0


def main():
    print("=" * 60)
    print("Stage 1 分类器训练 — Logistic Regression")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    X, y, families, X_benign, X_dga, y_benign, y_dga = load_data()
    n_benign = len(X_benign)
    n_dga = len(X_dga)
    print(f"   良性: {n_benign:,} | DGA: {n_dga:,} | 总计: {len(X):,}")

    # 2. 划分训练/验证/测试集（60/20/20）
    print("[2/5] 划分训练/验证/测试集（60/20/20）...")
    # 先分 80/20（train+val / test）
    X_tv, X_test, y_tv, y_test, fam_tv, fam_test = train_test_split(
        X, y, families, test_size=0.2, stratify=y, random_state=42
    )
    # 再从 80 中分 75/25 得到 60/20（train/val）
    X_train, X_val, y_train, y_val, fam_train, fam_val = train_test_split(
        X_tv, y_tv, fam_tv, test_size=0.25, stratify=y_tv, random_state=42
    )
    print(f"   训练集: {len(X_train):,} | 验证集: {len(X_val):,} | 测试集: {len(X_test):,}")

    # 3. 训练两个版本
    all_results = {}

    for version_name, feature_indices, out_path in [
        ("全14维", list(range(14)), OUT_FULL),
        ("前6强特征", TOP6_INDICES, OUT_TOP6),
    ]:
        print(f"\n[3/5] 训练版本: {version_name}...")

        # 选取特征
        X_train_v = X_train[:, feature_indices]
        X_val_v = X_val[:, feature_indices]
        X_test_v = X_test[:, feature_indices]

        # 训练
        lr = LogisticRegression(class_weight='balanced', max_iter=1000,
                                random_state=42)
        lr.fit(X_train_v, y_train)

        # 在验证集上调阈值（⚠ 不在测试集上调！）
        y_prob_val = lr.predict_proba(X_val_v)[:, 1]
        best_threshold = find_threshold_for_recall(y_val, y_prob_val, 0.95)

        # 在测试集上做最终评估
        y_prob_test = lr.predict_proba(X_test_v)[:, 1]
        y_pred_test = (y_prob_test >= 0.5).astype(int)
        y_pred_tuned = (y_prob_test >= best_threshold).astype(int)

        # 评估（默认阈值 0.5）
        results_default = evaluate(y_test, y_pred_test, y_prob_test, f"{version_name}(def)")

        # 评估（验证集上调优的阈值）
        results_tuned = evaluate(y_test, y_pred_tuned, y_prob_test,
                                 f"{version_name}(recall≥95%)")
        results_tuned['threshold'] = round(best_threshold, 4)

        # 保存模型
        with open(out_path, 'wb') as f:
            pickle.dump({
                'model': lr,
                'feature_names': [FEATURE_NAMES[i] for i in feature_indices],
                'threshold': best_threshold,
                'version': version_name,
            }, f)
        print(f"   模型 → {out_path}")

        # --- 分类型评估 ---
        dga_test_mask = (y_test == 1)
        dga_families_test = [fam_train[i] for i in range(len(fam_train))
                            if y_train[i] == 1]
        # 实际测试集里的DGA家族
        test_families = [f for f, label in zip(fam_test, y_test) if label == 1]

        # 词典型评估
        dict_mask = np.array([f in DICT_DGA_FAMILIES for f in test_families])
        random_mask = np.array([f not in DICT_DGA_FAMILIES for f in test_families])

        if dict_mask.any():
            dict_results = evaluate(
                np.ones(dict_mask.sum()),
                y_pred_tuned[dga_test_mask][dict_mask],
                y_prob_test[dga_test_mask][dict_mask],
                "词典型DGA"
            )
        else:
            dict_results = None

        if random_mask.any():
            random_results = evaluate(
                np.ones(random_mask.sum()),
                y_pred_tuned[dga_test_mask][random_mask],
                y_prob_test[dga_test_mask][random_mask],
                "随机型DGA"
            )
        else:
            random_results = None

        # --- 打印结果 ---
        print(f"\n   📊 {version_name} 评估结果:")
        print(f"  {'指标':<20} {'默认阈值':<15} {'Recall≥95%调优':<15}")
        print(f"  {'-'*50}")
        print(f"  {'Recall':<20} {results_default['recall']:<15.4f} {results_tuned['recall']:<15.4f}")
        print(f"  {'Precision':<20} {results_default['precision']:<15.4f} {results_tuned['precision']:<15.4f}")
        print(f"  {'F1':<20} {results_default['f1']:<15.4f} {results_tuned['f1']:<15.4f}")
        print(f"  {'良性过滤率(TNR)':<20} {results_default['true_negative_rate']:<15.4f} {results_tuned['true_negative_rate']:<15.4f}")
        print(f"  {'AUC':<20} {results_default['auc']:<15.4f} {results_tuned['auc']:<15.4f}")
        print(f"  {'阈值':<20} {'0.5':<15} {results_tuned['threshold']:<15.4f}")

        if dict_results and random_results:
            print(f"\n  📊 分类型评估 (调优后):")
            print(f"  {'类型':<15} {'Recall':<10} {'Precision':<10} {'F1':<10} {'样本数':<10}")
            print(f"  {'-'*50}")
            print(f"  {'随机型DGA':<15} {random_results['recall']:<10.4f} {random_results['precision']:<10.4f} {random_results['f1']:<10.4f} {random_mask.sum():<10}")
            print(f"  {'词典型DGA':<15} {dict_results['recall']:<10.4f} {dict_results['precision']:<10.4f} {dict_results['f1']:<10.4f} {dict_mask.sum():<10}")

        # 保存结果
        all_results[version_name] = {
            'default': results_default,
            'recall95_tuned': results_tuned,
            'dict_dga': dict_results,
            'random_dga': random_results,
            'feature_count': len(feature_indices),
            'threshold_tuned': round(best_threshold, 4),
        }

    # 4. 保存全量结果
    print(f"\n[4/5] 保存结果 → {OUT_RESULTS}")
    with open(OUT_RESULTS, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 5. 结论
    print(f"\n[5/5] 总结")
    print("=" * 60)
    for ver in ["全14维", "前6强特征"]:
        r = all_results[ver]['recall95_tuned']
        print(f"  {ver}: Recall={r['recall']:.4f}, "
              f"良性过滤率={r['true_negative_rate']:.4f}, "
              f"Precision={r['precision']:.4f}, "
              f"F1={r['f1']:.4f}")
    print("=" * 60)

    # 建议
    full_fpr = all_results['全14维']['recall95_tuned']['true_negative_rate']
    top6_fpr = all_results['前6强特征']['recall95_tuned']['true_negative_rate']
    if top6_fpr > full_fpr:
        print(f"\n💡 建议: 前6强特征在 Recall≥95% 时过滤率更高，推荐使用")
    else:
        print(f"\n💡 建议: 全14维在 Recall≥95% 时过滤率更高，推荐使用")


if __name__ == "__main__":
    main()

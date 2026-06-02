"""
消融实验 — Stage 1 特征组合系统性测试

测试 12 种特征组合，在验证集上调阈值 Recall≥95%，在测试集上评估。
"""
import sys, json, pickle
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, confusion_matrix, recall_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BENIGN = ROOT / 'data' / 'processed' / 'benign_features.csv'
DGA = ROOT / 'data' / 'processed' / 'dga_features.csv'
OUT = ROOT / 'output' / 'stats' / 'ablation_results.json'

FEATURE_NAMES = [
    'domain_length', 'sld_length', 'digit_ratio', 'vowel_ratio',
    'consonant_ratio', 'unique_char_ratio', 'normalized_entropy',
    'max_consonant_run', 'max_digit_run', 'vc_alternation',
    'subdomain_count', 'has_hyphen', 'bigram_anomaly', 'meaningful_ratio',
]

# 待测试的组合（按系数排名/分离度/领域知识组合）
ABLATIONS = {
    'all_14': list(range(14)),                                              # 基线
    'top6_sep': [12, 7, 0, 1, 9, 13],                                      # 分离度前6
    'top6_coef': [3, 4, 6, 10, 5, 11],                                     # |系数|前6(含反向)
    'top10_coef': [3, 4, 6, 10, 5, 11, 12, 9, 2, 13],                     # |系数|前10
    'no_vowel_consonant': [i for i in range(14) if i not in (3, 4)],       # 去掉元辅比
    'no_weak3': [i for i in range(14) if i not in (6, 11, 10)],            # 去掉3个弱特征
    'no_entropy': [i for i in range(14) if i != 6],                        # 只去熵
    'only_strong3': [12, 7, 0],                                            # 仅bigram+consonant+length
    'only_strong5': [12, 7, 0, 1, 9],                                     # 分离度前5
    'no_bigram': [i for i in range(14) if i != 12],                        # 去掉bigram
    'no_meaningful': [i for i in range(14) if i != 13],                    # 去掉meaningful_ratio
    'string_only_v1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],                    # 前10维(无高级特征)
}


def load():
    ben = np.loadtxt(BENIGN, delimiter=',', skiprows=1)
    Xb, yb = ben[:, 1:], np.zeros(len(ben))
    
    import csv
    dga_feats, dga_fams = [], []
    with open(DGA, 'r') as f:
        for row in csv.DictReader(f):
            dga_feats.append([float(row[n]) for n in FEATURE_NAMES])
            dga_fams.append(row['family'])
    Xd, yd = np.array(dga_feats), np.ones(len(dga_feats))
    
    X = np.vstack([Xb, Xd])
    y = np.concatenate([yb, yd])
    return X, y


def find_threshold(y_true, y_prob, target=0.95):
    p, r, t = precision_recall_curve(y_true, y_prob)
    for thr, rec in zip(reversed(t), reversed(r[:-1])):
        if rec >= target:
            return float(thr)
    return float(t[0]) if len(t) > 0 else 0.0


def main():
    X, y = load()
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=42)
    
    results = {}
    for name, indices in ABLATIONS.items():
        X_tr = X_train[:, indices]
        X_va = X_val[:, indices]
        X_te = X_test[:, indices]
        
        lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        lr.fit(X_tr, y_train)
        
        # 在验证集上调阈值
        prob_val = lr.predict_proba(X_va)[:, 1]
        threshold = find_threshold(y_val, prob_val)
        
        # 在测试集上评估
        prob_test = lr.predict_proba(X_te)[:, 1]
        pred_tuned = (prob_test >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_test, pred_tuned).ravel()
        recall = tp / (tp + fn)
        tnr = tn / (tn + fp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[name] = {
            'n_features': len(indices),
            'threshold': round(threshold, 4),
            'recall': round(recall, 4),
            'tnr': round(tnr, 4),
            'precision': round(precision, 4),
            'f1': round(f1, 4),
            'features': [FEATURE_NAMES[i] for i in indices],
        }
        
        bar = '█' * int(tnr * 40)
        print(f'{name:20s} n={len(indices):>2d} | Recall={recall:.4f} | TNR={tnr:.4f} | thresh={threshold:.4f} {bar}')
    
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n结果 → {OUT}')
    
    # 总结最佳组合
    print(f'\n=== 推荐排名（按 TNR × Recall 综合评分） ===')
    scored = sorted(results.items(), key=lambda x: x[1]['recall'] * x[1]['tnr'], reverse=True)
    for name, r in scored:
        print(f'{name:20s} Recall={r["recall"]:.4f} TNR={r["tnr"]:.4f} 得分={r["recall"]*r["tnr"]:.4f}')


if __name__ == '__main__':
    main()

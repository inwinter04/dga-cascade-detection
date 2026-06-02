"""
5-fold 交叉验证 + 多阈值测试（Stage 1 LR）
"""
import sys, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BENIGN = ROOT / 'data' / 'processed' / 'benign_features.csv'
DGA = ROOT / 'data' / 'processed' / 'dga_features.csv'
OUT = ROOT / 'output' / 'stats' / 'cv_results.json'

FEATURE_NAMES = [
    'domain_length', 'sld_length', 'digit_ratio', 'vowel_ratio',
    'consonant_ratio', 'unique_char_ratio', 'normalized_entropy',
    'max_consonant_run', 'max_digit_run', 'vc_alternation',
    'subdomain_count', 'has_hyphen', 'bigram_anomaly', 'meaningful_ratio',
]


def load():
    import csv
    ben = np.loadtxt(BENIGN, delimiter=',', skiprows=1)
    Xb, yb = ben[:, 1:], np.zeros(len(ben))
    dga_feats = []
    with open(DGA, 'r') as f:
        for row in csv.DictReader(f):
            dga_feats.append([float(row[n]) for n in FEATURE_NAMES])
    Xd, yd = np.array(dga_feats), np.ones(len(dga_feats))
    X = np.vstack([Xb, Xd])
    y = np.concatenate([yb, yd])
    return X, y


def find_threshold(y_true, y_prob, target):
    p, r, t = precision_recall_curve(y_true, y_prob)
    for thr, rec in zip(reversed(t), reversed(r[:-1])):
        if rec >= target:
            return float(thr)
    return float(t[0]) if len(t) > 0 else 0.0


def main():
    X, y = load()
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    targets = [0.90, 0.95, 0.98]
    all_results = {str(t): [] for t in targets}
    
    fold = 0
    for train_idx, test_idx in kf.split(X, y):
        fold += 1
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        # 从训练集划出验证集
        from sklearn.model_selection import train_test_split
        X_tr2, X_val, y_tr2, y_val = train_test_split(
            X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=42)
        
        lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        lr.fit(X_tr2, y_tr2)
        
        for target in targets:
            prob_val = lr.predict_proba(X_val)[:, 1]
            thr = find_threshold(y_val, prob_val, target)
            
            prob_te = lr.predict_proba(X_te)[:, 1]
            pred = (prob_te >= thr).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
            all_results[str(target)].append({
                'fold': fold,
                'threshold': round(thr, 4),
                'recall': round(tp / (tp + fn), 4),
                'tnr': round(tn / (tn + fp), 4),
                'precision': round(tp / (tp + fp), 4) if (tp+fp)>0 else 0,
            })
        
        if fold % 1 == 0:
            print(f'Fold {fold}/5 done')
    
    print(f'\n{"="*60}')
    print('5-fold CV — Stage 1 Logistic Regression')
    print(f'{"="*60}')
    
    for target in targets:
        results = all_results[str(target)]
        recalls = [r['recall'] for r in results]
        tnrs = [r['tnr'] for r in results]
        thrs = [r['threshold'] for r in results]
        precs = [r['precision'] for r in results]
        
        print(f'\n目标 Recall ≥ {target:.0%}')
        print(f'  Threshold:   {np.mean(thrs):.4f} ± {np.std(thrs):.4f}')
        print(f'  Recall:      {np.mean(recalls):.4f} ± {np.std(recalls):.4f}')
        print(f'  TNR:         {np.mean(tnrs):.4f} ± {np.std(tnrs):.4f}')
        print(f'  Precision:   {np.mean(precs):.4f} ± {np.std(precs):.4f}')
        for r in results:
            print(f'    Fold {r["fold"]}: Recall={r["recall"]:.4f} TNR={r["tnr"]:.4f} th={r["threshold"]:.4f}')
    
    summary = {}
    for target in targets:
        r = all_results[str(target)]
        summary[f'recall_ge_{int(target*100)}'] = {
            'threshold_mean': round(float(np.mean([x['threshold'] for x in r])), 4),
            'threshold_std': round(float(np.std([x['threshold'] for x in r])), 4),
            'recall_mean': round(float(np.mean([x['recall'] for x in r])), 4),
            'recall_std': round(float(np.std([x['recall'] for x in r])), 4),
            'tnr_mean': round(float(np.mean([x['tnr'] for x in r])), 4),
            'tnr_std': round(float(np.std([x['tnr'] for x in r])), 4),
        }
    
    with open(OUT, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n结果 → {OUT}')
    
    # 级联效果估算：Recall=98%阈值
    r98 = summary['recall_ge_98']
    print(f'\n{"="*60}')
    print(f'在 Recall=98% 阈值下的级联估算')
    print(f'{"="*60}')
    print(f'  S1 Threshold: {r98["threshold_mean"]} ± {r98["threshold_std"]}')
    print(f'  S1 Recall:    {r98["recall_mean"]} ± {r98["recall_std"]}')
    print(f'  S1 TNR:       {r98["tnr_mean"]} ± {r98["tnr_std"]}')
    # S2 recall ≈ 0.90 (from v2 training on S1-filtered data)
    cascade_recall = r98['recall_mean'] * 0.90
    cascade_fpr = (1 - r98['tnr_mean']) * (1 - 0.95)  # S2 TNR ≈ 0.95
    print(f'  级联 Recall(est): {cascade_recall:.4f}')
    print(f'  级联 FPR(est):    {cascade_fpr:.4f}')


if __name__ == '__main__':
    main()

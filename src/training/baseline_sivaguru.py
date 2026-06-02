"""
Sivaguru 2020 基线复现 — RF + 12维手工特征

参考文献: Sivaguru et al., "An Evaluation of DGA Classifiers", IEEE BigData 2020

Sivaguru 的 12 维特征:
  1. domain_length     — 域名总长度
  2. sld_length        — SLD 长度
  3. subdomain_count   — 子域名数（dots）
  4. has_hyphen        — 是否含连字符
  5. digit_ratio       — 数字占比
  6. vowel_run_ratio   — 最长元音串 / SLD长度
  7. cons_run_ratio    — 最长辅音串 / SLD长度
  8. char_diversity    — 不同字符数 / SLD长度
  9. normalized_entropy — 归一化熵
  10. bigram_anomaly   — bigram 异常分数
  11. meaningful_ratio — 英文单词覆盖率
  12. gini_index       — 字符分布基尼系数（补充特征）

在相同数据上训练 RF，对比 F1。
"""
import sys, json, csv
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "features"))

from string_features import extract_all_string_features, FEATURE_NAMES as S1_FEATURES

# ── Sivaguru 12维特征定义 ──
# 从提取的 14 维特征中映射 + 补充 gini_index

BIGRAM_FILE = ROOT / 'data' / 'processed' / 'bigram_freq.json'
WORD_FILE = ROOT / 'data' / 'raw' / 'english_words.txt'

OUT_RESULTS = ROOT / 'output' / 'stats' / 'sivaguru_baseline.json'


def gini_index(s):
    """计算字符分布的基尼系数"""
    if not s:
        return 0.0
    counts = Counter(s.lower())
    values = np.array(sorted(counts.values(), reverse=True))
    n = len(values)
    if n == 0 or sum(values) == 0:
        return 0.0
    # 基尼系数 = (2 * sum(i * v_i) / (n * sum(v)) - (n+1)/n)
    cumsum = np.cumsum(values)
    gini = (2 * np.sum((np.arange(1, n+1) * values)) / (n * sum(values))) - (n + 1) / n
    return float(gini)


def extract_sivaguru_features(domain, bigram_freq, english_words):
    """提取 Sivaguru 12 维特征"""
    sld = domain.split('.')[0] if '.' in domain else domain
    sld_len = len(sld)
    domain_len = len(domain)
    
    # 1-5. 基础特征（复用已有函数）
    all_feats = extract_all_string_features(domain, bigram_freq, english_words)
    feat_dict = dict(zip(S1_FEATURES, all_feats))
    
    # 6. vowel_run_ratio — 最长元音串 / SLD长度
    vowels = set('aeiou')
    max_vowel_run = 0
    current = 0
    for c in sld.lower():
        if c in vowels:
            current += 1
            max_vowel_run = max(max_vowel_run, current)
        else:
            current = 0
    vowel_run_ratio = max_vowel_run / sld_len if sld_len > 0 else 0
    
    # 7. cons_run_ratio — 最长辅音串 / SLD长度
    max_cons_run = 0
    current = 0
    for c in sld.lower():
        if c.isalpha() and c not in vowels:
            current += 1
            max_cons_run = max(max_cons_run, current)
        else:
            current = 0
    cons_run_ratio = max_cons_run / sld_len if sld_len > 0 else 0
    
    # 8. char_diversity — 不同字符数 / SLD长度
    char_diversity = len(set(sld.lower())) / sld_len if sld_len > 0 else 0
    
    # 9. normalized_entropy（直接用已有）
    entropy = feat_dict['normalized_entropy']
    
    # 10. bigram_anomaly（直接用已有）
    bigram = feat_dict['bigram_anomaly']
    
    # 11. meaningful_ratio（直接用已有）
    meaningful = feat_dict['meaningful_ratio']
    
    # 12. gini_index
    gini = gini_index(sld)
    
    return [
        feat_dict['domain_length'],     # 1
        feat_dict['sld_length'],        # 2
        feat_dict['subdomain_count'],   # 3
        feat_dict['has_hyphen'],        # 4
        feat_dict['digit_ratio'],       # 5
        vowel_run_ratio,                # 6
        cons_run_ratio,                 # 7
        char_diversity,                 # 8
        entropy,                        # 9
        bigram,                         # 10
        meaningful,                     # 11
        gini,                           # 12
    ]


SIVAGURU_FEATURES = [
    'domain_length', 'sld_length', 'subdomain_count', 'has_hyphen',
    'digit_ratio', 'vowel_run_ratio', 'cons_run_ratio', 'char_diversity',
    'entropy', 'bigram_anomaly', 'meaningful_ratio', 'gini_index',
]


def main():
    import json
    with open(BIGRAM_FILE, 'r') as f:
        bigram_freq = json.load(f)
    with open(WORD_FILE, 'r', encoding='utf-8') as f:
        english_words = set(line.strip().lower() for line in f if line.strip())
    
    # 从 Tranco 读取良性域名
    tranco_path = ROOT / 'data' / 'raw' / 'tranco_top1m.csv'
    benign_domains = []
    with open(tranco_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 200000:  # 和良性特征文件一致
                break
            parts = line.strip().split(',')
            if len(parts) >= 2:
                benign_domains.append(parts[1].strip())
    
    # dga_domains.csv 格式: family,domain
    dga_domains, dga_families = [], []
    with open(ROOT / 'data' / 'processed' / 'dga_domains.csv', 'r', encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                dga_domains.append(parts[1].strip())
                dga_families.append(parts[0].strip())
    
    print(f'良性域名: {len(benign_domains):,}')
    print(f'DGA 域名: {len(dga_domains):,}')
    
    # 提取 Sivaguru 特征
    print('提取 Sivaguru 特征...')
    X_ben = np.array([extract_sivaguru_features(d, bigram_freq, english_words)
                      for d in benign_domains])
    X_dga = np.array([extract_sivaguru_features(d, bigram_freq, english_words)
                      for d in dga_domains])
    
    X = np.vstack([X_ben, X_dga])
    y = np.concatenate([np.zeros(len(X_ben)), np.ones(len(X_dga))])
    
    # 按域名划分 60/20/20
    n_ben = len(X_ben)
    n_dga = len(X_dga)
    indices = np.arange(len(X))
    labels = y.copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=42
    )
    
    print(f'\n训练: {len(X_train):,} | 验证: {len(X_val):,} | 测试: {len(X_test):,}')
    
    # RF (Sivaguru 用的参数)
    rf = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    y_prob = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f'\n{"="*50}')
    print(f'Sivaguru 基线 — RF (默认阈值)')
    print(f'{"="*50}')
    print(f'  Recall:     {recall:.4f}')
    print(f'  Precision:  {precision:.4f}')
    print(f'  F1:         {f1:.4f}')
    print(f'  AUC:        {auc:.4f}')
    print(f'  TNR:        {tn/(tn+fp):.4f}')
    print(f'  FPR:        {fp/(fp+tn):.4f}')
    
    # 特征重要性
    importances = sorted(zip(SIVAGURU_FEATURES, rf.feature_importances_),
                         key=lambda x: -x[1])
    print(f'\n  特征重要性:')
    for name, imp in importances:
        print(f'  {name:20s} {imp:.4f}')
    
    print(f'\n{"="*50}')
    print(f'对比: 本项目 Stage 1 LR')
    print(f'{"="*50}')
    print(f'  Recall:     0.9475')
    print(f'  F1:         0.4738 (Recall优先低阈值)')
    print(f'  AUC:        0.9246')
    print(f'  TNR:        0.6166')
    
    # 保存结果
    results = {
        'method': 'Sivaguru 2020 baseline (RF + 12 features)',
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'recall': round(recall, 4),
        'precision': round(precision, 4),
        'f1': round(f1, 4),
        'auc': round(auc, 4),
        'tnr': round(tn/(tn+fp), 4),
        'feature_importance': {n: round(imp, 4) for n, imp in importances},
    }
    with open(OUT_RESULTS, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n结果 → {OUT_RESULTS}')


if __name__ == '__main__':
    main()

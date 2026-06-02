"""
Stage 2 分类器训练 — 流量特征 + Random Forest
v2: 在 Stage 1 过滤后的数据上训练（级联正确做法）

训练流程：
  1. 加载 DNS 日志
  2. 用 Stage 1 过滤所有记录
  3. 对过滤后的记录提取 8 维流量特征
  4. 按域名拆分（训练/验证/测试）
  5. 训练 RF
"""

import sys
import csv
import json
import pickle
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))

from string_features import extract_all_string_features

# ── 配置 ──
DNS_LOG = PROJECT_ROOT / "data" / "processed" / "dns_log_train_v3.csv"
S1_MODEL_PATH = PROJECT_ROOT / "src" / "models" / "stage1_lr_full.pkl"
S1_RESULTS_PATH = PROJECT_ROOT / "output" / "stats" / "stage1_results.json"
BIGRAM_PATH = PROJECT_ROOT / "data" / "processed" / "bigram_freq.json"
WORD_PATH = PROJECT_ROOT / "data" / "raw" / "english_words.txt"
OUT_MODEL = PROJECT_ROOT / "src" / "models" / "stage2_rf.pkl"
OUT_RESULTS = PROJECT_ROOT / "output" / "stats" / "stage2_results.json"
OUT_CASCADE = PROJECT_ROOT / "output" / "stats" / "cascade_results.json"

WINDOW_MINUTES = 5
SLIDE_MINUTES = 5

FEATURE_NAMES = [
    'query_count', 'unique_src_ips', 'nxdomain_ratio', 'mean_ttl',
    'ttl_std', 'qtype_diversity', 'first_seen_delta', 'burstiness',
]

QTYPE_SET = {'A', 'AAAA', 'MX', 'TXT', 'CNAME'}

DICT_DGA_FAMILIES = {'suppobox', 'nymaim', 'proslikefan', 'pushdo',
                     'bumblebee', 'charbot', 'chinad', 'gozi'}


def load_resources():
    """加载 bigram、词表、Stage 1 模型"""
    with open(BIGRAM_PATH, 'r') as f:
        bigram_freq = json.load(f)
    with open(WORD_PATH, 'r', encoding='utf-8') as f:
        english_words = set(line.strip().lower() for line in f if line.strip())
    with open(S1_MODEL_PATH, 'rb') as f:
        s1 = pickle.load(f)
    with open(S1_RESULTS_PATH, 'r') as f:
        s1_results = json.load(f)
    s1_threshold = s1_results['全14维']['recall95_tuned']['threshold']
    return bigram_freq, english_words, s1, s1_threshold


def load_dns_log(path):
    records = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    records.sort(key=lambda r: r['timestamp'])
    return records


def parse_ts(ts_str):
    return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f').timestamp()


def stage1_filter(records, s1, s1_threshold, bigram_freq, english_words):
    """用 Stage 1 过滤 DNS 记录"""
    filtered = []
    s1_tp = s1_fn = s1_tn = s1_fp = 0
    for rec in records:
        domain = rec['domain']
        label = int(rec['label'])
        feat_list = extract_all_string_features(domain, bigram_freq, english_words)
        prob = s1['model'].predict_proba(np.array([feat_list]))[0, 1]
        if prob >= s1_threshold:
            filtered.append(rec)
            if label == 1: s1_tp += 1
            else: s1_fp += 1
        else:
            if label == 0: s1_tn += 1
            else: s1_fn += 1
    return filtered, s1_tp, s1_fn, s1_tn, s1_fp


def sliding_window(records, window_sec, slide_sec):
    if not records:
        return
    start_ts = parse_ts(records[0]['timestamp'])
    end_ts = parse_ts(records[-1]['timestamp'])
    ws = start_ts
    while ws < end_ts:
        we = ws + window_sec
        wr = [r for r in records if parse_ts(r['timestamp']) >= ws and parse_ts(r['timestamp']) < we]
        if wr:
            yield ws, wr
        ws += slide_sec


def extract_features(records):
    """从 DNS 记录中提取 8 维流量特征"""
    window_sec = WINDOW_MINUTES * 60
    slide_sec = SLIDE_MINUTES * 60
    all_samples = []
    first_seen_times = {}
    
    for r in records:
        d = r['domain']
        if d not in first_seen_times:
            first_seen_times[d] = parse_ts(r['timestamp'])
    
    for ws, window_recs in sliding_window(records, window_sec, slide_sec):
        groups = defaultdict(list)
        for rec in window_recs:
            groups[rec['domain']].append(rec)
        
        for domain, queries in groups.items():
            n = len(queries)
            src_ips = len(set(q['src_ip'] for q in queries))
            nx_count = sum(1 for q in queries if q['rcode'] == 'NXDOMAIN')
            nx_r = nx_count / n
            ttls = [int(q['ttl']) for q in queries]
            mt = np.mean(ttls)
            ts_std = np.std(ttls) if len(ttls) > 1 else 0
            qtypes = len(set(q['qtype'] for q in queries))
            qd = qtypes / len(QTYPE_SET)
            cur_ts = parse_ts(queries[0]['timestamp'])
            fsd = cur_ts - first_seen_times[domain]
            burst = n / max(len(window_recs), 1)
            
            all_samples.append({
                'domain': domain,
                'label': int(queries[0]['label']),
                'family': queries[0].get('family', ''),
                'features': [n, src_ips, nx_r, mt, ts_std, qd, fsd, burst],
            })
    
    return all_samples


def evaluate_binary(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'accuracy': round((tp + tn) / (tp + tn + fp + fn), 4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'true_negative_rate': round(tn / (tn + fp), 4),
        'false_positive_rate': round(fp / (fp + tn), 4),
        'auc': round(roc_auc_score(y_true, y_prob), 4),
    }


def main():
    print("=" * 60)
    print("Stage 2 分类器训练 v2 — 级联数据训练")
    print("=" * 60)
    
    # 1. 加载资源
    print(f"\n[1/6] 加载资源...")
    bigram_freq, english_words, s1, s1_threshold = load_resources()
    print(f"   Stage 1 阈值: {s1_threshold:.4f}")
    
    # 2. 加载 DNS 日志
    print(f"\n[2/6] 加载 DNS 日志...")
    all_records = load_dns_log(DNS_LOG)
    total = len(all_records)
    print(f"   总记录: {total:,}")
    
    # 3. Stage 1 过滤
    print(f"\n[3/6] Stage 1 过滤（14维字符串特征）...")
    filtered, s1_tp, s1_fn, s1_tn, s1_fp = stage1_filter(
        all_records, s1, s1_threshold, bigram_freq, english_words
    )
    n_dga = sum(1 for r in all_records if int(r['label']) == 1)
    n_ben = sum(1 for r in all_records if int(r['label']) == 0)
    print(f"   Stage 1: Recall={s1_tp/n_dga:.4f}, TNR={s1_tn/n_ben:.4f}")
    print(f"   进入 Stage 2: {len(filtered):,} ({len(filtered)/total*100:.1f}%)")
    
    # 4. 提取 8 维流量特征
    print(f"\n[4/6] 提取 8 维流量特征（{WINDOW_MINUTES}min 滑动窗口）...")
    samples = extract_features(filtered)
    labels = Counter(s['label'] for s in samples)
    print(f"   提取样本: {len(samples):,}（良性: {labels[0]:,} | DGA: {labels[1]:,}）")
    
    # 5. 按域名拆分（60/20/20）
    print(f"\n[5/6] 按域名拆分（60/20/20）...")
    domain_samples = defaultdict(list)
    for s in samples:
        domain_samples[s['domain']].append(s)
    
    all_domains = list(domain_samples.keys())
    all_labels = [domain_samples[d][0]['label'] for d in all_domains]
    
    dom_train, dom_test, _, _ = train_test_split(
        all_domains, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )
    dom_train, dom_val, _, _ = train_test_split(
        dom_train,
        [domain_samples[d][0]['label'] for d in dom_train],
        test_size=0.25,
        stratify=[domain_samples[d][0]['label'] for d in dom_train],
        random_state=42
    )
    print(f"   训练域名: {len(dom_train):,} | 验证: {len(dom_val):,} | 测试: {len(dom_test):,}")
    
    def build_matrix(domains):
        Xl, yl = [], []
        for d in domains:
            for s in domain_samples[d]:
                Xl.append(s['features'])
                yl.append(s['label'])
        return np.array(Xl), np.array(yl)
    
    X_train, y_train = build_matrix(dom_train)
    X_val, y_val = build_matrix(dom_val)
    X_test, y_test = build_matrix(dom_test)
    print(f"   训练样本: {len(X_train):,} | 验证: {len(X_val):,} | 测试: {len(X_test):,}")
    
    # 6. 训练 Random Forest
    print(f"\n[6/6] 训练 Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight='balanced', random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    
    y_prob_test = rf.predict_proba(X_test)[:, 1]
    y_pred_test = rf.predict(X_test)
    
    results = evaluate_binary(y_test, y_pred_test, y_prob_test)
    
    print(f"\n   📊 Stage 2 RF 评估结果（在 S1 过滤后的数据上）:")
    print(f"  {'指标':<20} {'值':<12}")
    print(f"  {'-'*35}")
    for k, v in results.items():
        print(f"  {k:<20} {v:<12.4f}")
    
    # 特征重要性
    importances = sorted(zip(FEATURE_NAMES, rf.feature_importances_),
                         key=lambda x: -x[1])
    print(f"\n   📊 特征重要性排序:")
    for name, imp in importances:
        bar = '█' * int(imp * 50)
        print(f"  {name:<20s} {imp:.4f} {bar}")
    
    # 分类型评估
    test_dga_domains = [d for d in dom_test if domain_samples[d][0]['label'] == 1]
    test_dict_domains = [d for d in test_dga_domains
                         if domain_samples[d][0].get('family', '') in DICT_DGA_FAMILIES]
    test_random_domains = [d for d in test_dga_domains
                           if domain_samples[d][0].get('family', '') not in DICT_DGA_FAMILIES]
    
    print(f"\n   📊 分类型 DGA Recall:")
    if test_random_domains:
        Xr = np.array([domain_samples[d][0]['features'] for d in test_random_domains])
        r = recall_score(np.ones(len(test_random_domains)), rf.predict(Xr))
        print(f"  {'随机型 DGA':<20} Recall={r:.4f} (n={len(test_random_domains)})")
    if test_dict_domains:
        Xd = np.array([domain_samples[d][0]['features'] for d in test_dict_domains])
        d = recall_score(np.ones(len(test_dict_domains)), rf.predict(Xd))
        print(f"  {'词典型 DGA':<20} Recall={d:.4f} (n={len(test_dict_domains)})")
    
    # 保存模型
    model_data = {
        'model': rf,
        'feature_names': FEATURE_NAMES,
        'window_minutes': WINDOW_MINUTES,
        'results': results,
        'feature_importance': dict(importances),
        'trained_on_s1_filtered': True,
    }
    with open(OUT_MODEL, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n   模型 → {OUT_MODEL}")
    
    all_results = {
        'overall': results,
        'feature_importance': dict(importances),
        'feature_names': FEATURE_NAMES,
        'total_samples': len(samples),
        's1_filtered_count': len(filtered),
        'trained_on_s1_filtered': True,
    }
    with open(OUT_RESULTS, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"   结果 → {OUT_RESULTS}")


if __name__ == '__main__':
    main()

"""
Stage 2 分类器训练 — 流量特征 + Random Forest

特征提取：5 分钟滑动窗口 × 8 维流量特征
分类器：Random Forest（适合中小规模流量特征）

输出：
  - models/stage2_rf.pkl
  - output/stats/stage2_results.json
  - 级联系统效果汇总 (cascade_results.json)
"""

import sys
import csv
import json
import pickle
import math
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ── 配置 ──
DNS_LOG = PROJECT_ROOT / "data" / "processed" / "dns_log_train_v2.csv"
OUT_MODEL = PROJECT_ROOT / "src" / "models" / "stage2_rf.pkl"
OUT_RESULTS = PROJECT_ROOT / "output" / "stats" / "stage2_results.json"
OUT_CASCADE = PROJECT_ROOT / "output" / "stats" / "cascade_results.json"

WINDOW_MINUTES = 5
SLIDE_MINUTES = 5  # 非重叠窗口

# 8 维流量特征名
FEATURE_NAMES = [
    'query_count',          # 15: 窗口内查询次数
    'unique_src_ips',       # 16: 不同源 IP 数
    'nxdomain_ratio',       # 17: NXDOMAIN 占比
    'mean_ttl',             # 18: 平均 TTL
    'ttl_std',              # 19: TTL 标准差
    'qtype_diversity',      # 20: 查询类型多样性
    'first_seen_delta',     # 21: 距离首次出现时间(秒)
    'burstiness',           # 22: 查询突发度
]

# 查询类型集合（用于多样性计算）
QTYPE_SET = {'A', 'AAAA', 'MX', 'TXT', 'CNAME'}


def load_dns_log(path):
    """加载 DNS 日志，按时间排序"""
    records = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    # 按时间戳排序（确保）
    records.sort(key=lambda r: r['timestamp'])
    return records


def parse_ts(ts_str):
    """解析时间戳为浮点数（秒）"""
    from datetime import datetime
    dt = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f')
    return dt.timestamp()


def sliding_window(records, window_sec, slide_sec):
    """滑动窗口生成器：yield (window_start_ts, window_records)"""
    if not records:
        return
    start_ts = parse_ts(records[0]['timestamp'])
    end_ts = parse_ts(records[-1]['timestamp'])
    
    window_start = start_ts
    while window_start < end_ts:
        window_end = window_start + window_sec
        window_recs = [r for r in records
                       if parse_ts(r['timestamp']) >= window_start
                       and parse_ts(r['timestamp']) < window_end]
        if window_recs:
            yield window_start, window_recs
        window_start += slide_sec


def compute_domain_features(domain_queries, first_seen_times, window_records_count):
    """对单个域名的窗口内所有查询，计算 8 维特征"""
    n = len(domain_queries)
    if n == 0:
        return None
    
    # 1. query_count — 窗口内查询次数
    query_count = n
    
    # 2. unique_src_ips — 不同源 IP 数
    src_ips = set(q['src_ip'] for q in domain_queries)
    unique_src_ips = len(src_ips)
    
    # 3. nxdomain_ratio — NXDOMAIN 占比
    nx_count = sum(1 for q in domain_queries if q['rcode'] == 'NXDOMAIN')
    nxdomain_ratio = nx_count / n
    
    # 4. mean_ttl — 平均 TTL
    ttls = [int(q['ttl']) for q in domain_queries]
    mean_ttl = np.mean(ttls) if ttls else 0
    
    # 5. ttl_std — TTL 标准差
    ttl_std = np.std(ttls) if len(ttls) > 1 else 0
    
    # 6. qtype_diversity — 查询类型多样性
    qtypes = set(q['qtype'] for q in domain_queries)
    qtype_diversity = len(qtypes) / len(QTYPE_SET)
    
    # 7. first_seen_delta — 距离首次出现的时间（秒）
    domain = domain_queries[0]['domain']
    if domain in first_seen_times:
        current_ts = parse_ts(domain_queries[0]['timestamp'])
        first_seen_delta = current_ts - first_seen_times[domain]
    else:
        first_seen_delta = 0
    
    # 8. burstiness — 查询突发度
    # 用窗口内该域名查询数的占比反应突发性
    burstiness = query_count / max(window_records_count, 1)
    
    return {
        'domain': domain_queries[0]['domain'],
        'label': int(domain_queries[0]['label']),
        'family': domain_queries[0].get('family', ''),
        'features': [query_count, unique_src_ips, nxdomain_ratio,
                     mean_ttl, ttl_std, qtype_diversity,
                     first_seen_delta, burstiness],
    }


def extract_features(records):
    """提取所有窗口 × 域名的 8 维特征"""
    window_sec = WINDOW_MINUTES * 60
    slide_sec = SLIDE_MINUTES * 60
    
    all_samples = []
    first_seen_times = {}  # domain -> first seen timestamp
    
    # 先记录所有域名的首次出现时间
    for r in records:
        domain = r['domain']
        if domain not in first_seen_times:
            first_seen_times[domain] = parse_ts(r['timestamp'])
    
    window_count = 0
    for window_start, window_recs in sliding_window(records, window_sec, slide_sec):
        window_count += 1
        # 按域名分组
        domain_groups = defaultdict(list)
        for rec in window_recs:
            domain_groups[rec['domain']].append(rec)
        
        for domain, queries in domain_groups.items():
            sample = compute_domain_features(
                queries, first_seen_times, len(window_recs)
            )
            if sample and sample['label'] != -1:  # 有标签
                all_samples.append(sample)
        
        if window_count % 10 == 0:
            print(f"   已处理 {window_count} 个窗口...", end='\r')
    
    print(f"   共处理 {window_count} 个窗口")
    return all_samples


def evaluate_stage2(y_true, y_pred, y_prob):
    """评估 Stage 2 分类器"""
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
    print("Stage 2 分类器训练 — 流量特征 + Random Forest")
    print("=" * 60)
    
    # 1. 加载 DNS 日志
    print(f"\n[1/5] 加载 DNS 日志...")
    records = load_dns_log(DNS_LOG)
    print(f"   记录数: {len(records):,}")
    
    # 2. 提取滑动窗口特征
    print(f"\n[2/5] 提取 8 维流量特征（{WINDOW_MINUTES}min 滑动窗口）...")
    print(f"   注意: 使用 ALL records（Stage 2 在原始流量上训练，部署时接收 Stage 1 过滤后的流量）")
    samples = extract_features(records)
    print(f"   提取到 {len(samples):,} 个样本（域名×窗口）")
    
    # 统计
    labels = Counter(s['label'] for s in samples)
    print(f"   良性样本: {labels[0]:,} | DGA 样本: {labels[1]:,}")
    
    # 3. 构建特征矩阵 — 按域名拆分（修数据泄漏）
    print(f"\n[3/5] 按域名构建特征矩阵...")
    
    # 按域名分组
    domain_samples = defaultdict(list)
    for s in samples:
        domain_samples[s['domain']].append(s)
    
    all_domains = list(domain_samples.keys())
    all_labels = [domain_samples[d][0]['label'] for d in all_domains]
    
    # 按域名划分 60/20/20
    dom_train, dom_test, _, _ = train_test_split(
        all_domains, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )
    dom_train, dom_val, _, _ = train_test_split(
        dom_train,
        [domain_samples[d][0]['label'] for d in dom_train],
        test_size=0.25, stratify=[domain_samples[d][0]['label'] for d in dom_train],
        random_state=42
    )
    print(f"   训练域名: {len(dom_train):,} | 验证域名: {len(dom_val):,} | 测试域名: {len(dom_test):,}")
    
    # 重建特征矩阵（每个域名的所有窗口样本）
    def build_matrix(domains):
        X_list, y_list = [], []
        for d in domains:
            for s in domain_samples[d]:
                X_list.append(s['features'])
                y_list.append(s['label'])
        return np.array(X_list), np.array(y_list)
    
    X_train, y_train = build_matrix(dom_train)
    X_val, y_val = build_matrix(dom_val)
    X_test, y_test = build_matrix(dom_test)
    
    print(f"   训练样本: {len(X_train):,} | 验证样本: {len(X_val):,} | 测试样本: {len(X_test):,}")
    
    # 4. 训练 RF
    print(f"\n[4/5] 训练 Random Forest...")
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    
    # 在验证集上调阈值（优化 Recall 优先）
    y_prob_val = rf.predict_proba(X_val)[:, 1]
    # 这里我们用默认阈值（RF 的 predict 已经自带 0.5 阈值）
    # 可以调阈值优化 F1 但不做阈值 Recall 优先（Stage 2 是精确判定阶段）
    
    # 在测试集上评估
    y_prob_test = rf.predict_proba(X_test)[:, 1]
    y_pred_test = rf.predict(X_test)
    
    results = evaluate_stage2(y_test, y_pred_test, y_prob_test)
    
    print(f"\n   📊 Stage 2 RF 评估结果:")
    print(f"  {'指标':<20} {'值':<12}")
    print(f"  {'-'*35}")
    for k, v in results.items():
        print(f"  {k:<20} {v:<12.4f}")
    
    # 5. 特征重要性
    print(f"\n   📊 特征重要性排序:")
    importances = sorted(zip(FEATURE_NAMES, rf.feature_importances_),
                         key=lambda x: -x[1])
    for name, imp in importances:
        bar = '█' * int(imp * 50)
        print(f"  {name:<20s} {imp:.4f} {bar}")
    
    # 6. 分类型评估（按域名，修正数据泄漏）
    dict_fams = {'suppobox', 'nymaim', 'proslikefan', 'pushdo',
                 'bumblebee', 'charbot', 'chinad', 'gozi'}
    
    # 从测试域名中收集 DGA 样本
    test_dga_domains = [d for d in dom_test if domain_samples[d][0]['label'] == 1]
    test_dict_domains = [d for d in test_dga_domains
                         if domain_samples[d][0].get('family', '') in dict_fams]
    test_random_domains = [d for d in test_dga_domains
                           if domain_samples[d][0].get('family', '') not in dict_fams]
    
    print(f"\n   📊 分类型 DGA Recall（按域名评估）:")
    if test_random_domains:
        X_random = np.vstack([domain_samples[d][0]['features'] for d in test_random_domains])
        # 对每个域名取第一个窗口（代表性窗口）做评估
        y_pred_random = rf.predict(X_random)
        r = recall_score(np.ones(len(test_random_domains)), y_pred_random, zero_division=0)
        print(f"  {'随机型 DGA':<20} Recall={r:.4f} (n={len(test_random_domains)})")
    if test_dict_domains:
        X_dict = np.vstack([domain_samples[d][0]['features'] for d in test_dict_domains])
        y_pred_dict = rf.predict(X_dict)
        d = recall_score(np.ones(len(test_dict_domains)), y_pred_dict, zero_division=0)
        print(f"  {'词典型 DGA':<20} Recall={d:.4f} (n={len(test_dict_domains)})")
    
    # 7. 保存模型
    print(f"\n[5/5] 保存模型和结果...")
    model_data = {
        'model': rf,
        'feature_names': FEATURE_NAMES,
        'window_minutes': WINDOW_MINUTES,
        'results': results,
        'feature_importance': dict(importances),
    }
    with open(OUT_MODEL, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"   模型 → {OUT_MODEL}")
    
    all_results = {
        'overall': results,
        'feature_importance': dict(importances),
        'feature_names': FEATURE_NAMES,
        'window_minutes': WINDOW_MINUTES,
        'total_samples': len(samples),
        'benign_samples': labels[0],
        'dga_samples': labels[1],
        'dict_dga_recall': round(d, 4) if test_dict_domains else None,
        'random_dga_recall': round(r, 4) if test_random_domains else None,
    }
    with open(OUT_RESULTS, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"   结果 → {OUT_RESULTS}")
    
    # 8. 级联效果估算
    print(f"\n{'='*60}")
    print("级联系统效果估算")
    print(f"{'='*60}")
    # 假设：原始流量100万查询，95%良性，5% DGA
    # Stage 1 过滤掉 54.45% 良性
    # Stage 2 处理剩余流量
    total_queries = 1_000_000
    benign_ratio = 0.95
    dga_ratio = 0.05
    
    s1_tnr = 0.5445  # Stage 1 良性过滤率
    s1_recall = 0.9501  # Stage 1 Recall
    
    s2_recall = results['recall']
    s2_tnr = results['true_negative_rate']
    
    benign_initial = total_queries * benign_ratio
    dga_initial = total_queries * dga_ratio
    
    # Stage 1
    benign_to_s2 = benign_initial * (1 - s1_tnr)  # 未被过滤的良性
    dga_to_s2 = dga_initial * s1_recall  # 被检出的 DGA
    
    # Stage 2
    benign_final_fp = benign_to_s2 * (1 - s2_tnr)  # Stage 2 误报
    dga_final_tp = dga_to_s2 * s2_recall  # Stage 2 正确检出
    
    cascade_recall = s1_recall * s2_recall
    cascade_fpr = (1 - s1_tnr) * (1 - s2_tnr)  # 两级都漏过的比例
    
    print(f"\n  假设场景: 100万 DNS 查询（{int(benign_ratio*100)}% 良性 / {int(dga_ratio*100)}% DGA）")
    print(f"  {'='*50}")
    print(f"  Stage 1 过滤后进入 Stage 2: {benign_to_s2 + dga_to_s2:.0f} 条")
    print(f"    - 良性（待确认）: {benign_to_s2:.0f}")
    print(f"    - DGA（真正例）: {dga_to_s2:.0f}")
    print(f"  Stage 2 最终判定 DGA: {benign_final_fp + dga_final_tp:.0f} 条")
    print(f"    - 误报（良性→DGA）: {benign_final_fp:.0f}")
    print(f"    - 正确检出 DGA: {dga_final_tp:.0f}")
    print(f"  {'='*50}")
    print(f"  级联 Recall: {cascade_recall:.4f}")
    print(f"  级联误报率: {cascade_fpr:.4f}")
    print(f"  总过滤率（良性通过）: {s1_tnr + (1-s1_tnr)*s2_tnr:.4f}")
    
    cascade_results = {
        'total_queries': total_queries,
        'benign_ratio': benign_ratio,
        'dga_ratio': dga_ratio,
        'stage1_tnr': s1_tnr,
        'stage1_recall': s1_recall,
        'stage2_tnr': s2_tnr,
        'stage2_recall': s2_recall,
        'cascade_recall': round(cascade_recall, 4),
        'cascade_fpr': round(cascade_fpr, 4),
        'benign_to_stage2': round(benign_to_s2),
        'dga_to_stage2': round(dga_to_s2),
        'final_fp': round(benign_final_fp),
        'final_tp': round(dga_final_tp),
    }
    with open(OUT_CASCADE, 'w', encoding='utf-8') as f:
        json.dump(cascade_results, f, indent=2, ensure_ascii=False)
    print(f"\n  级联估算 → {OUT_CASCADE}")


import sys
if __name__ == '__main__':
    main()

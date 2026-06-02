"""
DGA 域名特征提取 + 与良性基线的对比分析

从 dga_domains.csv 提取 14 维特征，与良性基线对比
输出:
  1. data/processed/dga_features.csv — DGA 特征矩阵
  2. output/stats/benign_vs_dga_comparison.json — 对比统计
  3. output/figures/ — 对比图
"""

import sys
import json
import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.string_features import (
    extract_all_string_features, compute_bigram_anomaly,
    extract_sld, FEATURE_NAMES
)

# ── 配置 ──
DGA_CSV = PROJECT_ROOT / "data" / "processed" / "dga_domains.csv"
BENIGN_CSV = PROJECT_ROOT / "data" / "processed" / "benign_features.csv"
BIGRAM_JSON = PROJECT_ROOT / "output" / "stats" / "benign_baseline.json"
WORDS_PATH = PROJECT_ROOT / "data" / "raw" / "english_words.txt"
OUT_DGA_FEATURES = PROJECT_ROOT / "data" / "processed" / "dga_features.csv"
OUT_COMPARISON = PROJECT_ROOT / "output" / "stats" / "benign_vs_dga_comparison.json"


def load_english_words(path) -> set:
    words = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().lower()
            if len(word) >= 2:
                words.add(word)
    return words


def build_bigram_freq_from_benign():
    """从良性特征 CSV 读取前 1000 个域名重建 bigram 频率"""
    bigram_counts = Counter()
    with open(BENIGN_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 100000:
                break
            sld = extract_sld(row['domain_length'].split('.')[0])  # not ideal, skip
            # Actually read the raw domains
    # 直接用之前保存的 bigram 频率
    # 先试试从特征矩阵本身恢复
    return None


def compute_stats(feature_matrix):
    """计算特征的 P1~P99 统计"""
    arr = np.array(feature_matrix)
    stats = {}
    for i, name in enumerate(FEATURE_NAMES):
        col = arr[:, i]
        stats[name] = {
            "mean": round(float(np.mean(col)), 6),
            "std": round(float(np.std(col)), 6),
            "min": round(float(np.min(col)), 6),
            "max": round(float(np.max(col)), 6),
            "p5": round(float(np.percentile(col, 5)), 6),
            "p25": round(float(np.percentile(col, 25)), 6),
            "p50": round(float(np.percentile(col, 50)), 6),
            "p75": round(float(np.percentile(col, 75)), 6),
            "p95": round(float(np.percentile(col, 95)), 6),
        }
    return stats


def main():
    print("=" * 60)
    print("DGA vs 良性 特征对比分析")
    print("=" * 60)

    # 1. 加载 DGA 域名
    print("\n[1/5] 加载 DGA 域名...")
    dga_domains = []
    dga_families = []
    with open(DGA_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dga_domains.append(row['domain'])
            dga_families.append(row['family'])
    print(f"   → {len(dga_domains)} 个 DGA 域名, {len(set(dga_families))} 个家族")

    # 2. 加载 bigram 频率和英文词表
    print("[2/5] 加载辅助数据...")
    # 用良性域名重建 bigram 频率（采样）
    bigram_counts = Counter()
    with open(BENIGN_CSV, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        # 从原始 Tranco 重建：使用存的是 rank, f1, f2, ...
        # 实际上我们直接重新从 Tranco 加载更简单

    # 更好的方式：从 bigram_anomaly 的值反推
    # 我们需要 bigram 频率字典给 compute_bigram_anomaly 用
    # 直接从 Tranco 读
    print("   重新从 Tranco 构建 bigram 频率...")
    from src.features.extract_features import build_bigram_freq, load_domains
    raw_path = PROJECT_ROOT / "data" / "raw" / "tranco_top1m.csv"
    benign_domains = load_domains(raw_path, max_count=100000)
    bigram_freq = build_bigram_freq(benign_domains, sample_size=100000)

    english_words = load_english_words(WORDS_PATH)
    print(f"   bigram: {len(bigram_freq)} | 单词: {len(english_words)}")

    # 3. 提取 DGA 特征
    print("[3/5] 提取 14 维特征...")
    dga_features = []
    for i, domain in enumerate(dga_domains):
        feats = extract_all_string_features(domain, benign_bigram_freq=bigram_freq,
                                             english_words=english_words)
        dga_features.append(feats)
        if (i + 1) % 3000 == 0:
            print(f"   → 已完成 {i+1}/{len(dga_domains)}")
    print(f"   → 完成")

    # 4. 保存 DGA 特征矩阵
    print("[4/5] 保存 DGA 特征矩阵...")
    with open(OUT_DGA_FEATURES, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['family'] + FEATURE_NAMES)
        for fam, feats in zip(dga_families, dga_features):
            writer.writerow([fam] + feats)
    print(f"   → {OUT_DGA_FEATURES}")

    # 5. 对比分析
    print("[5/5] 对比分析...")
    dga_stats = compute_stats(dga_features)

    # 加载良性基线
    benign_data = np.loadtxt(BENIGN_CSV, delimiter=',', skiprows=1)
    benign_features = benign_data[:, 1:]  # skip rank

    # 计算良性域名的 stats（用所有数据）
    benign_stats = compute_stats(benign_features)

    # 对比
    print("\n" + "=" * 70)
    print(f"{'特征':<22} {'良性均值':<10} {'DGA均值':<10} {'差值':<10} {'DGA-P50':<10} {'良-P50':<10}")
    print("-" * 70)

    comparison = {}
    for name in FEATURE_NAMES:
        b = benign_stats[name]
        d = dga_stats[name]
        diff = round(d['mean'] - b['mean'], 6)

        comparison[name] = {
            'benign': b,
            'dga': d,
            'diff_mean': diff,
            'direction': 'DGA更高' if diff > 0 else '良性更高',
            'sep_score': round(abs(diff) / ((b['std'] + d['std']) / 2), 4)
        }

        # 用颜色字符简单标记强分离特征
        marker = '⭐' if abs(diff) / ((b['std'] + d['std']) / 2) > 0.8 else ''
        marker = '✨' if abs(diff) / ((b['std'] + d['std']) / 2) > 1.5 else marker
        print(f"{marker} {name:<20} {b['mean']:<10.3f} {d['mean']:<10.3f} {diff:<10.3f} {d['p50']:<10.3f} {b['p50']:<10.3f}")

    # 保存对比
    with open(OUT_COMPARISON, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\n对比结果 → {OUT_COMPARISON}")

    # 按分离度排序
    print("\n" + "=" * 70)
    print("特征分离度排名（Sep Score = |均值差| / 平均标准差）")
    print("=" * 70)
    ranked = sorted(comparison.items(), key=lambda x: x[1]['sep_score'], reverse=True)
    for i, (name, data) in enumerate(ranked, 1):
        star = '⭐' if data['sep_score'] > 1.5 else ('✨' if data['sep_score'] > 0.8 else '  ')
        print(f"  {i:2d}. {star} {name:<22s} sep={data['sep_score']:.3f} ({data['direction']})")


if __name__ == "__main__":
    main()

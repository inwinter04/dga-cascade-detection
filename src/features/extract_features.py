"""
良性域名特征提取 — Stage 1 Baseline

从 Tranco Top 1M 提取 14 维字符串特征，输出：
  1. 特征矩阵 CSV (data/processed/benign_features.csv)
  2. 基线统计 JSON (output/stats/benign_baseline.json)
  3. 分布图 (output/figures/)
"""

import sys
import os
import json
import math
import csv
from collections import Counter
from pathlib import Path

import numpy as np

# ── 确保能找到项目根 ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.string_features import (
    extract_all_string_features, FEATURE_NAMES,
    compute_bigram_anomaly
)


def build_bigram_freq(domains, sample_size=500000):
    """从良性域名中构建 bigram 频率字典（用于特征13）"""
    bigram_counts = Counter()
    total = 0
    for domain in domains:
        if sample_size and total >= sample_size:
            break
        sld = domain.rsplit('.')[-2] if len(domain.split('.')) >= 2 else domain.split('.')[0]
        sld = sld.lower()
        for i in range(len(sld) - 1):
            bigram_counts[sld[i:i+2]] += 1
        total += 1

    total_bigrams = sum(bigram_counts.values())
    freq = {bg: c / total_bigrams for bg, c in bigram_counts.items()}
    return freq


def load_domains(path, max_count=None):
    """从 Tranco CSV 加载域名列表"""
    domains = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if max_count and i >= max_count:
                break
            if len(row) >= 2:
                domains.append(row[1].strip())
    return domains


def compute_baseline_stats(feature_matrix, names):
    """计算每个特征的基线统计量"""
    stats = {}
    arr = np.array(feature_matrix)
    for i, name in enumerate(names):
        col = arr[:, i]
        stats[name] = {
            "mean": round(float(np.mean(col)), 6),
            "std": round(float(np.std(col)), 6),
            "min": round(float(np.min(col)), 6),
            "max": round(float(np.max(col)), 6),
            "p1": round(float(np.percentile(col, 1)), 6),
            "p5": round(float(np.percentile(col, 5)), 6),
            "p25": round(float(np.percentile(col, 25)), 6),
            "p50": round(float(np.percentile(col, 50)), 6),
            "p75": round(float(np.percentile(col, 75)), 6),
            "p95": round(float(np.percentile(col, 95)), 6),
            "p99": round(float(np.percentile(col, 99)), 6),
        }
    return stats


def load_english_words(path) -> set:
    """加载英文词表"""
    words = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().lower()
            if len(word) >= 2:  # 忽略单字母
                words.add(word)
    return words


def main():
    # ── 配置 ──
    raw_path = PROJECT_ROOT / "data" / "raw" / "tranco_top1m.csv"
    words_path = PROJECT_ROOT / "data" / "raw" / "english_words.txt"
    out_csv = PROJECT_ROOT / "data" / "processed" / "benign_features.csv"
    out_stats = PROJECT_ROOT / "output" / "stats" / "benign_baseline.json"
    out_fig_dir = PROJECT_ROOT / "output" / "figures"
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    # 采样大小（Tranco 太大，用前 N 个已经能反映分布）
    SAMPLE_SIZE = 200_000   # 提取 20 万个域名的特征
    BIGRAM_SAMPLE = 100_000  # bigram 频率用的样本

    print(f"[1/5] 加载域名（前 {SAMPLE_SIZE} 个）...")
    domains = load_domains(raw_path, max_count=SAMPLE_SIZE)
    print(f"   → {len(domains)} 个域名")

    print(f"[2/5] 构建 bigram 频率字典（样本 {BIGRAM_SAMPLE}）...")
    bigram_freq = build_bigram_freq(domains, sample_size=BIGRAM_SAMPLE)
    print(f"   → {len(bigram_freq)} 个唯一 bigram")

    print(f"[2b/5] 加载英文词表...")
    english_words = load_english_words(words_path)
    print(f"   → {len(english_words)} 个单词")

    print(f"[3/5] 提取 14 维特征...")
    feature_matrix = []
    for i, domain in enumerate(domains):
        feats = extract_all_string_features(domain, benign_bigram_freq=bigram_freq,
                                            english_words=english_words)
        feature_matrix.append(feats)
        if (i + 1) % 50_000 == 0:
            print(f"   → 已完成 {i+1}/{len(domains)}")

    print(f"[4/5] 保存特征矩阵 CSV...")
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['rank'] + FEATURE_NAMES)
        for rank, feats in enumerate(feature_matrix, 1):
            writer.writerow([rank] + feats)
    print(f"   → {out_csv} ({os.path.getsize(out_csv)/1024/1024:.1f} MB)")

    print(f"[5/5] 计算基线统计...")
    stats = compute_baseline_stats(feature_matrix, FEATURE_NAMES)
    with open(out_stats, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"   → {out_stats}")

    # ── 打印摘要 ──
    print("\n" + "=" * 70)
    print(f"✅ 良性域名特征基线完成（N={len(domains)}）")
    print("=" * 70)
    print(f"{'特征':<22} {'均值':<10} {'标准差':<10} {'P5':<10} {'P50':<10} {'P95':<10}")
    print("-" * 70)
    for name in FEATURE_NAMES:
        s = stats[name]
        print(f"{name:<22} {s['mean']:<10.4f} {s['std']:<10.4f} {s['p5']:<10.4f} {s['p50']:<10.4f} {s['p95']:<10.4f}")

    print("\n💡 提示：这些基线将用于：")
    print("   1. 与 DGA 域名的特征分布对比（看分离度）")
    print("   2. 为 bigram 异常度特征提供参考分布")
    print("   3. 设定 Logistic Regression 的初始权重参考")


if __name__ == "__main__":
    main()

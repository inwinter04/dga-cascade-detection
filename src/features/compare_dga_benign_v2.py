"""
DGA vs 良性 特征对比分析（v2 — 修复版）

修复 v1 的问题:
  - 统一从 bigram_freq.json 读取 bigram 频率
  - 已验证 DGA 域名数量 42K+
  - 加入特征提取速度基准测试
  - 规范化输出报告格式
"""

import sys
import json
import csv
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.string_features import (
    extract_all_string_features, FEATURE_NAMES
)

# ── 配置 ──
DGA_CSV = PROJECT_ROOT / "data" / "processed" / "dga_domains.csv"
BENIGN_FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "benign_features.csv"
BIGRAM_JSON = PROJECT_ROOT / "data" / "processed" / "bigram_freq.json"
WORDS_PATH = PROJECT_ROOT / "data" / "raw" / "english_words.txt"
OUT_DGA_FEATURES = PROJECT_ROOT / "data" / "processed" / "dga_features.csv"
OUT_COMPARISON = PROJECT_ROOT / "output" / "stats" / "benign_vs_dga_comparison.json"


def load_json_freq(path):
    """加载统一 bigram 频率字典"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    meta = data.pop("_meta", {})
    # 将键从字符串转回字符串（本来就是字符串），值是概率
    freq = {k: v for k, v in data.items()}
    return freq, meta


def benchmark_extract_speed(domains, bigram_freq, english_words, n=10000):
    """实测特征提取速度"""
    print(f"   基准测试: {n} 个域名...")
    start = time.perf_counter()
    for domain in domains[:n]:
        extract_all_string_features(domain,
                                     benign_bigram_freq=bigram_freq,
                                     english_words=english_words)
    elapsed = time.perf_counter() - start
    per_domain = elapsed / n
    return per_domain


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
    print("DGA vs 良性 特征对比分析 v2")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    print(f"   良性特征: {BENIGN_FEATURES_CSV}")
    print(f"   DGA 域名: {DGA_CSV}")
    print(f"   bigram 频率: {BIGRAM_JSON}")
    print(f"   英文词表: {WORDS_PATH}")

    dga_domains = []
    dga_families = []
    with open(DGA_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dga_domains.append(row['domain'])
            dga_families.append(row['family'])
    print(f"   → {len(dga_domains)} 个 DGA 域名, {len(set(dga_families))} 个家族")

    # 2. 加载辅助数据
    print("[2/6] 加载 bigram 频率和英文词表...")
    bigram_freq, meta = load_json_freq(BIGRAM_JSON)
    print(f"   bigram: {meta['unique_bigrams']} 个唯一 (来自 {meta['total_domains']} 个良性域名)")

    english_words = set()
    with open(WORDS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().lower()
            if len(word) >= 2:
                english_words.add(word)
    print(f"   英文单词: {len(english_words)}")

    # 3. 基准测试
    print("[3/6] 特征提取速度基准测试...")
    speed = benchmark_extract_speed(dga_domains, bigram_freq, english_words, n=5000)
    print(f"   → 每域名 {(speed*1e6):.1f}μs")
    print(f"   → 每秒可处理 {1/speed:.0f} 个域名")

    # 4. 提取 DGA 特征
    print("[4/6] 提取全部 DGA 域名特征...")
    dga_features = []
    for i, domain in enumerate(dga_domains):
        feats = extract_all_string_features(domain, benign_bigram_freq=bigram_freq,
                                             english_words=english_words)
        dga_features.append(feats)
        if (i + 1) % 10000 == 0:
            print(f"   → 已完成 {i+1}/{len(dga_domains)}")
    print(f"   → 完成")

    # 5. 保存 DGA 特征矩阵
    print("[5/6] 保存 DGA 特征矩阵...")
    with open(OUT_DGA_FEATURES, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['family'] + FEATURE_NAMES)
        for fam, feats in zip(dga_families, dga_features):
            writer.writerow([fam] + feats)
    print(f"   → {OUT_DGA_FEATURES}")

    # 6. 对比分析
    print("[6/6] 对比分析...")
    dga_stats = compute_stats(dga_features)

    benign_data = np.loadtxt(BENIGN_FEATURES_CSV, delimiter=',', skiprows=1)
    benign_features = benign_data[:, 1:]  # skip rank column
    benign_stats = compute_stats(benign_features)

    print("\n" + "=" * 80)
    print(f"{'特征':<22} {'良性均值':<10} {'DGA均值':<10} {'差值':<10} {'分离度':<10} {'DGA-P50':<10} {'良-P50':<10}")
    print("-" * 80)

    comparison = {}
    for name in FEATURE_NAMES:
        b = benign_stats[name]
        d = dga_stats[name]
        diff = round(d['mean'] - b['mean'], 6)
        sep = round(abs(diff) / ((b['std'] + d['std']) / 2), 4)

        comparison[name] = {
            'benign': b,
            'dga': d,
            'diff_mean': diff,
            'direction': 'DGA更高' if diff > 0 else '良性更高',
            'sep_score': sep,
        }

        marker = ''
        if sep > 1.5:
            marker = '⭐⭐'
        elif sep > 0.8:
            marker = '⭐'
        elif sep < 0.3:
            marker = '  '

        print(f"{marker} {name:<20} {b['mean']:<10.3f} {d['mean']:<10.3f} {diff:<10.3f} {sep:<10.3f} {d['p50']:<10.3f} {b['p50']:<10.3f}")

    # 排序
    ranked = sorted(comparison.items(), key=lambda x: x[1]['sep_score'], reverse=True)

    print("\n" + "=" * 60)
    print("特征分离度排名")
    print("=" * 60)
    for i, (name, data) in enumerate(ranked, 1):
        print(f"  {i:2d}. {name:<22s} sep={data['sep_score']:.3f} ({data['direction']})")

    # 保存
    with open(OUT_COMPARISON, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\n→ {OUT_COMPARISON}")
    print(f"→ 提取速度: {(speed*1e6):.1f}μs/域名")


if __name__ == "__main__":
    main()

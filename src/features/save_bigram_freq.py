"""
保存统一的 bigram 频率字典，供所有特征提取脚本共用
"""
import json
import csv
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.string_features import extract_sld

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "tranco_top1m.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "bigram_freq.json"
SAMPLE_SIZE = 100_000

bigram_counts = Counter()
total = 0
with open(RAW_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i >= SAMPLE_SIZE:
            break
        domain = row[1].strip()
        sld = extract_sld(domain).lower()
        for j in range(len(sld) - 1):
            bigram_counts[sld[j:j+2]] += 1
        total += 1

total_bigrams = sum(bigram_counts.values())
freq = {bg: c / total_bigrams for bg, c in bigram_counts.items()}
freq["_meta"] = {
    "total_domains": total,
    "total_bigrams": total_bigrams,
    "unique_bigrams": len(bigram_counts)
}

with open(OUT_PATH, 'w') as f:
    json.dump(freq, f)

print(f"已保存: {OUT_PATH}")
print(f"域名: {total}, 唯一 bigram: {len(bigram_counts)}")

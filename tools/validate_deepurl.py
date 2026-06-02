"""DeepURLBench Stage 1 validation — single partition"""
import sys, json, urllib.request
from pathlib import Path
from collections import Counter
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "features"))

import pyarrow.parquet as pq
from string_features import extract_all_string_features, FEATURE_NAMES
import pickle
import random

BIGRAM_PATH = ROOT / 'data' / 'processed' / 'bigram_freq.json'
WORD_PATH = ROOT / 'data' / 'raw' / 'english_words.txt'
S1_MODEL_PATH = ROOT / 'src' / 'models' / 'stage1_lr_full.pkl'
S1_RESULTS_PATH = ROOT / 'output' / 'stats' / 'stage1_results.json'
OUT_RESULTS = ROOT / 'output' / 'stats' / 'deepurlbench_validation.json'
DATA_DIR = ROOT / 'data' / 'raw' / 'deepurlbench'

# Load resources
with open(BIGRAM_PATH, 'r') as f:
    bigram_freq = json.load(f)
with open(WORD_PATH, 'r', encoding='utf-8') as f:
    english_words = set(line.strip().lower() for line in f if line.strip())
with open(S1_MODEL_PATH, 'rb') as f:
    s1 = pickle.load(f)
with open(S1_RESULTS_PATH, 'r') as f:
    s1_threshold = json.load(f)['全14维']['recall95_tuned']['threshold']
print('Stage 1 threshold:', s1_threshold)


def extract_domain(url):
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split('/')[0]
        if ':' in domain:
            domain = domain.split(':')[0]
        return domain.lower().strip()
    except:
        return None


# Download if needed
local_path = DATA_DIR / 'part-00000.parquet'
if not local_path.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/deepinstinct-algo/DeepURLBench/main/urls_with_dns/part-00000.parquet"
    print('Downloading part-00000...')
    urllib.request.urlretrieve(url, local_path)
    print('Done:', local_path.stat().st_size / 1024 / 1024, 'MB')

# Read data
table = pq.read_table(str(local_path))
print('Rows:', table.num_rows)

# Extract domains and labels
domains, labels, ttls = [], [], []
for i in range(table.num_rows):
    row = {col: table.column(col)[i].as_py() for col in table.column_names}
    domain = extract_domain(str(row.get('url', '')))
    label = row.get('label', '')
    if domain and label in ('benign', 'mal', 'phishing'):
        domains.append(domain)
        labels.append(label)
        ttls.append(row.get('TTL', 0))

print(f'Valid domains: {len(domains):,}')
label_counts = Counter(labels)
for lbl, cnt in label_counts.most_common():
    print(f'  {lbl}: {cnt:,} ({cnt/len(domains)*100:.1f}%)')

# Stage 1 validation (sampled for speed)
random.seed(42)
SAMPLE = 20000
if len(domains) > SAMPLE:
    idx = random.sample(range(len(domains)), SAMPLE)
    domains_s = [domains[i] for i in idx]
    labels_s = [labels[i] for i in idx]
else:
    domains_s, labels_s = domains, labels

print(f'\nEvaluating on {len(domains_s):,} samples...')

results = {'mal': {'tp': 0, 'total': 0},
           'phishing': {'tp': 0, 'total': 0},
           'benign': {'fp': 0, 'total': 0}}

for i, (domain, label) in enumerate(zip(domains_s, labels_s)):
    feats = extract_all_string_features(domain, bigram_freq, english_words)
    prob = s1['model'].predict_proba([feats])[0, 1]
    pred = int(prob >= s1_threshold)
    
    if label == 'mal':
        results['mal']['total'] += 1
        if pred == 1: results['mal']['tp'] += 1
    elif label == 'phishing':
        results['phishing']['total'] += 1
        if pred == 1: results['phishing']['tp'] += 1
    else:
        results['benign']['total'] += 1
        if pred == 1: results['benign']['fp'] += 1
    
    if (i + 1) % 5000 == 0:
        print(f'  {i+1}/{len(domains_s)}...', end='\r')

print()
print('=' * 50)
print('Stage 1 on DeepURLBench (real URLs)')
print('=' * 50)

for label in ['mal', 'phishing', 'benign']:
    r = results[label]
    if label in ('mal', 'phishing'):
        rec = r['tp'] / r['total'] if r['total'] > 0 else 0
        print(f'  {label:10s} Recall={rec:.4f} ({r["tp"]}/{r["total"]})')
    else:
        tnr = 1 - r['fp'] / r['total'] if r['total'] > 0 else 0
        print(f'  {label:10s} TNR={tnr:.4f} (FP={r["fp"]}/{r["total"]})')

total_mal = results['mal']['total'] + results['phishing']['total']
total_tp = results['mal']['tp'] + results['phishing']['tp']
combined_recall = total_tp / total_mal if total_mal > 0 else 0
benign_tnr = 1 - results['benign']['fp'] / results['benign']['total']

print(f'\n  mal+phishing Recall={combined_recall:.4f}')
print(f'  benign TNR={benign_tnr:.4f}')

summary = {
    'source': 'DeepURLBench (part-00000)',
    'sample_size': len(domains_s),
    'mal_recall': round(results['mal']['tp'] / results['mal']['total'], 4),
    'phishing_recall': round(results['phishing']['tp'] / results['phishing']['total'], 4),
    'combined_recall': round(combined_recall, 4),
    'benign_tnr': round(benign_tnr, 4),
}
with open(OUT_RESULTS, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nResults saved -> {OUT_RESULTS}')

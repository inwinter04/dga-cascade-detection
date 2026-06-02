"""DeepURLBench: strip www prefix, re-evaluate Stage 1"""
import sys, json
from pathlib import Path
from urllib.parse import urlparse
from collections import Counter
import pyarrow.parquet as pq
import pickle
import random

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "features"))
from string_features import extract_all_string_features

BIGRAM_PATH = ROOT / 'data' / 'processed' / 'bigram_freq.json'
WORD_PATH = ROOT / 'data' / 'raw' / 'english_words.txt'
S1_MODEL_PATH = ROOT / 'src' / 'models' / 'stage1_lr_full.pkl'
S1_RESULTS_PATH = ROOT / 'output' / 'stats' / 'stage1_results.json'
DATA_DIR = ROOT / 'data' / 'raw' / 'deepurlbench'

with open(BIGRAM_PATH, 'r') as f: bigram_freq = json.load(f)
with open(WORD_PATH, 'r', encoding='utf-8') as f:
    english_words = set(line.strip().lower() for line in f if line.strip())
with open(S1_MODEL_PATH, 'rb') as f: s1 = pickle.load(f)
with open(S1_RESULTS_PATH, 'r') as f:
    thresh = json.load(f)['全14维']['recall95_tuned']['threshold']

def get_bare_domain(url):
    """提取裸域名（去掉 www 等子域名前缀）"""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split('/')[0]
        if ':' in domain: domain = domain.split(':')[0]
        domain = domain.lower().strip()
        # 去掉常见的子域名前缀
        parts = domain.split('.')
        common_prefixes = {'www', 'mail', 'm', 'ftp', 'smtp', 'api', 'blog', 'admin', 'test',
                          'web', 'mail2', 'ww2', 'ww1', 'ww3', 'www2', 'www1', 'www3'}
        if len(parts) >= 3 and parts[0] in common_prefixes:
            return '.'.join(parts[1:])  # 去掉第一级
        return domain
    except:
        return None

table = pq.read_table(str(DATA_DIR / 'part-00000.parquet'))

# 收集并去重
all_records = []
for i in range(table.num_rows):
    row = {col: table.column(col)[i].as_py() for col in table.column_names}
    bare = get_bare_domain(str(row.get('url', '')))
    label = row.get('label', '')
    if bare and label in ('benign', 'mal', 'phishing'):
        all_records.append({'domain': bare, 'label': label, 
                           'original_url': str(row.get('url', ''))[:100]})

print(f'总计: {len(all_records):,} 条')

# 统计原始 vs 裸域名的变化
has_www = sum(1 for r in all_records if 'www.' in r['original_url'][:20])
changed = sum(1 for r in all_records if 'www.' in r['original_url'][:20])
print(f'含 www 前缀: {has_www:,} ({has_www/len(all_records)*100:.1f}%)')

# 采样评估
random.seed(42)
SAMPLE = 20000
sample = random.sample(all_records, min(SAMPLE, len(all_records)))

results = {'mal': {'tp': 0, 'total': 0}, 'phishing': {'tp': 0, 'total': 0}, 'benign': {'fp': 0, 'total': 0}}

for i, rec in enumerate(sample):
    feats = extract_all_string_features(rec['domain'], bigram_freq, english_words)
    prob = s1['model'].predict_proba([feats])[0, 1]
    pred = int(prob >= thresh)
    
    label = rec['label']
    if label == 'mal':
        results['mal']['total'] += 1
        if pred: results['mal']['tp'] += 1
    elif label == 'phishing':
        results['phishing']['total'] += 1
        if pred: results['phishing']['tp'] += 1
    else:
        results['benign']['total'] += 1
        if pred: results['benign']['fp'] += 1
    
    if (i+1) % 5000 == 0: print(f'  {i+1}/{len(sample)}')

print('\n' + '='*50)
print('Stage 1 on DeepURLBench — 裸域名版本')
print('='*50)

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
print(f'\n  mal+phishing Recall={total_tp/total_mal:.4f} ({total_tp}/{total_mal})')
benign_tnr = 1 - results['benign']['fp']/results['benign']['total']
print(f'  benign TNR={benign_tnr:.4f}')

# 和原始版本对比
print(f'\n{"="*50}')
print(f'对比: 原始版本 vs 裸域名版本')
print(f'{"="*50}')
print(f'  {"指标":<20} {"原始(带www)":<15} {"裸域名":<15}')
print(f'  {"mal Recall":<20} {"0.8252":<15} {results["mal"]["tp"]/results["mal"]["total"]:.4f}')
mal_tnr_orig = 1 - 15862/19305
mal_tnr_bare = 1 - results['benign']['fp']/results['benign']['total']
print(f'  {"benign TNR":<20} {"0.1783":<15} {mal_tnr_bare:.4f}')

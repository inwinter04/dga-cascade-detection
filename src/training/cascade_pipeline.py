"""
Cascade Pipeline — Stage 1 (LR string features) → Stage 2 (RF traffic features)

End-to-end evaluation of the two-stage DGA detection system:
  1. Per-record S1 filtering via 14-dim string features + LR
  2. Suspicious records grouped by 5-min sliding windows
  3. Per-domain-per-window 8-dim traffic features extracted
  4. S2 RF classifies each window sample
  5. Domain-level: detected if any window flagged as DGA
  6. Cascade: DGA domains missed by S1 OR S2 = cascade FN

Output: output/stats/cascade_end_to_end.json
"""

import sys
import csv
import json
import pickle
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.metrics import precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))

from string_features import extract_all_string_features

# ── Config ──
DNS_LOG = PROJECT_ROOT / "data" / "processed" / "dns_log_train_v3.csv"
S1_MODEL_PATH = PROJECT_ROOT / "src" / "models" / "stage1_lr_full.pkl"
S2_MODEL_PATH = PROJECT_ROOT / "src" / "models" / "stage2_rf.pkl"
BIGRAM_PATH = PROJECT_ROOT / "data" / "processed" / "bigram_freq.json"
WORD_PATH = PROJECT_ROOT / "data" / "raw" / "english_words.txt"
S1_RESULTS_PATH = PROJECT_ROOT / "output" / "stats" / "stage1_results.json"
OUT_PATH = PROJECT_ROOT / "output" / "stats" / "cascade_end_to_end.json"

WINDOW_MINUTES = 5
SLIDE_MINUTES = 5

QTYPE_SET = {'A', 'AAAA', 'MX', 'TXT', 'CNAME'}


# ── Resource loading ──

def load_resources():
    """Load S1/S2 models and string-feature support data."""
    with open(BIGRAM_PATH, 'r') as f:
        bigram_freq = json.load(f)
    with open(WORD_PATH, 'r', encoding='utf-8') as f:
        english_words = set(line.strip().lower() for line in f if line.strip())
    with open(S1_MODEL_PATH, 'rb') as f:
        s1_bundle = pickle.load(f)
    s1_model = s1_bundle['model']
    with open(S1_RESULTS_PATH, 'r') as f:
        s1_results = json.load(f)
    s1_threshold = s1_results['全14维']['recall95_tuned']['threshold']
    with open(S2_MODEL_PATH, 'rb') as f:
        s2_bundle = pickle.load(f)
    s2_model = s2_bundle['model']
    return bigram_freq, english_words, s1_model, s1_threshold, s2_model


# ── Data loading ──

def load_dns_log(path):
    """Load DNS log CSV, return records sorted by timestamp."""
    records = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    records.sort(key=lambda r: r['timestamp'])
    return records


def parse_ts(ts_str):
    """Parse 'YYYY-MM-DD HH:MM:SS.ffffff' → Unix epoch float."""
    return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f').timestamp()


# ── Stage 1 ──

def stage1_filter(records, s1_model, s1_threshold, bigram_freq, english_words):
    """
    Run Stage 1 LR on every DNS record.

    Returns:
        suspicious: list of records with prob >= threshold
        metrics: dict with tp, fn, tn, fp (record-level counts)
    """
    suspicious = []
    tp = 0
    fn = 0
    tn = 0
    fp = 0

    for rec in records:
        domain = rec['domain']
        label = int(rec['label'])
        feat_list = extract_all_string_features(domain, bigram_freq, english_words)
        prob = s1_model.predict_proba(np.array([feat_list]))[0, 1]

        if prob >= s1_threshold:
            suspicious.append(rec)
            if label == 1:
                tp += 1
            else:
                fp += 1
        else:
            if label == 0:
                tn += 1
            else:
                fn += 1

    return suspicious, {'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp}


# ── Sliding window ──

def sliding_window(records, window_sec, slide_sec):
    """Yield (window_start_epoch, window_records) tuples."""
    if not records:
        return
    start_ts = parse_ts(records[0]['timestamp'])
    end_ts = parse_ts(records[-1]['timestamp'])
    ws = start_ts
    while ws < end_ts:
        we = ws + window_sec
        wr = [r for r in records if ws <= parse_ts(r['timestamp']) < we]
        if wr:
            yield ws, wr
        ws += slide_sec


# ── Stage 2 — traffic feature extraction ──

def extract_traffic_features(records):
    """
    Extract 8-dim traffic features from S1-filtered records.

    Groups records by 5-min sliding window, then by domain within each window.
    Returns list of {domain, label, family, features, window_start}.
    """
    window_sec = WINDOW_MINUTES * 60
    slide_sec = SLIDE_MINUTES * 60
    samples = []
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
            mt = float(np.mean(ttls))
            ts_std = float(np.std(ttls)) if len(ttls) > 1 else 0.0
            qtypes = len(set(q['qtype'] for q in queries))
            qd = qtypes / len(QTYPE_SET)
            cur_ts = parse_ts(queries[0]['timestamp'])
            fsd = cur_ts - first_seen_times[domain]
            burst = n / max(len(window_recs), 1)

            samples.append({
                'domain': domain,
                'label': int(queries[0]['label']),
                'family': queries[0].get('family', ''),
                'features': [n, src_ips, nx_r, mt, ts_std, qd, fsd, burst],
                'window_start': ws,
            })

    return samples


# ── Main pipeline ──

def main():
    print("=" * 60)
    print("Cascade Pipeline — End-to-End Evaluation")
    print("=" * 60)

    # ── 1. Load resources ──
    print("\n[1/5] Loading resources...")
    bigram_freq, english_words, s1_model, s1_threshold, s2_model = load_resources()
    print(f"   S1 threshold (Recall>=95% tuned): {s1_threshold:.4f}")

    # ── 2. Load DNS log ──
    print("\n[2/5] Loading DNS log...")
    all_records = load_dns_log(DNS_LOG)
    total_queries = len(all_records)
    n_dga = sum(1 for r in all_records if int(r['label']) == 1)
    n_benign = sum(1 for r in all_records if int(r['label']) == 0)
    print(f"   Total queries: {total_queries:,}  "
          f"(benign={n_benign:,}  dga={n_dga:,})")

    # ── 3. Stage 1 — per-record LR filtering ──
    print("\n[3/5] Stage 1 — per-record LR filtering (14 string features)...")
    suspicious, s1_m = stage1_filter(
        all_records, s1_model, s1_threshold, bigram_freq, english_words)
    s1_recall = s1_m['tp'] / n_dga if n_dga > 0 else 0.0
    s1_tnr = s1_m['tn'] / n_benign if n_benign > 0 else 0.0
    print(f"   S1 Recall={s1_recall:.4f}  TNR={s1_tnr:.4f}")
    print(f"   Suspicious -> S2: {len(suspicious):,} records  "
          f"(TP={s1_m['tp']:,}  FP={s1_m['fp']:,}  "
          f"TN={s1_m['tn']:,}  FN={s1_m['fn']:,})")

    # ── 4. Stage 2 — traffic features + RF ──
    print("\n[4/5] Stage 2 — extracting 8-dim traffic features + RF inference...")
    samples = extract_traffic_features(suspicious)
    n_samples = len(samples)
    print(f"   Window-domain samples extracted: {n_samples:,}")

    # Predict all samples with S2 RF
    s2_preds = []
    s2_sample_labels = [s['label'] for s in samples]
    s2_domain_flags = {}  # domain -> bool: any window flagged DGA

    if n_samples > 0:
        X_s2 = np.array([s['features'] for s in samples])
        s2_preds = s2_model.predict(X_s2).tolist()

        for s, pred in zip(samples, s2_preds):
            if pred == 1:
                s2_domain_flags[s['domain']] = True
    else:
        s2_preds = []

    s2_precision_val = float(precision_score(
        s2_sample_labels, s2_preds, zero_division=0))

    # ── 5. Domain-level aggregation ──
    print("\n[5/5] Domain-level aggregation...")

    # Ground-truth domain labels from all records
    domain_labels = {}
    for r in all_records:
        domain_labels[r['domain']] = int(r['label'])

    all_domains = sorted(domain_labels.keys())
    dga_domains = [d for d in all_domains if domain_labels[d] == 1]
    benign_domains = [d for d in all_domains if domain_labels[d] == 0]
    n_dga_domains = len(dga_domains)
    n_benign_domains = len(benign_domains)

    # S1 domain-level: detected if ANY record passed S1 threshold
    s1_domain_hits = {}
    for r in suspicious:
        s1_domain_hits[r['domain']] = True

    # Cascade: domain detected if S1 detected AND S2 detected
    cascade_tp = 0
    cascade_fp = 0
    cascade_tn = 0
    cascade_fn = 0

    for d in all_domains:
        label = domain_labels[d]
        s1_detected = s1_domain_hits.get(d, False)
        s2_detected = s2_domain_flags.get(d, False)
        cascade_detected = s1_detected and s2_detected

        if label == 1:
            if cascade_detected:
                cascade_tp += 1
            else:
                cascade_fn += 1
        else:
            if cascade_detected:
                cascade_fp += 1
            else:
                cascade_tn += 1

    cascade_recall = cascade_tp / n_dga_domains if n_dga_domains > 0 else 0.0
    cascade_fpr = cascade_fp / n_benign_domains if n_benign_domains > 0 else 0.0
    cascade_precision = (cascade_tp / (cascade_tp + cascade_fp)
                         if (cascade_tp + cascade_fp) > 0 else 0.0)

    print(f"\n   Unique domains: {len(all_domains):,}  "
          f"(DGA={n_dga_domains:,}  Benign={n_benign_domains:,})")
    print(f"   Cascade: TP={cascade_tp}  FP={cascade_fp}  "
          f"TN={cascade_tn}  FN={cascade_fn}")
    print(f"   Cascade Recall={cascade_recall:.4f}  "
          f"FPR={cascade_fpr:.4f}  Precision={cascade_precision:.4f}")

    # ── Build output dict ──
    output = {
        'total_queries': total_queries,
        'benign_queries': n_benign,
        'dga_queries': n_dga,
        'unique_domains': len(all_domains),
        'unique_dga_domains': n_dga_domains,
        'unique_benign_domains': n_benign_domains,
        # Stage 1 — record-level
        'stage1_threshold': round(s1_threshold, 4),
        'stage1_filtered': len(suspicious),
        'stage1_tp': s1_m['tp'],
        'stage1_fn': s1_m['fn'],
        'stage1_tn': s1_m['tn'],
        'stage1_fp': s1_m['fp'],
        'stage1_recall': round(s1_recall, 4),
        'stage1_tnr': round(s1_tnr, 4),
        # Stage 2 — sample-level
        'stage2_samples': n_samples,
        'stage2_precision': round(s2_precision_val, 4),
        # Cascade — domain-level
        'cascade_tp': cascade_tp,
        'cascade_fp': cascade_fp,
        'cascade_tn': cascade_tn,
        'cascade_fn': cascade_fn,
        'cascade_recall': round(cascade_recall, 4),
        'cascade_fpr': round(cascade_fpr, 4),
        'cascade_precision': round(cascade_precision, 4),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n   Output -> {OUT_PATH}")

    print("\n" + "=" * 60)
    print("Cascade pipeline complete.")
    print("=" * 60)


if __name__ == '__main__':
    main()

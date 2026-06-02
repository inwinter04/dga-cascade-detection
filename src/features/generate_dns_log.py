"""
DNS 日志生成器 v1 — 为 Stage 2 流量特征训练提供合成数据

设计原理：
  - 良性域名：模拟正常用户 DNS 查询（随机时间分布、NOERROR、正常 TTL）
  - DGA 域名：模拟恶意软件批量查询（突发聚合、高 NXDOMAIN 率、短 TTL）
  - 输出 CSV 日志，每行一条 DNS 查询记录

使用方式：
  python3 generate_dns_log.py --benign 10000 --dga 5000 --duration 24 --output dns_log.csv

输出格式：
  timestamp,domain,qtype,rcode,ttl,src_ip,label
"""

import csv
import json
import random
import argparse
import math
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[2]

random.seed(42)

# ── 配置 ──
BENIGN_CSV = PROJECT_ROOT / "data" / "processed" / "benign_features.csv"
DGA_DOMAINS = PROJECT_ROOT / "data" / "processed" / "dga_domains.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# 查询类型分布（模拟真实 DNS 流量）
QTYPES = ['A', 'AAAA', 'MX', 'TXT', 'CNAME']
QTYPE_WEIGHTS = [0.85, 0.08, 0.04, 0.02, 0.01]  # A 占绝大多数

# 良性 TTL 分布（秒）
TTL_BENIGN_WEIGHTS = {
    (60, 120): 0.15,      # 短 TTL（CDN）
    (300, 600): 0.35,     # 常规
    (3600, 14400): 0.30,  # 较长
    (86400, 172800): 0.20 # 长 TTL
}

# DGA TTL 分布（模拟恶意域名更短的 TTL）
TTL_DGA_WEIGHTS = {
    (60, 120): 0.30,
    (300, 600): 0.50,
    (3600, 14400): 0.15,
    (86400, 172800): 0.05
}

# 源 IP 池（模拟多个内网客户端）
NUM_CLIENTS = 50

# DGA 突发窗口参数
# 模拟 DGA bot 每 N 分钟批量查询一批域名
DGA_BURST_INTERVAL_MIN = 5      # 每 5-15 分钟一次突发
DGA_BURST_INTERVAL_MAX = 15
DGA_DOMAINS_PER_BURST = 50       # 每次突发查询 50-200 个域名
DGA_DOMAINS_PER_BURST_MAX = 200

# DGA NXDOMAIN 率
# 真实 DGA：大部分域名解析失败，只有少量成功
DGA_NX_RATE = 0.85   # 85% 的 DGA 查询返回 NXDOMAIN

# 良性 NXDOMAIN 率（真实网络中打字错误、过期域名等也会产生）
BENIGN_NX_RATE = 0.03  # 3% 的良性查询返回 NXDOMAIN


def load_domains():
    """加载良性域名和 DGA 域名"""
    # 从 Tranco 读取良性域名
    benign = []
    tranco_path = PROJECT_ROOT / "data" / "raw" / "tranco_top1m.csv"
    with open(tranco_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                benign.append(parts[1].strip())

    # DGA 域名
    dga = []
    families = []
    with open(DGA_DOMAINS, 'r') as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                dga.append(parts[1].strip())
                families.append(parts[0].strip())

    return benign, dga, families


def sample_ttl(is_benign):
    """按权重采样 TTL 值"""
    weights = TTL_BENIGN_WEIGHTS if is_benign else TTL_DGA_WEIGHTS
    ranges = list(weights.keys())
    probs = list(weights.values())
    chosen = random.choices(ranges, weights=probs, k=1)[0]
    return random.randint(chosen[0], chosen[1])


def generate_benign_traffic(domains, num_queries, duration_hours, start_time):
    """生成良性 DNS 查询流量（自然分布）"""
    records = []
    clients = [f"10.0.0.{i+1}" for i in range(NUM_CLIENTS)]
    # 选前 num_queries 个（前 = 更常见的域名）
    sample = domains[:min(num_queries, len(domains))]
    
    # 模拟昼夜波动：10:00-22:00 高峰，其他时间低峰
    for i, domain in enumerate(sample):
        t = random.uniform(0, duration_hours * 3600)
        # 调整时间权重：偏向白天
        hour = (start_time.hour + t / 3600) % 24
        if 10 <= hour <= 22:
            pass  # 白天，自然概率
        else:
            t *= 1.5  # 夜间稀疏，用乘法增大分散
        if t >= duration_hours * 3600:
            t = random.uniform(0, duration_hours * 3600)
        
        ts = start_time + timedelta(seconds=t)
        qtype = random.choices(QTYPES, weights=QTYPE_WEIGHTS, k=1)[0]
        # 加入良性 NXDOMAIN 噪声（模拟打字错误、过期域名）
        rcode = 'NXDOMAIN' if random.random() < BENIGN_NX_RATE else 'NOERROR'
        ttl = sample_ttl(is_benign=True)
        src_ip = random.choice(clients)
        
        records.append({
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'domain': domain,
            'qtype': qtype,
            'rcode': rcode,
            'ttl': ttl,
            'src_ip': src_ip,
            'label': 0,
        })
    
    return records


def generate_dga_traffic(domains, families, num_queries, duration_hours, start_time):
    """生成 DGA 域名查询流量（突发批量模式）"""
    records = []
    clients = [f"10.0.0.{i+1}" for i in range(NUM_CLIENTS)]
    
    sample = random.sample(domains, min(num_queries, len(domains)))
    
    # 按突发窗口组织
    burst_interval = random.randint(DGA_BURST_INTERVAL_MIN, DGA_BURST_INTERVAL_MAX)
    domains_per_burst = random.randint(DGA_DOMAINS_PER_BURST, DGA_DOMAINS_PER_BURST_MAX)
    
    idx = 0
    current_time_sec = 0
    max_sec = duration_hours * 3600
    
    while idx < len(sample) and current_time_sec < max_sec:
        # 一次突发
        burst_end = min(idx + domains_per_burst, len(sample))
        burst_domains = sample[idx:burst_end]
        
        # 真实 DGA 行为：大部分域名查 1 次，少数"重点域名"查多次（模拟 C2 反复探测）
        base_domains = []  # 查 1 次
        repeat_domains = []  # 查多次
        
        for domain in burst_domains:
            if random.random() < 0.25:  # 25% 的域名是"重点"
                repeat_count = random.randint(3, 10)
                repeat_domains.extend([domain] * repeat_count)
            else:
                base_domains.append(domain)
        
        all_burst_queries = base_domains + repeat_domains
        random.shuffle(all_burst_queries)  # 打乱，模拟真实查询顺序
        
        for domain in all_burst_queries:
            t = current_time_sec + random.uniform(0, 5.0)  # 5 秒内密集查询
            if t >= max_sec:
                break
            
            ts = start_time + timedelta(seconds=t)
            qtype = random.choices(QTYPES, weights=QTYPE_WEIGHTS, k=1)[0]
            
            # DGA 高 NXDOMAIN 率
            rcode = 'NXDOMAIN' if random.random() < DGA_NX_RATE else 'NOERROR'
            ttl = sample_ttl(is_benign=False)
            src_ip = random.choice(clients)
            
            # 查找对应的 family
            domain_idx = domains.index(domain) if domain in domains else 0
            family = families[domain_idx] if domain_idx < len(families) else 'unknown'
            
            records.append({
                'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'domain': domain,
                'qtype': qtype,
                'rcode': rcode,
                'ttl': ttl,
                'src_ip': src_ip,
                'label': 1,
                'family': family,
            })
        
        idx = burst_end
        # 进入静默期
        current_time_sec += burst_interval * 60
    
    return records


def generate_mixed_traffic(benign_domains, dga_domains, dga_families,
                           num_benign, num_dga, duration_hours, start_time):
    """生成混合 DNS 流量"""
    benign_records = generate_benign_traffic(
        benign_domains, num_benign, duration_hours, start_time
    )
    dga_records = generate_dga_traffic(
        dga_domains, dga_families, num_dga, duration_hours, start_time
    )
    
    all_records = benign_records + dga_records
    # 按时间排序
    all_records.sort(key=lambda r: r['timestamp'])
    
    return all_records


def write_csv(records, output_path):
    """写入 CSV 文件"""
    fieldnames = ['timestamp', 'domain', 'qtype', 'rcode', 'ttl', 'src_ip', 'label']
    if any('family' in r for r in records):
        fieldnames.append('family')
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    
    # 统计
    total = len(records)
    benign = sum(1 for r in records if r['label'] == 0)
    dga = sum(1 for r in records if r['label'] == 1)
    nx_total = sum(1 for r in records if r['rcode'] == 'NXDOMAIN')
    nx_dga = sum(1 for r in records if r['label'] == 1 and r['rcode'] == 'NXDOMAIN')
    
    print(f"\n{'='*50}")
    print(f"DNS 日志生成完成")
    print(f"{'='*50}")
    print(f"  输出文件: {output_path}")
    print(f"  总记录数: {total:,}")
    print(f"  良性查询: {benign:,} ({benign/total*100:.1f}%)")
    print(f"  DGA 查询: {dga:,} ({dga/total*100:.1f}%)")
    print(f"  NXDOMAIN 总数: {nx_total:,} ({nx_total/total*100:.1f}%)")
    print(f"  DGA 中 NXDOMAIN: {nx_dga:,} ({nx_dga/dga*100:.1f}%)")
    print(f"  时间跨度: {records[0]['timestamp']} ~ {records[-1]['timestamp']}")
    print(f"{'='*50}\n")
    
    # 保存统计信息
    stats = {
        'total_records': total,
        'benign_queries': benign,
        'dga_queries': dga,
        'nxdomain_total': nx_total,
        'nxdomain_dga': nx_dga,
        'nxdomain_benign': nx_total - nx_dga,
        'time_start': records[0]['timestamp'],
        'time_end': records[-1]['timestamp'],
        'dga_burst_interval_min': DGA_BURST_INTERVAL_MIN,
        'dga_burst_interval_max': DGA_BURST_INTERVAL_MAX,
        'dga_domains_per_burst_range': f"{DGA_DOMAINS_PER_BURST}-{DGA_DOMAINS_PER_BURST_MAX}",
        'dga_nx_rate': DGA_NX_RATE,
    }
    stats_path = output_path.with_suffix('.stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  统计信息 → {stats_path}")


def main():
    parser = argparse.ArgumentParser(description='DNS 日志生成器 v1')
    parser.add_argument('--benign', type=int, default=50000,
                        help='良性查询数 (default: 50000)')
    parser.add_argument('--dga', type=int, default=5000,
                        help='DGA 查询数 (default: 5000)')
    parser.add_argument('--duration', type=int, default=24,
                        help='模拟时长（小时）(default: 24)')
    parser.add_argument('--output', type=str, default='dns_log_train.csv',
                        help='输出文件名 (default: dns_log_train.csv)')
    parser.add_argument('--start-time', type=str, default='2026-06-01 08:00:00',
                        help='模拟开始时间 (default: 2026-06-01 08:00:00)')
    args = parser.parse_args()
    
    print("=" * 50)
    print("DNS 日志生成器 v1 — 为 Stage 2 提供训练数据")
    print("=" * 50)
    
    print(f"\n[1/3] 加载域名数据...")
    benign_domains, dga_domains, dga_families = load_domains()
    print(f"   良性域名池: {len(benign_domains):,}")
    print(f"   DGA 域名池: {len(dga_domains):,}")
    
    print(f"\n[2/3] 生成 DNS 查询流量...")
    print(f"   良性查询: {args.benign:,}")
    print(f"   DGA 查询: {args.dga:,}")
    print(f"   模拟时长: {args.duration} 小时")
    
    start_time = datetime.fromisoformat(args.start_time)
    
    records = generate_mixed_traffic(
        benign_domains, dga_domains, dga_families,
        args.benign, args.dga, args.duration, start_time
    )
    
    print(f"\n[3/3] 写入 CSV...")
    output_path = OUTPUT_DIR / args.output
    write_csv(records, output_path)


if __name__ == '__main__':
    main()

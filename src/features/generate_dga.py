"""
DGA 域名生成器

基于 baderj/domain_generation_algorithms (53个家族)
用 example_domains.txt + 直接运行 dga.py 的方式生成 DGA 域名

输出: data/processed/dga_domains.csv (family,domain)
"""

import sys
import os
import subprocess
import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BADERJ_DIR = PROJECT_ROOT / "data" / "raw" / "baderj_dga"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "dga_domains.csv"
DOMAINS_PER_FAMILY = 500  # 每家族生成 500 个

# ── 53 个家族清单 ──
FAMILIES = [
    'banjori', 'ramnit', 'qakbot', 'necurs', 'simda', 'nymaim', 'tinba',
    'padcrypt', 'shiotob', 'gozi', 'kraken', 'locky', 'murofet', 'mydoom',
    'pykspa', 'ranbyus', 'suppobox', 'corebot', 'dnschanger', 'expiro',
    'fobber', 'fosniw', 'newgoz', 'ngioweb', 'orchard', 'pitou', 'pizd',
    'proslikefan', 'pushdo', 'qadars', 'qsnatch', 'ramdo', 'reconyc',
    'sharkbot', 'sisron', 'symmi', 'tempedreve', 'tufik',
    'unnamed_downloader', 'unnamed_javascript_dga', 'vawtrak', 'verblecon',
    'zloader', 'bazarbackdoor', 'bumblebee', 'charbot', 'chinad',
    'darkcracks', 'dircrypt', 'dmsniff', 'm0yv', 'monerodownloader'
]


def collect_example_domains(family: str) -> list:
    """从 example_domains.txt 收集域名"""
    fam_dir = BADERJ_DIR / family
    domains = []

    # 找所有 example/domain 相关的文件
    patterns = [
        'example_domains*.txt', 'domains_*.txt',
        'domains_examples.txt'
    ]
    for pattern in patterns:
        for f in sorted(fam_dir.glob(pattern)):
            with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                for line in fh:
                    domain = line.strip()
                    if domain and not domain.startswith('#') and '.' in domain:
                        # 清理可能带有的注释
                        domain = domain.split('#')[0].strip()
                        domains.append(domain)

    return domains


def run_dga_script(family: str) -> list:
    """直接运行 dga.py 生成域名"""
    fam_dir = BADERJ_DIR / family
    domains = []

    dga_file = fam_dir / 'dga.py'
    if not dga_file.exists():
        # 有些家族分版本的
        for sub in sorted(fam_dir.iterdir()):
            if sub.is_dir():
                sub_dga = sub / 'dga.py'
                if sub_dga.exists():
                    try:
                        result = subprocess.run(
                            [sys.executable, str(sub_dga)],
                            capture_output=True, text=True, timeout=30,
                            cwd=str(fam_dir)
                        )
                        for line in result.stdout.split('\n'):
                            d = line.strip()
                            if d and '.' in d:
                                domains.append(d)
                    except Exception:
                        pass
        return domains

    # 检查脚本类型
    with open(dga_file, 'r', encoding='utf-8', errors='ignore') as f:
        code = f.read()

    # 方式1: 有 argparse 的 → 尝试用默认参数跑
    if 'argparse' in code:
        try:
            result = subprocess.run(
                [sys.executable, str(dga_file)],
                capture_output=True, text=True, timeout=30,
                cwd=str(fam_dir)
            )
            for line in result.stdout.split('\n'):
                d = line.strip()
                if d and '.' in d:
                    domains.append(d)
        except Exception:
            pass

        # 如果没输出或报错，尝试带参数
        if not domains:
            for args in [
                ['-n', str(DOMAINS_PER_FAMILY)],
                ['-x', str(DOMAINS_PER_FAMILY)],
                ['-d', str(DOMAINS_PER_FAMILY)],
                ['--number', str(DOMAINS_PER_FAMILY)],
            ]:
                try:
                    result = subprocess.run(
                        [sys.executable, str(dga_file)] + args,
                        capture_output=True, text=True, timeout=30,
                        cwd=str(fam_dir)
                    )
                    for line in result.stdout.split('\n'):
                        d = line.strip()
                        if d and '.' in d:
                            domains.append(d)
                    if domains:
                        break
                except Exception:
                    continue

    # 方式2: 简单脚本（直接打印）
    elif 'for i in range' in code or 'print(domain)' in code or 'print(' in code:
        try:
            result = subprocess.run(
                [sys.executable, str(dga_file)],
                capture_output=True, text=True, timeout=30,
                cwd=str(fam_dir)
            )
            for line in result.stdout.split('\n'):
                d = line.strip()
                if d and '.' in d:
                    domains.append(d)
        except Exception:
            pass

    return domains


def main():
    os.chdir(str(BADERJ_DIR))

    all_domains = []  # (family, domain)
    stats = []

    print(f"DGA 域名生成 — 共 {len(FAMILIES)} 个家族")
    print("=" * 60)

    for family in FAMILIES:
        print(f"[{family:22s}] ", end='', flush=True)

        # 优先用 example_domains.txt
        domains = collect_example_domains(family)[:DOMAINS_PER_FAMILY]

        if domains:
            source = "example"
        else:
            # 没有例子文件的，运行 dga.py
            domains = run_dga_script(family)[:DOMAINS_PER_FAMILY]
            source = "generated"

        # 去重
        seen = set()
        unique_domains = []
        for d in domains:
            if d not in seen:
                seen.add(d)
                unique_domains.append(d)

        all_domains.extend((family, d) for d in unique_domains)
        stats.append((family, len(unique_domains), source))
        print(f" {len(unique_domains):>4} 个域名 ({source})")

    # 保存 CSV
    print(f"\n保存 {len(all_domains)} 个 DGA 域名 → {OUTPUT_CSV}")
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['family', 'domain'])
        for family, domain in all_domains:
            writer.writerow([family, domain])

    # 统计
    print("\n" + "=" * 60)
    print(f"总计: {len(all_domains)} 个 DGA 域名")
    print(f"家族: {len([s for s in stats if s[1] > 0])}/{len(FAMILIES)} 个成功")
    print()

    zero = [s for s in stats if s[1] == 0]
    if zero:
        print("以下家族未能生成域名:")
        for fam, _, _ in zero:
            print(f"  - {fam}")


if __name__ == "__main__":
    main()

"""
良性域名特征分布可视化

为 14 维特征中最重要的几个特征生成分布直方图
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 中文字体支持
import os
CN_FONT = None
for fp in ['/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc']:
    if os.path.exists(fp):
        CN_FONT = FontProperties(fname=fp)
        break
# 直接用全局
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC',
                                     'SimHei', 'DejaVu Sans']
# 不改标签，直接用中文

# ── 路径 ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
csv_path = PROJECT_ROOT / "data" / "processed" / "benign_features.csv"
out_dir = PROJECT_ROOT / "output" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

# ── 加载数据 ──
data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
ranks = data[:, 0]
features = {
    'domain_length':      data[:, 1],
    'sld_length':         data[:, 2],
    'digit_ratio':        data[:, 3],
    'vowel_ratio':        data[:, 4],
    'consonant_ratio':    data[:, 5],
    'unique_char_ratio':  data[:, 6],
    'normalized_entropy': data[:, 7],
    'max_consonant_run':  data[:, 8],
    'max_digit_run':      data[:, 9],
    'vc_alternation':     data[:, 10],
    'subdomain_count':    data[:, 11],
    'has_hyphen':         data[:, 12],
    'bigram_anomaly':     data[:, 13],
    'meaningful_ratio':   data[:, 14],
}

# ── 中文标签映射 ──
CN_LABELS = {
    'domain_length':      '域名总长度',
    'sld_length':         'SLD 长度',
    'digit_ratio':        '数字占比',
    'vowel_ratio':        '元音占比',
    'consonant_ratio':    '辅音占比',
    'unique_char_ratio':  '唯一字符占比',
    'normalized_entropy': 'Shannon 熵 (归一化)',
    'max_consonant_run':  '最大连续辅音',
    'max_digit_run':      '最大连续数字',
    'vc_alternation':     '元辅音切换率',
    'subdomain_count':    '子域名层数',
    'has_hyphen':         '含连字符',
    'bigram_anomaly':     'Bigram 异常度',
    'meaningful_ratio':   '英文单词覆盖率',
}

# ── 要画的特征（包含最重要的几个） ──
KEY_FEATURES = [
    'domain_length', 'sld_length', 'digit_ratio',
    'vowel_ratio', 'meaningful_ratio',
    'normalized_entropy', 'max_consonant_run', 'max_digit_run',
    'vc_alternation', 'bigram_anomaly',
]

# ── 配色 ──
COLOR = '#4A7CF7'

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'figure.dpi': 150,
})


def plot_feature(name, data_array, cn_label):
    fig, ax = plt.subplots(figsize=(7, 4))

    # 根据特征类型选择 bin 策略
    if name in ('digit_ratio', 'vowel_ratio', 'consonant_ratio',
                 'unique_char_ratio', 'vc_alternation', 'meaningful_ratio'):
        # 连续值 [0,1]
        bins = np.linspace(0, 1, 51)
    elif name in ('max_digit_run', 'subdomain_count', 'has_hyphen'):
        # 离散小整数 -> 用整数 bin
        bins = np.arange(-0.5, data_array.max() + 1.5, 1)
    elif name in ('domain_length', 'sld_length'):
        bins = np.arange(-0.5, data_array.max() + 1.5, 1)
    else:
        bins = 50

    ax.hist(data_array, bins=bins, color=COLOR, edgecolor='white',
            linewidth=0.3, alpha=0.85)

    # 统计信息标注
    mean_val = np.mean(data_array)
    median_val = np.median(data_array)
    ymax = ax.get_ylim()[1]

    ax.axvline(mean_val, color='#E74C3C', linestyle='--', linewidth=1.5,
               label=f'均值={mean_val:.3f}')
    ax.axvline(median_val, color='#F39C12', linestyle=':', linewidth=1.5,
               label=f'中位数={median_val:.3f}')
    ax.legend(fontsize=9)

    ax.set_xlabel(cn_label, fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    ax.set_title(f'{cn_label} 分布 (N=200,000)', fontsize=13, fontweight='bold')

    # 添加 P5/P95 范围标注
    p5, p95 = np.percentile(data_array, [5, 95])
    ax.annotate(f'P5={p5:.3f}  P95={p95:.3f}',
                xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # 保存
    out_path = out_dir / f'benign_{name}.png'
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✅ {out_path.name}')
    return out_path


# ── 单独画 meaningful_ratio 的放大版（含分段标注） ──
def plot_meaningful_ratio_detailed():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    mr = features['meaningful_ratio']

    bins = np.linspace(0, 1, 51)
    counts, edges, patches = ax.hist(mr, bins=bins, color=COLOR, edgecolor='white',
                                      linewidth=0.3, alpha=0.85)

    # 着色区间
    for patch, left, right in zip(patches, edges[:-1], edges[1:]):
        if left < 0.01:
            patch.set_facecolor('#E74C3C')  # 低覆盖 - 红
            patch.set_alpha(0.7)
        elif left < 0.5:
            patch.set_facecolor('#F39C12')  # 中等 - 橙
            patch.set_alpha(0.7)

    mean_val = np.mean(mr)
    median_val = np.median(mr)
    ax.axvline(mean_val, color='#E74C3C', linestyle='--', linewidth=1.5,
               label=f'均值={mean_val:.3f}')
    ax.axvline(median_val, color='#2ECC71', linestyle='-', linewidth=1.5,
               label=f'中位数={median_val:.3f}')

    # 分段标注
    p5 = np.percentile(mr, 5)
    ax.annotate(f'P5={p5:.3f}\n仅 5% 低于此值',
                xy=(p5, ax.get_ylim()[1]*0.7),
                fontsize=9, color='#E74C3C',
                arrowprops=dict(arrowstyle='->', color='#E74C3C'))

    ax.set_xlabel('英文单词覆盖率', fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    ax.set_title('meaningful_ratio 分布 (N=200,000) — 强区分特征', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    # 分段统计表格
    tab_data = [
        ['= 0', f'{(mr==0).sum():,}', f'{(mr==0).mean()*100:.1f}%'],
        ['(0, 0.5)', f'{((mr>0)&(mr<0.5)).sum():,}', f'{((mr>0)&(mr<0.5)).mean()*100:.1f}%'],
        ['[0.5, 1)', f'{((mr>=0.5)&(mr<1)).sum():,}', f'{((mr>=0.5)&(mr<1)).mean()*100:.1f}%'],
        ['= 1', f'{(mr==1).sum():,}', f'{(mr==1).mean()*100:.1f}%'],
    ]
    table = ax.table(cellText=tab_data,
                     colLabels=['区间', '数量', '占比'],
                     loc='upper right',
                     cellLoc='center',
                     bbox=[0.65, 0.55, 0.33, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    out_path = out_dir / f'benign_meaningful_ratio_detailed.png'
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✅ {out_path.name}')
    return out_path


# ── 执行 ──
print('生成特征分布图...')
saved = []
for name in KEY_FEATURES:
    path = plot_feature(name, features[name], CN_LABELS[name])
    saved.append(path)

path = plot_meaningful_ratio_detailed()
saved.append(path)

print(f'\n共生成 {len(saved)} 张图 → {out_dir}/')

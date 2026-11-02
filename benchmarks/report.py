"""Export aggregate experiment evidence; never export raw rows or predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .run import write_json

METRICS = ['precision', 'recall', 'f1', 'fpr', 'average_precision',
           'pr_auc_trapezoid', 'roc_auc', 'alarm_fraction']
LABELS = {'kronos_adapter': 'Kronos adapter', 'random_adapter': 'Random adapter',
          'isolation_forest': 'Isolation Forest'}


def export(runs, cache, output, document):
    env = json.loads((runs / 'environment.json').read_text())
    expected = {(d, m, s) for d in env['argv']['datasets'] for m in env['argv']['methods']
                for s in env['argv']['seeds']}
    records = [json.loads(p.read_text()) for p in sorted(runs.glob('*/*/result.json'))]
    found = {(r['dataset'], r['method'], r['seed']) for r in records}
    if found != expected or any(r['status'] != 'complete' for r in records):
        raise ValueError(f'Incomplete or unexpected runs: missing={expected-found}, unexpected={found-expected}')
    output.mkdir(parents=True, exist_ok=False)
    # Paths do not affect scientific reproducibility and need not expose a user workspace.
    for key in ('cache', 'output', 'weights'):
        env['argv'][key] = {'cache': '.local_data/unsupervised_v1',
                            'output': '.local_data/unsupervised_runs_v1_full',
                            'weights': 'weights/Kronos-mini'}[key]
    write_json(output / 'environment.json', env)
    write_json(output / 'runs.json', records)
    splits = {d: json.loads((cache / d / 'manifest.json').read_text()) for d in env['argv']['datasets']}
    write_json(output / 'splits.json', splits)
    flat = []
    for r in records:
        flat.append({'dataset': r['dataset'], 'method': r['method'], 'seed': r['seed'],
                     'train_rows_used': r['train_rows_used'], 'calibration_rows_used': r['calibration_rows_used'],
                     'pot_threshold': r['threshold']['threshold'], 'train_seconds': r['train_seconds'],
                     'total_seconds': r['total_seconds'], **r['metrics']})
    frame = pd.DataFrame(flat).sort_values(['dataset', 'method', 'seed'])
    frame.to_csv(output / 'metrics.csv', index=False)
    summary = []
    for (d, m), group in frame.groupby(['dataset', 'method'], sort=False):
        row = {'dataset': d, 'method': m, 'seeds': len(group),
               'evaluated_per_seed': int(group.evaluated.iloc[0]),
               'positives_per_seed': int(group.positives.iloc[0]),
               'unknown_excluded_per_seed': int(group.unknown_excluded.iloc[0])}
        for metric in METRICS:
            row[metric+'_mean'] = float(group[metric].mean())
            row[metric+'_std'] = float(group[metric].std(ddof=0))
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(output / 'summary.csv', index=False)
    plot(summary, output / 'comparison.png', env['argv']['datasets'])

    def fmt(value):
        return f'{100*value:.2f}'

    methods = ['kronos_adapter', 'isolation_forest', 'random_adapter']
    ordered = {d: summary[summary.dataset.eq(d)].set_index('method') for d in env['argv']['datasets']}
    lines = ['# 无监督金融异常检测实验结果 v1', '',
             f'完成 **{len(splits)} 个任务配置、{len(methods)} 种方法、每种 3 个种子，共 {len(records)} 次真实运行**。9 组数据均已覆盖；SD-FMM 分 stock/crypto 两个任务。Sapienza 本轮使用 15S，同事件的 5S/25S 不作独立数据集重复汇总。', '',
             '**结论必须按数据集看。以下 Kronos 成绩来自新加入的特征自编码适配器，不是原版 OHLC 预测误差检测器。单 token 骨干不建模跨时间注意力；本轮验证的是无标签特征重构迁移。**', '',
             '训练与 POT 校准不访问标签，全部测试记录保留自然类别比例。每个种子最多均匀抽取 50,000 个训练样本、100,000 个校准样本；测试不抽样。阈值使用固定 POT（q=0.85，risk=0.01，min_peaks=20），不是根据测试 F1 选择。训练包含未知异常污染。详见[预先固定的协议](unsupervised_benchmark_protocol_CN.md)。', '',
             '## Kronos 适配器结果', '',
             '下列指标均为三个种子的均值，单位 %；F1 同时列出总体标准差。AP 为 Average Precision，梯形 PR-AUC 另存 CSV。Precision/Recall/F1 使用固定 POT 阈值。测试数仅计算有真值的记录。', '',
             '| 数据/任务 | 有标签测试数（正例） | Precision | Recall | F1 ± std | AP | ROC-AUC | FPR |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for d, g in ordered.items():
        a = g.loc['kronos_adapter']
        lines.append(f'| {d} | {int(a.evaluated_per_seed):,} ({int(a.positives_per_seed):,}) | {fmt(a.precision_mean)} | {fmt(a.recall_mean)} | {fmt(a.f1_mean)} ± {fmt(a.f1_std)} | {fmt(a.average_precision_mean)} | {fmt(a.roc_auc_mean)} | {fmt(a.fpr_mean)} |')
    lines += ['', '## 同协议对照', '',
              '表中为 F1 / AP（%，三种子均值）。随机适配器与 Kronos 适配器的输入/输出层初始化和训练样本一致；随机骨干也保持冻结。未进行显著性检验，不以少量百分点差异声称预训练优势。', '',
              '| 数据/任务 | Kronos 适配器 | Isolation Forest | 随机骨干适配器 |',
              '| --- | ---: | ---: | ---: |']
    for d, g in ordered.items():
        cells = [f'{fmt(g.loc[m].f1_mean)} / {fmt(g.loc[m].average_precision_mean)}' for m in methods]
        lines.append('| ' + d + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '![三种方法的 F1 和 AP 对比](../results/unsupervised_v1/comparison.png)', '',
              '## 解释范围与限制', '',
              '- **检测能力以实测为准。** 异常标签代表欺诈/操纵，统计离群分数不必与其一致。高 ROC-AUC 也可能伴随很低的 Precision/AP；极度不平衡时不可只报告准确率。',
              '- **TSTR 为整段序列级评价**，不是实时点级检测。训练和校准约一半为正，POT 的主要正常假设不成立；不能依据该数据声称稀有异常阈值已经校准。',
              '- **SD-FMM 两个任务为点级评价**，复用出版者已处理的 test 数组重新分组，stock 仅 6 个测试序列；不是原论文基准的独立复现，原全段归一化可能包含未来信息。',
              '- **Sapienza 为作者标注的操纵起点块**，不是完整事件持续区间。不使用 point adjustment、事件补全或邻近容忍窗口，不能与采用这些处理的 F1 直接比较。',
              '- **Elliptic 未知标签不算正常**。未知节点可用于无标签拟合；测试只评价已知合法/非法节点。此处不使用图边，原聚合特征也不能保证因果实时可用。',
              '- **支付数据按时间留出**，并非未见账户留出；PaySim、TabFormer、AML 为合成数据，不能据此承诺真实部署效果。AML/TabFormer 的哈希名义特征只是本轮固定适配，不是完整交易图方法。',
              '- **IEEE-CIS 仅评价官方训练表内的时间留出**，不涉及 Kaggle 官方测试成绩。未接身份表、未针对该竞赛调参；原始数据、训练缓存和逐行输出未上传。',
              '- 一个通用的固定阈值/固定步数配置可能欠拟合或对分布漂移失效。本轮是可复现的初步对照，不是经调参得到的最终性能上限；未跨数据集计算混合平均排名。', '',
              '## 可核验证据与复现', '',
              f'- 实验代码起点：`{env["code_git_head"]}`；运行设备 `{env["argv"]["device"]}`，Python {env["python"]}，PyTorch {env["packages"]["torch"]}，scikit-learn {env["packages"]["scikit-learn"]}。',
              f'- Kronos-mini 权重 SHA256：`{env["weights_sha256"]}`。',
              '- [逐种子全部指标与混淆矩阵](../results/unsupervised_v1/metrics.csv) · [均值和标准差](../results/unsupervised_v1/summary.csv) · [完整聚合运行记录、训练损失、POT 参数和评分哈希](../results/unsupervised_v1/runs.json)。',
              '- [各分区样本数、标签组成和特征清单](../results/unsupervised_v1/splits.json) · [依赖、命令和运行环境](../results/unsupervised_v1/environment.json)。',
              '- 原始数据依照各自授权保留；本报告只发布聚合证据。模型权重、逐行分数、训练抽样索引及预处理参数保存在本地 `.local_data/unsupervised_runs_v1_full/`，未提交到 Git。', '',
              '```bash', 'python -m benchmarks.run --cache .local_data/unsupervised_v1 \\',
              '  --output .local_data/unsupervised_runs_v1_full \\',
              '  --weights weights/Kronos-mini --device mps',
              'python -m benchmarks.report --runs .local_data/unsupervised_runs_v1_full \\',
              '  --cache .local_data/unsupervised_v1 --output results/unsupervised_v1 \\',
              '  --document docs/unsupervised_benchmark_results_CN.md', '```', '',
              '其他设备可选 cpu/cuda；硬件差异不保证逐位相同。输出目录必须为新目录，以保留既有实验。']
    document.write_text('\n'.join(lines)+'\n')
    print(summary.to_string(index=False))


def plot(summary, destination, order):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    colors = ['#2762b3', '#d88823', '#6c8f49']
    for axis, metric, title in zip(axes, ['f1', 'average_precision'], ['F1 at frozen POT threshold', 'Average precision (AP)']):
        for i, (method, label) in enumerate(LABELS.items()):
            s = summary[summary.method.eq(method)].set_index('dataset').loc[order]
            axis.bar(np.arange(len(order))+(i-1)*.25, s[metric+'_mean']*100, width=.24,
                     yerr=s[metric+'_std']*100, label=label, color=colors[i], capsize=2)
        axis.set_ylabel(title+' (%)'); axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True); axis.set_ylim(bottom=0)
    axes[0].legend(ncols=3, loc='upper right')
    axes[1].set_xticks(np.arange(len(order)), order, rotation=25, ha='right')
    fig.suptitle('Unsupervised financial anomaly benchmarks — 3 seeds; no point adjustment\nNew feature adapter; not original OHLC Kronos forecasting', fontsize=12)
    fig.tight_layout(); fig.savefig(destination, dpi=180); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--document', type=Path, required=True)
    args = parser.parse_args()
    export(args.runs, args.cache, args.output, args.document)

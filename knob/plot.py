"""Figures for the Knob experiment (static PNG for the repository report)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, MUTED, INK, SURFACE = '#2a78d6', '#eb6834', '#1baf7a', '#8a8985', '#0b0b0b', '#fcfcfb'
NAMES = {'predictable': 'Boundary A: predictable vs unpredictable', 'smooth': 'Boundary B: smooth vs jumpy forecast'}


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc='left', fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color='#52514e')
    ax.set_ylabel(ylabel, fontsize=9, color='#52514e')
    ax.grid(axis='y', color='#e6e5e1', linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c3c2b7')
    ax.tick_params(labelsize=8, colors='#52514e')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True)
    a = p.parse_args()
    out = Path(a.results)
    r = json.loads((out/'results.json').read_text())
    plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE, 'font.family': 'DejaVu Sans'})

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (b, info) in zip(axes, r['knobs'].items()):
        ks = range(1, len(info['topk_bal_acc']) + 1)
        ax.plot(ks, info['topk_bal_acc'], color=BLUE, linewidth=2, marker='o', markersize=4, label='Top-k ranked dimensions')
        ax.axhline(info['full_probe_bal_acc'], color=INK, linewidth=1.2, linestyle='--', label='All 5 x 256 states')
        ax.axhline(info['random_subsets_bal_acc_mean'], color=ORANGE, linewidth=1.2, linestyle=':', label=f"Random {info['k']} dimensions (mean)")
        ax.axhline(0.5, color=MUTED, linewidth=0.8)
        ax.text(len(info['topk_bal_acc']), 0.505, 'chance', ha='right', fontsize=8, color=MUTED)
        style(ax, NAMES[b], 'Number of hidden dimensions k', 'Held-out balanced accuracy')
        ax.set_ylim(0.48, 0.8)
    axes[0].legend(fontsize=8, frameon=False, loc='lower right')
    fig.tight_layout()
    fig.savefig(out/'knob_identification.png', dpi=160)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    panels = [('jump_mean', 'forecast jump', '|pred close - last close| (window sigma)'),
              ('entropy_mean', 'forecast entropy', 'Next-token entropy (nats)'),
              ('delta_nll', 'change in NLL', 'NLL of realised bar minus unsteered (nats)')]
    for row, b in zip(axes, r['knobs']):
        for ax, (m, short, label) in zip(row, panels):
            for how, color, name in (('knob', BLUE, 'Knob dimensions, normal'), ('dense', AQUA, 'All dimensions, class-mean shift')):
                s = r['selection'][f'{b}/{how}']
                names = ['base' if t == 0 else f'{b}/{how}/normal/{t:+g}' for t in s['taus']]
                ax.plot(s['taus'], [r['effects'][n][m] for n in names], color=color, linewidth=2, marker='o', markersize=4, label=name)
            for d, mk, name in (('knob/tangent', 's', 'Knob dims, tangent (control)'), ('knob/random', '^', 'Knob dims, random direction (control)'),
                                ('random_dims/normal', 'D', 'Random dims, normal (control)')):
                ts = [-4, 4]
                ax.plot(ts, [r['effects'][f'{b}/{d}/{t:+g}'][m] for t in ts], linestyle='none', marker=mk, markersize=6,
                        markerfacecolor='none', markeredgecolor=ORANGE, markeredgewidth=1.5, label=name)
            style(ax, f"{NAMES[b].split(':')[0]}: {short}", 'Step tau (positive = towards predictable / smooth)', label)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out/'knob_effect_on_forecast.png', dpi=160)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
    for row, b in zip(axes, r['knobs']):
        for ax, k in zip(row, ('nll', 'resid_price')):
            for how, color in (('knob', BLUE), ('dense', AQUA)):
                s = r['selection'][f'{b}/{how}']
                ax.plot(s['taus'], [100*v for v in s[k]['test_ap_by_tau']], color=color, linewidth=2, marker='o', markersize=4,
                        label=f"{'Knob dimensions' if how == 'knob' else 'All dimensions'} (test)")
                ax.plot(s['taus'], [100*v for v in s[k]['calibration_ap_by_tau']], color=color, linewidth=1.2, linestyle='--',
                        label=f"{'Knob dimensions' if how == 'knob' else 'All dimensions'} (calibration)")
                ax.axvline(s[k]['selected_tau'], color=color, linewidth=0.8, linestyle=':')
            style(ax, f"{NAMES[b].split(':')[0]} - score: {'token NLL' if k == 'nll' else 'price residual'}", 'Step size tau', 'Average precision (%)')
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out/'knob_detection.png', dpi=160)


if __name__ == '__main__':
    main()

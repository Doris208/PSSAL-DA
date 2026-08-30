#!/usr/bin/env python
"""Plot Fig. E1 process-level training curves for Goji PSSAL-DA.

The script copies a curated 1k-bin CSV into the output directory and writes
vector SVG/PDF line figures for the four process metrics.
"""

import argparse
import csv
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


DEFAULT_INPUT = (
    'shared-nvme/goji26IRRG2goji30RGB/'
    'fig_e1_recommended_4curves_1k_binmean.csv')
DEFAULT_OUT_DIR = (
    'shared-nvme/work_dirs/goji26IRRG2goji30RGB/E1_fig')
DEFAULT_PREFIX = 'goji26IRRG2goji30RGB_fig_e1_process_curves'

CURVES = (
    ('teacher_confidence', 'Teacher confidence', '#3b6fd8', 0.018),
    ('mparc_pseudo_weight', 'Reliability-weighted pseudo supervision',
     '#f0a500', -0.020),
    ('proto_target_update_ratio', 'Prototype update ratio', '#d95f02', 0.018),
    ('boundary_weight_x100', 'Boundary refinement weight (x100)', '#5b5b5b',
     -0.018),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create SVG/PDF Fig. E1 process-curve figures.')
    parser.add_argument(
        '--input',
        default=DEFAULT_INPUT,
        help='Input CSV with iter and four process metrics.')
    parser.add_argument(
        '--out-dir',
        default=DEFAULT_OUT_DIR,
        help='Output directory for SVG/PDF and copied CSV.')
    parser.add_argument(
        '--fig-prefix',
        default=DEFAULT_PREFIX,
        help='Output filename prefix.')
    parser.add_argument(
        '--formats',
        nargs='+',
        default=['svg', 'pdf'],
        choices=['svg', 'pdf', 'png', 'eps'],
        help='Figure formats to save.')
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for raster outputs.')
    return parser.parse_args()


def read_rows(path):
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {'iter': int(float(row['iter']))}
            for key, _, _color, _offset in CURVES:
                parsed[key] = float(row[key])
            rows.append(parsed)
    if not rows:
        raise RuntimeError(f'No rows found in {path}')
    return rows


def copy_source_csv(src, out_dir, prefix):
    dst = out_dir / f'{prefix}_data.csv'
    shutil.copyfile(src, dst)
    return dst


def annotate_selected_points(ax, xs, ys, color, offset, important_xs):
    for x, y in zip(xs, ys):
        if x not in important_xs:
            continue
        ax.text(
            x,
            y + offset,
            f'{y:.4f}',
            ha='center',
            va='bottom' if offset >= 0 else 'top',
            fontsize=7,
            color=color,
            fontweight='bold' if x in (6000, 20000) else 'normal')


def plot_curves(rows, out_dir, prefix, formats, dpi):
    xs = [row['iter'] for row in rows]
    max_iter = max(xs)
    important_xs = {1000, 3000, 6000, 9000, 12000, max_iter}

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'dejavuserif',
        'axes.unicode_minus': False,
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    for key, label, color, offset in CURVES:
        ys = [row[key] for row in rows]
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=1.7,
            marker='o',
            markersize=3.0,
            markerfacecolor='white',
            markeredgewidth=0.9,
            label=label)
        annotate_selected_points(ax, xs, ys, color, offset, important_xs)

    max_value = max(max(row[key] for row in rows) for key, *_ in CURVES)
    y_top = max(0.94, math.ceil((max_value + 0.08) * 10.0) / 10.0)
    ax.set_xlim(min(xs) - 200, max(xs) + 200)
    ax.set_ylim(-0.02, y_top)
    ax.set_xlabel('Training Iteration (x1000)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x // 1000) for x in xs], fontsize=8)
    ax.set_yticks([i / 10.0 for i in range(0, int(y_top * 10) + 1)])
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(axis='y', color='#e5e5e5', linewidth=0.8)
    ax.grid(axis='x', color='#f0f0f0', linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#cfcfcf')
        spine.set_linewidth(0.8)

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=9,
        handlelength=2.4,
        columnspacing=1.4)

    fig.tight_layout(rect=[0.035, 0.12, 0.995, 0.98])
    written = []
    for fmt in formats:
        path = out_dir / f'{prefix}.{fmt}'
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        written.append(path)
    plt.close(fig)
    return written


def annotate_panel_points(ax, xs, ys, color, offset, important_xs):
    for x, y in zip(xs, ys):
        if x not in important_xs:
            continue
        ax.text(
            x,
            y + offset,
            f'{y:.4f}',
            ha='center',
            va='bottom' if offset >= 0 else 'top',
            fontsize=6.5,
            color=color,
            fontweight='bold' if x in (6000, max(important_xs)) else 'normal')


def style_panel(ax, xs, ylabel, ylim):
    ax.set_xlim(min(xs) - 200, max(xs) + 200)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=9, fontweight='bold')
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x // 1000) for x in xs], fontsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(axis='y', color='#e5e5e5', linewidth=0.8)
    ax.grid(axis='x', color='#f0f0f0', linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#cfcfcf')
        spine.set_linewidth(0.8)


def plot_two_panel_curves(rows, out_dir, prefix, formats, dpi):
    xs = [row['iter'] for row in rows]
    max_iter = max(xs)
    important_xs = {1000, 3000, 6000, 10000, 14000, max_iter}
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)

    panel_a = (
        ('teacher_confidence', 'Teacher confidence', '#3b6fd8', 0.018),
    )
    panel_b = (
        ('mparc_pseudo_weight', 'Reliability-weighted pseudo supervision',
         '#f0a500', 0.018),
        ('proto_target_update_ratio', 'Prototype update ratio', '#d95f02',
         -0.018),
        ('boundary_weight_x100', 'Boundary refinement weight (x100)',
         '#5b5b5b', -0.018),
    )

    for ax, panel, tag in zip(axes, (panel_a, panel_b), ('(a)', '(b)')):
        for key, label, color, offset in panel:
            ys = [row[key] for row in rows]
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=1.7,
                marker='o',
                markersize=3.0,
                markerfacecolor='white',
                markeredgewidth=0.9,
                label=label)
            annotate_panel_points(ax, xs, ys, color, offset, important_xs)
        ax.text(
            0.01,
            0.94,
            tag,
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=10,
            fontweight='bold')
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.12),
            ncol=len(panel),
            frameon=False,
            fontsize=8,
            handlelength=2.2,
            columnspacing=1.1)

    style_panel(axes[0], xs, 'Supervision reliability', (-0.02, 0.94))
    style_panel(axes[1], xs, 'Weighted branch participation', (-0.02, 0.31))
    axes[1].set_xlabel('Training Iteration (x1000)',
                       fontsize=10,
                       fontweight='bold')
    fig.tight_layout(rect=[0.035, 0.055, 0.995, 0.99], h_pad=2.1)

    written = []
    for fmt in formats:
        path = out_dir / f'{prefix}_two_panel.{fmt}'
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        written.append(path)
    plt.close(fig)
    return written


def main():
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_path)
    data_path = copy_source_csv(input_path, out_dir, args.fig_prefix)
    figure_paths = plot_curves(rows, out_dir, args.fig_prefix, args.formats,
                               args.dpi)
    two_panel_paths = plot_two_panel_curves(rows, out_dir, args.fig_prefix,
                                            args.formats, args.dpi)

    print(f'Copied data CSV to {data_path}')
    for path in figure_paths:
        print(f'Wrote figure {path}')
    for path in two_panel_paths:
        print(f'Wrote two-panel figure {path}')


if __name__ == '__main__':
    main()

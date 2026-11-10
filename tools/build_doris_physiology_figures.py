"""Two physiology evidence plates: source-only VI distributions and fixed-probe evolution."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmseg.vi_utils import normalize_goji_index, prepare_goji_index
from tools.build_doris_mechanism_figures import plt, panel, read_node, save_figure

CLASSES = ('Goji', 'Vegetation', 'Building', 'Water', 'Road')
INDICES = ('ndvi', 'ndre', 'cire')


def geotags(image):
    return {key: repr(image.tag_v2[key]) for key in (33550, 33922, 34264, 34735, 34736, 34737)
            if hasattr(image, 'tag_v2') and key in image.tag_v2}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_csv(path, rows):
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    path = Path(path)
    content = text.getvalue()
    if path.exists():
        if path.read_text(encoding='utf-8') != content.replace('\r\n', '\n'):
            raise ValueError('Existing physiology data differ: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        stream.write(content)
    temporary.replace(path)


def image_group(name, group_regex=None):
    # Crop filenames with four terminal coordinates preserve the complete parent ID.
    match = re.match(group_regex or r'^(.*?)(?:[_-]\d+){4}$', Path(name).stem)
    return match.group(1) if match else Path(name).stem


def fisher(x, y):
    """Common multivariate Fisher trace ratio; singleton subsets equal the 1-D score."""
    if len(x) < 2 or len(y) < 2:
        return ''
    numerator = float(np.square(x.mean(0) - y.mean(0)).sum())
    denominator = float((x.var(0, ddof=1) + y.var(0, ddof=1)).sum())
    return numerator / denominator if denominator > 0 else ''


def analyze_vi(source, output, source_root=None, seed=0, max_tiles=128,
               per_tile_class=256, per_group_class=2048, group_regex=None, make_figure=True):
    """One deterministic source-train analysis, cached by inputs and sampling protocol."""
    if source.get('type') != 'GojiDataset' or not source.get('use_vi', False):
        raise ValueError('VI evidence requires the configured source GojiDataset with use_vi=True')
    if min(max_tiles, per_tile_class, per_group_class) <= 0:
        raise ValueError('Sampling limits must be positive')
    if tuple(source.get('vi_names', source.get('index_names', INDICES))) != INDICES:
        raise ValueError('Physiology evidence requires the actual NDVI/NDRE/CIRE order')
    root = Path(source_root or source['data_root'])
    img_dir, ann_dir = root / source['img_dir'], root / source['ann_dir']
    if Path(source['img_dir']).name != 'train' or Path(source['ann_dir']).name != 'train':
        raise ValueError('VI evidence only accepts source training images and annotations')
    files = sorted(img_dir.rglob('*.tif'))
    if source.get('split'):
        allowed = {line.strip() + '.tif' for line in (root / source['split']).read_text().splitlines()}
        files = [p for p in files if p.relative_to(img_dir).as_posix() in allowed]
    if not files:
        raise ValueError('No source training TIFFs found: ' + str(img_dir))
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(files), min(len(files), max_tiles), replace=False))
    files = [files[i] for i in chosen]
    suffix = source.get('index_suffix', '_dir')
    selected = []
    for image in files:
        relative = image.relative_to(img_dir)
        paths = [image, ann_dir / relative] + [
            root / (name + suffix) / 'train' / relative for name in INDICES]
        for p in paths:
            if not p.is_file(): raise ValueError('Missing spatial counterpart: ' + str(p))
        selected.append((relative, paths))
    protocol = dict(source=source, source_root=str(root.resolve()), seed=seed, max_tiles=max_tiles,
        per_tile_class=per_tile_class, per_group_class=per_group_class, group_regex=group_regex,
        normalization='same normalized vi_img as training; float32, clipping, NaN->0',
        version=1, tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        normalization_sha256=hashlib.sha256((ROOT/'mmseg/vi_utils.py').read_bytes()).hexdigest(),
        files=[(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                          for _, paths in selected for p in paths])
    identity = digest(protocol)
    output = Path(output)
    stats_path, separation_path = output / 'data/phy_vi_statistics.csv', output / 'data/phy_vi_separability.csv'
    png, pdf = output / 'figures/fig_phy_vi_distributions.png', output / 'figures/fig_phy_vi_distributions.pdf'
    required = (stats_path, separation_path, png, pdf) if make_figure else (stats_path, separation_path)
    if all(p.exists() for p in required):
        with stats_path.open(encoding='utf-8', newline='') as stream:
            first = next(csv.DictReader(stream))
        if first['analysis_identity'] != identity:
            raise ValueError('Existing source VI analysis uses different data/protocol; preserve it')
        if make_figure:
            with Image.open(png) as image:
                if image.info.get('Description') != identity: raise ValueError('VI figure provenance mismatch')
                image.verify()
        return output
    buckets, raw_counts, georef_verified = {}, {}, True
    for relative, paths in selected:
        with Image.open(paths[0]) as image:
            shape = (image.height, image.width)
            reference = geotags(image)
            complete_reference = 34735 in reference and (
                34264 in reference or (33550 in reference and 33922 in reference))
        with Image.open(paths[1]) as image:
            raw_label = np.asarray(image).copy()
            geometry = geotags(image)
        georef_verified &= complete_reference and geometry == reference
        if reference and geometry and geometry != reference:
            raise ValueError('Source GT/image georeference mismatch: ' + str(relative))
        if raw_label.ndim != 2 or raw_label.shape != shape:
            raise ValueError('Source GT/image shape mismatch: ' + str(relative))
        if not np.isin(raw_label, (0, 1, 2, 3, 4, 5, 255)).all():
            raise ValueError('Unexpected raw Goji label IDs: ' + str(relative))
        # GojiDataset.reduce_zero_label=True: raw 1..5 -> 0..4, 0/255 ignored.
        label = np.where((raw_label >= 1) & (raw_label <= 5), raw_label.astype(np.int64)-1, 255)
        indices = []
        for name, p in zip(INDICES, paths[2:]):
            with Image.open(p) as image:
                values = prepare_goji_index(np.asarray(image))
                geometry = geotags(image)
            georef_verified &= complete_reference and geometry == reference
            if reference and geometry and geometry != reference:
                raise ValueError('Source VI/image georeference mismatch: ' + str(p))
            if values.shape != shape:
                raise ValueError('Source VI/GT shape mismatch: ' + str(p))
            indices.append(normalize_goji_index(values, name, source.get('vi_norm_cfg')))
        values = np.stack(indices, axis=-1).reshape(-1, 3)
        group = image_group(str(relative), group_regex)
        for cls in range(5):
            candidate = np.flatnonzero(label.reshape(-1) == cls)
            raw_counts[(group, cls)] = raw_counts.get((group, cls), 0) + len(candidate)
            if len(candidate):
                positions = rng.choice(candidate, min(len(candidate), per_tile_class), replace=False)
                buckets.setdefault((group, cls), []).append(values[positions])
    groups = {key: np.concatenate(chunks) for key, chunks in buckets.items()}
    samples, rows = {}, []
    for cls, class_name in enumerate(CLASSES):
        available = [(group, values) for (group, c), values in groups.items() if c == cls]
        # Equal counts from each nonempty image group prevent large scenes dominating.
        n = min([len(v) for _, v in available] + [per_group_class]) if available else 0
        portions = []
        for group, values in sorted(available):
            picked = values[rng.choice(len(values), n, replace=False)]
            portions.append(picked)
            for index, name in enumerate(INDICES):
                rows.append(stat_row(identity, cls, class_name, group, name, picked[:, index],
                    raw_counts[(group, cls)], len(selected), seed))
        samples[cls] = np.concatenate(portions) if portions else np.empty((0, 3))
        for index, name in enumerate(INDICES):
            rows.append(stat_row(identity, cls, class_name, 'ALL_BALANCED_GROUPS', name,
                samples[cls][:, index], sum(v for (g, c), v in raw_counts.items() if c == cls),
                len(selected), seed))
    separation = []
    for subset in ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)):
        separation.append(dict(analysis_identity=identity, indices='+'.join(INDICES[i] for i in subset),
            fisher_trace_ratio=fisher(samples[0][:, subset], samples[1][:, subset]),
            n_goji=len(samples[0]), n_vegetation=len(samples[1]),
            definition='squared mean-distance / trace(sum of within-class sample covariance)',
            normalization='training vi_img [0,1]', seed=seed))
    for row in rows:
        normalizations = source.get('vi_norm_cfg') or {}
        norm = normalizations.get(row['index'], normalizations)
        row.update(raw_clip_min=norm.get('min', -1), raw_clip_max=norm.get('max', 1),
            source_root=str(root.resolve()), per_tile_class=per_tile_class,
            per_group_class=per_group_class, max_tiles=max_tiles,
            grouping=group_regex or 'strip four terminal coordinates; otherwise patch filename',
            alignment_check='filename+shape+matching TIFF geotags' if georef_verified else
                            'filename+shape; missing geotags, spatial registration not proven')
    write_csv(stats_path, rows); write_csv(separation_path, separation)
    if not make_figure:
        return output
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), constrained_layout=True)
    for index, (axis, name) in enumerate(zip(axes, INDICES)):
        present = [cls for cls in range(5) if len(samples[cls])]
        if present:
            axis.boxplot([samples[c][:, index] for c in present], positions=present,
                         showfliers=False, widths=.55, patch_artist=True,
                         boxprops=dict(facecolor='#a5bdca'), medianprops=dict(color='#253e4a'))
        axis.set_xticks(range(5)); axis.set_xticklabels(CLASSES, rotation=35, ha='right')
        normalizations = source.get('vi_norm_cfg') or {}
        norm = normalizations.get(name, normalizations)
        axis.set_ylim(0, 1)
        axis.set_title('{}\nraw clip [{}, {}]'.format(name.upper(), norm.get('min', -1), norm.get('max', 1)))
        axis.set_ylabel('Normalized VI (training definition)' if index == 0 else '')
        for cls in range(5):
            if not len(samples[cls]): axis.text(cls, .5, 'No data', rotation=90, ha='center')
    fig.suptitle('Source training VI distributions | group-balanced pixel samples' +
                (' | SYNTHETIC TEST DATA' if source.get('synthetic_fixture') else ''), fontsize=9)
    fig.supxlabel('Boxes: Q1–Q3; median line; whiskers: 1.5 IQR; outliers hidden', fontsize=6)
    try:
        png.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, png, identity); save_figure(fig, pdf, identity)
    finally:
        plt.close(fig)
    return output


def stat_row(identity, cls, class_name, group, index, values, available, n_tiles, seed):
    n = len(values)
    return dict(analysis_identity=identity, class_id=cls, class_name=class_name,
        image_group=group, index=index, count=n, available_pixels=available,
        mean=float(values.mean()) if n else '', std=float(values.std(ddof=1)) if n > 1 else '',
        median=float(np.median(values)) if n else '', Q1=float(np.quantile(values, .25)) if n else '',
        Q3=float(np.quantile(values, .75)) if n else '', source_train_tiles=n_tiles, seed=seed,
        normalization='training vi_img [0,1]', quantiles='sampled pixels; group-balanced within class')


def build_phy_response_figure(analysis):
    analysis = Path(analysis)
    paths = sorted((analysis / 'physiology/arrays').glob('iter_*_phy.npz'))
    if not paths: raise ValueError('No saved physiology nodes; missing nodes are never synthesized')
    nodes = [read_node(p) for p in paths]
    identities = {(n['description']['config_sha256'], n['description']['sample_tensor_sha256'],
                   n['description']['sample_id']) for n in nodes}
    if len(identities) != 1: raise ValueError('Physiology nodes differ in sample or config')
    if any(n['description']['data_source'] != 'fixed_probe_recomputed' for n in nodes):
        raise ValueError('PHY-2 must use fixed-probe recomputed data')
    manifest = json.loads((analysis / 'manifest.json').read_text(encoding='utf-8'))
    start = manifest['phy_warmup_start']; end = start + manifest['phy_rampup_iters']
    steps = [int(n['description']['iter']) for n in nodes]
    requested = [min(2000, start), start, end, int(.75*steps[-1]), steps[-1]]
    selected = sorted(set(min(range(len(nodes)), key=lambda i: abs(steps[i]-r)) for r in requested))
    stamp = json.dumps(dict(requested=requested, selected=[steps[i] for i in selected],
        sample_id=nodes[0]['description']['sample_id'], source_sha=digest([
            hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]),
        plot_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        synthetic_fixture=nodes[0]['description'].get('synthetic_fixture', False)), sort_keys=True)
    folder = analysis / 'physiology/figures'
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(selected), 5, squeeze=False,
                             figsize=(7.2, 1.4*len(selected)+.6), constrained_layout=True)
    for row, i in enumerate(selected):
        maps = dict(nodes[i]); maps['w_base'] = maps['w_before_phy']
        for col, key in enumerate(('target_image', 'r_phy', 'w_base', 'w_t_phy', 'delta_w_phy')):
            panel(axes[row, col], maps, key)
        axes[row, 2].set_title(r'$W_{\mathrm{before\ phy}}$')
        if row: axes[row, 0].set_visible(False)  # Same static input shown only once.
        axes[row, 1].set_ylabel('Iter {:06d}'.format(steps[i]))
        mask = maps['m_phy'].astype(bool) & maps['valid_mask'].astype(bool)
        if mask.any() and not mask.all(): axes[row, 1].contour(mask, levels=[.5], colors='white', linewidths=.35)
    fig.colorbar(axes[0, 1].images[0], ax=list(axes[:, 1:4].flat), shrink=.3, ticks=[0,.5,1])
    fig.colorbar(axes[0, 4].images[0], ax=list(axes[:, 4].flat), shrink=.3, ticks=[-1,0,1])
    title = 'Vegetation-related physiology and pseudo-weight modulation'
    if nodes[0]['description'].get('synthetic_fixture'): title += ' | SYNTHETIC TEST DATA'
    fig.suptitle(title, fontsize=9)
    fig.supxlabel('White contour: Teacher vegetation mask; fixed scales; selected actual nodes ' +
                  ', '.join(str(steps[i]) for i in selected), fontsize=6)
    try:
        for suffix in ('png','pdf'):
            save_figure(fig, folder / ('fig_phy_response_evolution.'+suffix), stamp)
    finally:
        plt.close(fig)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--work-dir')
    parser.add_argument('--source-root', help='Explicit local mirror of source data_root; source train only')
    parser.add_argument('--only', choices=('all','vi','response'), default='all')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-tiles', type=int, default=128)
    parser.add_argument('--per-tile-class', type=int, default=256)
    parser.add_argument('--per-group-class', type=int, default=2048)
    parser.add_argument('--group-regex', help='First capture group identifies parent image; default removes four coordinates')
    args = parser.parse_args()
    from mmcv import Config
    cfg = Config.fromfile(args.config, import_custom_modules=False)
    analysis = Path(args.work_dir or cfg.work_dir) / 'analysis'
    if args.only in ('all','vi'):
        analyze_vi(cfg.data.train.source, analysis / 'physiology', args.source_root,
                   args.seed, args.max_tiles, args.per_tile_class, args.per_group_class, args.group_regex)
    if args.only in ('all','response'): build_phy_response_figure(analysis)
    print('Physiology evidence:', analysis / 'physiology')


if __name__ == '__main__':
    main()

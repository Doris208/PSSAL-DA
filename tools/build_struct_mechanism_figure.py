"""Render immutable Struct replay arrays and descriptive per-class statistics."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_hphy_mechanism_figure import (
    CROP, GRAY, PALETTE, crop_arrays, file_sha256, roi_coordinates, save_figure, synthetic_label)
from tools.doris_phy_evidence import error_auc

COMPONENTS = ('image_rgb', 'R_conf', 'R_loc', 'R_adj', 'R_prob', 'R_struct', 'struct_prediction')
COMPARISON = ('image_rgb', 'baseline_prediction', 'struct_prediction', 'R_struct', 'A_struct')
METRICS = ('R_conf', 'R_loc', 'R_adj', 'R_prob', 'R_struct', 'G_struct', 'A_struct', 'W_before', 'W_after', 'delta_W')
TITLES = dict(image_rgb='Target RGB', R_conf='Teacher confidence', R_loc='Local continuity',
    R_adj='Adjacency agreement', R_prob='Semantic agreement', R_struct='Structural response',
    G_struct='Structural gate', A_struct='Relative attenuation', W_before='Weight before gate',
    W_after='Weight after gate', delta_W='Weight change', baseline_prediction='Baseline',
    struct_prediction='Structural\nStudent\nprediction', teacher_pseudo_label='Teacher pseudo-label',
    teacher_confidence='Teacher confidence', teacher_top2_margin='Teacher top-two margin',
    adjacency_available='Adjacency prior available', semantic_available='Semantic prior available')


def output_path(path):
    return Path(path).resolve()


def applicability(arrays, key):
    valid = arrays['valid_mask'].astype(bool)
    if key in ('R_adj', 'R_prob'):
        return valid & arrays['adjacency_available' if key == 'R_adj' else 'semantic_available'].astype(bool)
    if key == 'R_struct':
        return arrays['teacher_struct_mask'].astype(bool)
    if key in ('G_struct', 'A_struct'):
        return arrays['positive_struct_weight_mask'].astype(bool)
    return valid


def validate_arrays(arrays, meta, tolerance=1e-5):
    required = {'image_rgb', 'teacher_pseudo_label', 'teacher_confidence', 'baseline_prediction',
        'struct_prediction', 'valid_mask', 'adjacency_available', 'semantic_available',
        'teacher_struct_mask', 'positive_base_weight_mask', 'positive_struct_weight_mask',
        'gate_active_mask', 'below_threshold_mask', *METRICS}
    if required - set(arrays):
        raise ValueError('Missing structural arrays: ' + ', '.join(sorted(required-set(arrays))))
    if (meta.get('direction') != 'goji26RGB2goji30RGB' or
            meta.get('teacher_network') != 'Struct EMA Teacher' or meta.get('prediction_network') != 'Student' or
            meta.get('class_palette') != PALETTE.tolist()):
        raise ValueError('Structural direction, network identity or palette differs from protocol')
    for key, value in arrays.items():
        if value.shape[:2] != (256, 256) or not np.isfinite(value).all():
            raise ValueError('Invalid spatial grid or non-finite values: ' + key)
        if key not in ('image_rgb', 'image_6ch', 'target_rgb') and value.ndim != 2:
            raise ValueError('Expected a scalar spatial map: ' + key)
    if arrays['image_rgb'].shape != (256, 256, 3) or arrays['image_rgb'].dtype != np.uint8:
        raise ValueError('Expected original uint8 RGB pixels')
    for key in ('teacher_pseudo_label', 'baseline_prediction', 'struct_prediction', 'gt_label'):
        if key in arrays and not np.isin(arrays[key], [0, 1, 2, 3, 4, 255]).all():
            raise ValueError('Invalid class ids: ' + key)
    for key in METRICS[:-1] + ('teacher_confidence',):
        if arrays[key].min() < -tolerance or arrays[key].max() > 1+tolerance:
            raise ValueError('Response or weight outside [0,1]: ' + key)
    gate = meta['gate']
    valid = arrays['valid_mask'].astype(bool)
    scope = valid & np.isin(arrays['teacher_pseudo_label'], gate['struct_positive_classes'])
    positive = valid & (arrays['W_before'] > 0)
    masks = dict(teacher_struct_mask=scope, positive_base_weight_mask=positive,
        positive_struct_weight_mask=positive & scope,
        gate_active_mask=positive & scope & (arrays['G_struct'] < 1),
        below_threshold_mask=scope & (arrays['R_struct'] < gate['struct_filter_threshold']))
    for key, expected in masks.items():
        if not np.array_equal(arrays[key], expected):
            raise ValueError('Incorrect structural mask: ' + key)
    if gate['fusion_mode'] != 'doris_confidence_guided':
        raise ValueError('Expected confidence-guided structural fusion')
    numerator, denominator = np.zeros((256, 256)), np.zeros((256, 256))
    for coefficient, key, available in (
            (gate['continuity_weight'], 'R_loc', np.ones((256, 256), bool)),
            (gate['adjacency_weight'], 'R_adj', arrays['adjacency_available']),
            (gate['semantic_weight'], 'R_prob', arrays['semantic_available'])):
        numerator += coefficient * arrays[key] * available
        denominator += coefficient * available
    expected_response = arrays['R_conf'] * numerator / np.maximum(denominator, gate['eps'])
    probability_dtype = meta.get('checkpoint_struct', {}).get('struct_probability_dtype', 'float32')
    if probability_dtype not in ('float16', 'float32'):
        raise ValueError('Unsupported structural probability computation dtype')
    response_tolerance = max(tolerance, float(np.finfo(probability_dtype).eps))
    if not np.allclose(arrays['R_struct'], expected_response, atol=response_tolerance, rtol=0):
        raise ValueError('Structural fusion identity failed')
    raw_gate = np.maximum(arrays['R_struct'], gate['struct_min_weight'])
    below = arrays['R_struct'] < gate['struct_filter_threshold']
    raw_gate = np.where(below, 0 if gate['struct_filter_mode'] == 'hard' else raw_gate*.5, raw_gate)
    raw_gate = np.where(np.isin(arrays['teacher_pseudo_label'], gate['struct_positive_classes']), raw_gate, 1)
    ramp = gate['ramp'] if gate['use_target_gate'] else 0.
    expected_gate = (1-ramp) + ramp*raw_gate
    identities = ((arrays['R_conf'], arrays['teacher_confidence']),
        (arrays['G_struct'], expected_gate),
        (arrays['A_struct'][positive & scope], 1-arrays['G_struct'][positive & scope]),
        (arrays['W_after'], arrays['W_before']*arrays['G_struct']),
        (arrays['delta_W'], arrays['W_after']-arrays['W_before']))
    for actual, expected in identities:
        if not np.allclose(actual, expected, atol=tolerance, rtol=0):
            raise ValueError('Structural response/gate/weight identity failed')
    if (arrays['W_after'] > arrays['W_before']+tolerance).any() or (arrays['W_before'][~valid] != 0).any():
        raise ValueError('Gate increases weights or invalid pixels carry supervision')


def summary(values):
    if not values.size:
        return dict(count=0, mean=None, std=None, q05=None, q25=None, median=None, q75=None, q95=None)
    quantiles = np.quantile(values, [.05, .25, .5, .75, .95]).tolist()
    return dict(count=int(values.size), mean=float(values.mean(dtype=np.float64)),
                std=float(values.std(dtype=np.float64)), **dict(zip(('q05', 'q25', 'median', 'q75', 'q95'), quantiles)))


def descriptive_statistics(arrays, meta):
    """Center-crop pixel summaries with explicit prior and supervision availability."""
    center = crop_arrays(arrays)
    valid = center['valid_mask'].astype(bool)
    teacher = center['teacher_pseudo_label']
    scope = center['teacher_struct_mask'].astype(bool)
    positive = center['positive_struct_weight_mask'].astype(bool)
    groups = [('all_structural', scope)]
    groups += [('teacher_' + name, valid & (teacher == cls))
               for cls, name in enumerate(meta['class_names']) if cls in meta['gate']['struct_positive_classes']]
    if 'gt_label' in center:
        known = valid & (center['gt_label'] != 255)
        for cls in meta['gate']['struct_positive_classes']:
            for correct in (True, False):
                mask = known & (teacher == cls) & ((teacher == center['gt_label']) == correct)
                groups.append(('teacher_{}_{}'.format(meta['class_names'][cls], 'correct' if correct else 'error'), mask))
    rows = []
    for group, mask in groups:
        for key in METRICS:
            rows.append(dict(group=group, metric=key, **summary(center[key][mask & applicability(center, key)])))
    effective = positive & (center['G_struct'] < 1)
    stats = dict(scope='center196', crop=list(CROP), unit='pixels within one replayed image',
        valid_pixels=int(valid.sum()), structural_pixels=int(scope.sum()),
        positive_struct_weight_pixels=int(positive.sum()), gate_active_pixels=int(effective.sum()),
        gate_active_fraction=float(effective.sum()/positive.sum()) if positive.any() else None,
        below_threshold_pixels=int(center['below_threshold_mask'].sum()),
        weight_mass_before=float(center['W_before'][scope].sum(dtype=np.float64)),
        weight_mass_after=float(center['W_after'][scope].sum(dtype=np.float64)),
        mean_attenuation=summary(center['A_struct'][positive])['mean'],
        gate=meta['gate'], rows=rows, gt_available='gt_label' in center)
    if 'gt_label' in center:
        eligible = positive & (center['gt_label'] != 255)
        errors = teacher[eligible] != center['gt_label'][eligible]
        stats['teacher_errors'] = int(errors.sum())
        stats['high_confidence_teacher_errors'] = int((errors &
            (center['teacher_confidence'][eligible] >= meta['pseudo_threshold'])).sum())
        stats['error_detection'] = {key: error_auc(errors, score[eligible]) for key, score in (
            ('one_minus_R_struct', 1-center['R_struct']), ('A_struct', center['A_struct']),
            ('teacher_uncertainty', 1-center['teacher_confidence']))}
        stats['prediction_iou'] = {}
        known = valid & (center['gt_label'] != 255)
        for key in ('baseline_prediction', 'struct_prediction'):
            per_class = []
            for cls in range(len(meta['class_names'])):
                truth, predicted = center['gt_label'] == cls, center[key] == cls
                intersection = int((known & truth & predicted).sum())
                union = int((known & (truth | predicted)).sum())
                per_class.append(dict(class_id=cls, class_name=meta['class_names'][cls],
                    intersection=intersection, union=union, iou=intersection/union if union else None))
            stats['prediction_iou'][key] = per_class
    return stats


def display_spec(key, meta):
    if key == 'image_rgb' or key.endswith('prediction') or key in ('teacher_pseudo_label', 'gt_label'):
        return None
    low, high = (-1., 0.) if key == 'delta_W' else (0., 1.)
    if key in ('ndvi', 'ndre', 'cire'):
        low, high = (meta['vi_normalization'][key][bound] for bound in ('min', 'max'))
    return dict(vmin=low, vmax=high, cmap='magma' if key == 'A_struct' else
                'RdBu_r' if key == 'delta_W' else 'viridis')


def colored_pixels(arrays, key, meta):
    value = arrays[key]
    if key == 'image_rgb':
        return value.copy()
    spec = display_spec(key, meta)
    if spec is None:
        pixels = np.full((*value.shape, 3), GRAY, dtype=np.uint8)
        for cls, color in enumerate(PALETTE):
            pixels[value == cls] = color
        return pixels
    pixels = plt.get_cmap(spec['cmap'])(Normalize(spec['vmin'], spec['vmax'])(value), bytes=True)[..., :3]
    pixels[~applicability(arrays, key)] = GRAY
    return pixels


def draw_panel(axis, arrays, key, meta, font_size):
    axis.imshow(colored_pixels(arrays, key, meta), interpolation='nearest')
    axis.set_axis_off()
    axis.set_title(TITLES.get(key, key), fontsize=font_size, pad=6)


def add_bar(fig, bounds, key, meta, font_size=7):
    spec = display_spec(key, meta)
    axis = fig.add_axes(bounds)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(spec['vmin'], spec['vmax']), cmap=spec['cmap']),
                 cax=axis, orientation='horizontal', ticks=[spec['vmin'], (spec['vmin']+spec['vmax'])/2, spec['vmax']])
    axis.tick_params(labelsize=font_size, length=2, pad=2)


def group_figure(arrays, meta, keys=COMPONENTS, roi=None, font_size=9):
    if tuple(keys) == COMPONENTS:
        from tools.doris_struct_evidence import prepare_struct_evidence
        from tools.struct_mechanism_panels import display_scales, panel_figure
        prepared, info = prepare_struct_evidence(arrays, meta)
        center = crop_arrays(prepared)
        return panel_figure(center, keys, 'formal', display_scales(center, info, 'formal'),
                            info, font_size, roi_coordinates(roi))
    center = crop_arrays(arrays)
    left, gap = .12, .08
    side = (7.2-2*left-(len(keys)-1)*gap)/len(keys)
    height = side+1.35
    fig = plt.figure(figsize=(7.2, height))
    for index, key in enumerate(keys):
        x = left + index*(side+gap)
        axis = fig.add_axes([x/7.2, .77/height, side/7.2, side/height])
        draw_panel(axis, center, key, meta, font_size)
        axis.set_title('({}) {}'.format(chr(97+index), TITLES.get(key, key).replace(' ', '\n', 1)),
                       fontsize=font_size, pad=6)
        if display_spec(key, meta) is not None:
            add_bar(fig, [(x+.08)/7.2, .55/height, (side-.16)/7.2, .05/height], key, meta)
        if roi is not None and key.endswith('prediction'):
            x1, y1, x2, y2 = roi_coordinates(roi)
            axis.add_patch(Rectangle((x1-.5, y1-.5), x2-x1, y2-y1, fill=False, edgecolor='white', linewidth=1))
    handles = [Patch(color=color/255., label=name.title()) for name, color in zip(meta['class_names'], PALETTE)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .08), ncol=5,
               frameon=False, fontsize=6.5, handlelength=.8, columnspacing=1.)
    synthetic_label(fig, meta)
    return fig


def export_panels(arrays, meta, folder, dpi, font_size, roi=None):
    folder.mkdir(parents=True)
    (folder / 'clean').mkdir()
    (folder / 'labeled').mkdir()
    for key in arrays:
        if key == 'image_6ch':
            continue
        pixels = colored_pixels(arrays, key, meta)
        description = 'SYNTHETIC TEST DATA' if meta.get('synthetic_fixture') else 'Struct offline response'
        for destination in (folder, folder / 'clean'):
            plt.imsave(str(destination / (key+'.png')), pixels, dpi=dpi, metadata={'Description': description})
        fig = plt.figure(figsize=(2.05, 2.4))
        axis = fig.add_axes([.1, .22, .8, .8*2.05/2.4])
        draw_panel(axis, arrays, key, meta, font_size)
        if display_spec(key, meta) is not None:
            add_bar(fig, [.16, .12, .68, .035], key, meta)
        synthetic_label(fig, meta)
        save_figure(fig, folder / 'labeled' / (key+'.png'), dpi)
    if roi is not None:
        x1, y1, x2, y2 = roi
        fig, axes = plt.subplots(1, 2, figsize=(3.3, 2.1))
        for axis, key in zip(axes, ('baseline_prediction', 'struct_prediction')):
            axis.imshow(colored_pixels(arrays, key, meta)[y1:y2, x1:x2], interpolation='nearest')
            axis.set_axis_off()
            axis.set_title(TITLES[key], fontsize=font_size)
        synthetic_label(fig, meta)
        save_figure(fig, folder / 'baseline_vs_struct_roi.png', dpi)


def load_arrays(input_npz, meta_json):
    meta = json.loads(Path(meta_json).read_text(encoding='utf-8'))
    if meta['arrays_sha256']['responses_full_256.npz'] != file_sha256(input_npz):
        raise ValueError('NPZ hash differs from inference provenance')
    with np.load(input_npz, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    validate_arrays(arrays, meta)
    return arrays, meta


def render(input_npz, meta_json, output_dir, roi=None, dpi=600, font_size=9):
    from tools.struct_mechanism_panels import render_struct_bundle
    arrays, meta = load_arrays(input_npz, meta_json)
    before = file_sha256(input_npz), file_sha256(meta_json)
    out = render_struct_bundle(arrays, Path(output_dir)/'struct_v4', meta, dpi=dpi,
        font_size=font_size, source_sha256=before[0], roi=roi)
    if before != (file_sha256(input_npz), file_sha256(meta_json)):
        raise RuntimeError('Inference arrays or metadata changed during rendering')
    return out


def export_online_views(input_npz, meta_json, output_dir, dpi=600, font_size=9):
    from tools.build_doris_mechanism_figures import render_online_node
    arrays, meta = load_arrays(input_npz, meta_json)
    aliases = dict(w_conf='R_conf', pseudo_label='teacher_pseudo_label', teacher_label='teacher_pseudo_label',
        r_conf='R_conf', r_loc='R_loc', r_adj='R_adj', r_prob='R_prob', r_struct='R_struct',
        w_base='W_before', w_before_struct='W_before', w_after_struct='W_after', w_t_ps='W_after',
        w_final='W_after', delta_w_ps='delta_W', valid_mask='valid_mask',
        adjacency_available='adjacency_available', semantic_available='semantic_available')
    maps = dict(arrays, **{key: arrays[source] for key, source in aliases.items()})
    completed = meta['checkpoint_struct']['completed_iter']
    info = dict(meta, iter=completed, rule_iter=meta['gate']['rule_iter'], capture_mode='checkpoint_replay',
        data_source='fixed_probe_recomputed', output_mode='branches', input_view=['R', 'G', 'B'],
        struct_cfg=meta['gate'], modules=dict(phy=False, struct=True, proto=False, edge=False),
        states=dict(phy='disabled', struct=meta['gate']['state'], proto='disabled', edge='disabled'),
        state_sha256=meta['checkpoint_struct']['state_sha256_before'],
        config_sha256=meta['config_struct']['resolved_sha256'], source_responses_sha256=file_sha256(input_npz))
    maps.update(target_image=arrays['image_rgb'].astype(np.float32)/255,
        rule_iter=np.array(info['rule_iter']), proto_active=np.array(False), metadata=np.array(json.dumps(info, sort_keys=True)))
    folder = output_path(output_dir)/'fixed_probe'
    path = folder/'selected_arrays'/('iter_{:06d}.npz'.format(completed))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **maps)
    return render_online_node(maps, folder/'snapshots', completed, array_sha256=file_sha256(path),
        render_timing='checkpoint_replay', dpi=dpi, font_size=font_size)


def main():
    import runpy
    from tools.struct_mechanism_panels import render_struct_bundle, write_run_summary
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-npz', required=True, nargs='+', type=Path)
    parser.add_argument('--meta-json', type=Path)
    parser.add_argument('--config', type=Path, help='Saved resolved training config for old NPZs without struct_cfg')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--roi', type=int, nargs=4)
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--font-size', type=float, default=9)
    args = parser.parse_args()
    if args.meta_json and len(args.input_npz) != 1:
        parser.error('--meta-json requires one input NPZ')
    cfg = None
    if args.config:
        resolved = runpy.run_path(str(args.config))
        if resolved.get('_base_'):
            parser.error('--config requires the saved, fully resolved training config')
        cfg = resolved['uda']['struct_cfg']
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    folders = []
    for index, path in enumerate(args.input_npz):
        with np.load(path, allow_pickle=False) as saved:
            maps = {key: saved[key] for key in saved.files}
        meta = json.loads(args.meta_json.read_text(encoding='utf-8')) if args.meta_json else json.loads(str(maps['metadata']))
        digest = file_sha256(path)
        if args.meta_json and meta.get('arrays_sha256', {}).get('responses_full_256.npz') != digest:
            raise ValueError('NPZ hash differs from inference provenance')
        if args.config:
            meta['visualization_config'] = dict(path=str(args.config.resolve()), sha256=file_sha256(args.config))
        sample = Path(meta.get('sample_id', path.stem)).stem
        folder = output/('{}_{:06d}_{}'.format(sample, int(meta.get('iter', meta.get('gate', {}).get('rule_iter', 0))), index))
        folders.append(render_struct_bundle(maps, folder, meta, cfg, args.dpi, args.font_size, digest, args.roi))
        if file_sha256(path) != digest:
            raise RuntimeError('Input NPZ changed during rendering')
    write_run_summary(folders, output)
    print('Saved CPU Structural diagnostics in '+str(output))


if __name__ == '__main__':
    main()

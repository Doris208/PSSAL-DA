"""Render the phy-only 1x5 plate from immutable arrays, without Torch or inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import runpy
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.doris_phy_evidence import error_auc

CROP = (30, 30, 226, 226)  # x1, y1, x2, y2, exclusive upper bounds
GOJI = runpy.run_path(str(ROOT / 'mmseg/datasets/goji_meta.py'))
PALETTE = np.asarray(GOJI['GOJI_PALETTE'], dtype=np.uint8)
GRAY = np.array([208, 208, 208], dtype=np.uint8)
REQUIRED = ('image_rgb', 'ndvi', 'ndre', 'cire', 'teacher_pseudo_label',
            'teacher_confidence', 'baseline_prediction', 'hphy_prediction',
            'R_veg', 'Q_G', 'P_G', 'P_V', 'P_N', 'S_phy', 'G_phy', 'A_phy',
            'W_before', 'W_after', 'valid_mask', 'teacher_goji_mask',
            'teacher_vegetation_mask', 'teacher_phy_mask')
MAIN = (('a_target', 'image_rgb', '(a) Target\nRGB'),
        ('b_teacher', 'teacher_confidence', '(b) Teacher\nconfidence'),
        ('c_s_phy', 'S_phy', '(c) Phy agreement\nS_phy'),
        ('d_attenuation', 'A_phy', '(d) Attenuation\nA_phy'),
        ('e_prediction', 'hphy_prediction', '(e) H-Phy\nprediction'))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def output_path(path):
    path = Path(path).resolve()
    if 'doris_stage1' in path.parts:
        raise ValueError('Mechanism outputs must be isolated from doris_stage1')
    return path


def crop_arrays(arrays):
    return {key: value[30:226, 30:226].copy() for key, value in arrays.items()}


def roi_coordinates(roi):
    if roi is None:
        return None
    x1, y1, x2, y2 = roi
    if not (30 <= x1 < x2 <= 226 and 30 <= y1 < y2 <= 226):
        raise ValueError('ROI must lie inside [30,30,226,226] on the original 256 grid')
    return [x1 - 30, y1 - 30, x2 - 30, y2 - 30]


def validate_arrays(arrays, meta, tolerance=1e-5):
    missing = sorted(set(REQUIRED) - set(arrays))
    if missing:
        raise ValueError('Missing mechanism arrays: ' + ', '.join(missing))
    if meta.get('direction') != 'goji26RGB2goji30RGB':
        raise ValueError('This plate requires the RGB26 to RGB30 direction')
    if meta.get('prediction_network') != 'Student' or meta.get('teacher_network') != 'H-Phy EMA Teacher':
        raise ValueError('Teacher/official Student prediction identities differ from the protocol')
    if meta.get('class_palette') != PALETTE.tolist():
        raise ValueError('Class palette differs from GojiDataset')
    for key, value in arrays.items():
        if value.shape[:2] != (256, 256) or not np.isfinite(value).all():
            raise ValueError('Non-finite or non-256 spatial array: ' + key)
        if key != 'image_rgb' and key != 'image_6ch' and value.ndim != 2:
            raise ValueError('Expected a scalar spatial map: ' + key)
    if arrays['image_rgb'].shape != (256, 256, 3) or arrays['image_rgb'].dtype != np.uint8:
        raise ValueError('image_rgb must contain original uint8 RGB pixels')
    for key in ('teacher_pseudo_label', 'baseline_prediction', 'hphy_prediction', 'gt_label'):
        if key in arrays and not np.isin(arrays[key], [0, 1, 2, 3, 4, 255]).all():
            raise ValueError('Invalid reduced class ids in ' + key)
    for key in ('R_veg', 'Q_G', 'P_G', 'P_V', 'P_N', 'S_phy', 'G_phy', 'A_phy', 'teacher_confidence'):
        if arrays[key].min() < -tolerance or arrays[key].max() > 1 + tolerance:
            raise ValueError('Probability/gate outside [0,1]: ' + key)
    valid = arrays['valid_mask'].astype(bool)
    label = arrays['teacher_pseudo_label']
    for key, expected in (('teacher_goji_mask', valid & (label == 0)),
                          ('teacher_vegetation_mask', valid & (label == 1)),
                          ('teacher_phy_mask', valid & np.isin(label, [0, 1]))):
        if not np.array_equal(arrays[key], expected):
            raise ValueError('Incorrect Teacher applicability mask: ' + key)
    probability_dtype = meta.get('checkpoint_phy', {}).get('hphy_probability_dtype', 'float32')
    if probability_dtype not in ('float16', 'float32'):
        raise ValueError('Unsupported H-Phy probability computation dtype')
    probability_tolerance = max(tolerance, float(np.finfo(probability_dtype).eps))
    equalities = [(arrays['P_G'] + arrays['P_V'] + arrays['P_N'], np.ones((256, 256)), probability_tolerance),
                  (arrays['P_G'], arrays['R_veg'] * arrays['Q_G'], probability_tolerance),
                  (arrays['P_V'], arrays['R_veg'] * (1 - arrays['Q_G']), probability_tolerance),
                  (arrays['P_N'], 1 - arrays['R_veg'], probability_tolerance),
                  (arrays['A_phy'], 1 - arrays['G_phy'], tolerance),
                  (arrays['W_after'], arrays['W_before'] * arrays['G_phy'], tolerance)]
    for actual, expected, identity_tolerance in equalities:
        if not np.allclose(actual, expected, atol=identity_tolerance, rtol=0):
            raise ValueError('H-Phy probability/gate/weight identity failed')
    for cls, probability in ((0, 'P_G'), (1, 'P_V')):
        mask = label == cls
        if not np.allclose(arrays['S_phy'][mask], arrays[probability][mask], atol=tolerance, rtol=0):
            raise ValueError('S_phy differs from the Teacher-selected class probability')
    scope = arrays['teacher_phy_mask'].astype(bool)
    if not np.allclose(arrays['G_phy'][~scope], 1, atol=tolerance, rtol=0):
        raise ValueError('Gate must be neutral outside the actual physiology scope')
    floor = meta['gate']['min_weight']
    if arrays['G_phy'].min() < floor - tolerance or (arrays['W_after'] > arrays['W_before'] + tolerance).any():
        raise ValueError('Physiology gate increases weights or violates its floor')
    positive = valid & (arrays['W_before'] > 0)
    for key, expected in (('positive_base_weight_mask', positive),
            ('positive_phy_weight_mask', positive & scope),
            ('gate_active_mask', positive & scope & (arrays['G_phy'] < 1))):
        if key in arrays and not np.array_equal(arrays[key], expected):
            raise ValueError('Incorrect supervision mask: ' + key)
    rho = meta['gate'].get('rho')
    if rho is not None:
        expected = (1 - floor) * rho * (1 - arrays['S_phy'][scope])
        if not np.allclose(arrays['A_phy'][scope], expected, atol=tolerance, rtol=0):
            raise ValueError('Attenuation differs from the recorded training gate strength')


def descriptive_statistics(arrays, meta, minimum_auc_pixels=10):
    """One-image, center-only GT statistics, separate from validation-set metrics."""
    if 'gt_label' not in arrays:
        return dict(available=False, reason='GT was not supplied', scope='center196', groups=[])
    center = crop_arrays(arrays)
    truth, teacher = center['gt_label'], center['teacher_pseudo_label']
    valid = center['valid_mask'].astype(bool) & (truth != 255)
    groups = []
    threshold = meta['pseudo_threshold']
    for gt, pred in ((0, 0), (0, 1), (1, 1), (1, 0)):
        mask = valid & (truth == gt) & (teacher == pred)
        row = dict(gt=gt, teacher=pred, pixels=int(mask.sum()),
                   high_confidence_errors=int((mask & (truth != teacher) &
                                               (center['teacher_confidence'] >= threshold)).sum()))
        for key in ('S_phy', 'A_phy', 'teacher_confidence'):
            values = center[key][mask]
            row[key] = (dict(mean=float(values.mean()), std=float(values.std()),
                             quantiles=np.quantile(values, [.05, .25, .5, .75, .95]).tolist())
                        if values.size else None)
        groups.append(row)
    mask = valid & np.isin(truth, [0, 1]) & np.isin(teacher, [0, 1])
    errors = truth[mask] != teacher[mask]
    reason = ('fewer than {} eligible pixels'.format(minimum_auc_pixels) if errors.size < minimum_auc_pixels
              else 'error or correct group is absent' if not errors.any() or errors.all() else None)
    auc = {name: None if reason else error_auc(errors, scores)['roc_auc'] for name, scores in
           (('A_phy', center['A_phy'][mask]), ('teacher_uncertainty', 1 - center['teacher_confidence'][mask]))}
    return dict(available=True, scope='center196', crop=list(CROP), ignore_index=255,
                groups=groups, eligible_pixels=int(errors.size), errors=int(errors.sum()),
                high_confidence_errors=int((errors & (center['teacher_confidence'][mask] >= threshold)).sum()),
                auroc=auc, auroc_null_reason=reason,
                interpretation='Descriptive within-image Goji/Vegetation confusion only')


def display_spec(key, meta, attenuation_max=None):
    if key == 'image_rgb' or key.endswith('prediction') or key == 'teacher_pseudo_label':
        return None
    if key in ('A_phy', 'G_phy'):
        mask = 'positive_phy_weight_mask'
    elif key == 'S_phy':
        mask = 'teacher_phy_mask'
    else:
        mask = 'valid_mask'
    maximum = 1 - meta['gate']['min_weight'] if key == 'A_phy' else 1.0
    if key == 'A_phy' and attenuation_max is not None:
        maximum = attenuation_max
    minimum = 0.0
    if key in ('ndvi', 'ndre', 'cire'):
        settings = meta['vi_normalization'][key]
        minimum, maximum = settings['min'], settings['max']
    return dict(vmin=minimum, vmax=maximum, cmap='magma' if key == 'A_phy' else 'viridis', mask=mask)


def colored_pixels(arrays, key, meta, attenuation_max=None):
    values = arrays[key]
    if key == 'image_rgb':
        return values.copy()
    if key.endswith('prediction') or key == 'teacher_pseudo_label':
        result = np.full((*values.shape, 3), GRAY, dtype=np.uint8)
        for cls, color in enumerate(PALETTE):
            result[values == cls] = color
        return result
    spec = display_spec(key, meta, attenuation_max)
    result = plt.get_cmap(spec['cmap'])(Normalize(spec['vmin'], spec['vmax'])(values), bytes=True)[..., :3]
    mask = arrays.get(spec['mask'])
    if mask is None and spec['mask'] == 'positive_phy_weight_mask':
        # Historical NPZ files predate the explicit positive-weight mask.
        mask = arrays['teacher_phy_mask'].astype(bool) & (arrays['W_before'] > 0)
    result[~mask.astype(bool)] = GRAY
    return result


def draw_map(axis, arrays, key, meta, attenuation_max=None):
    artist = axis.imshow(colored_pixels(arrays, key, meta, attenuation_max), interpolation='nearest')
    axis.set_axis_off()
    axis.set_aspect('equal')
    return artist


def synthetic_label(fig, meta):
    if meta.get('synthetic_fixture'):
        fig.text(.5, .015, 'SYNTHETIC TEST DATA', ha='center', va='bottom', fontsize=7, color='#8c2424')


def add_bar(fig, bounds, key, meta, maximum=None, font_size=7):
    spec = display_spec(key, meta, maximum)
    axis = fig.add_axes(bounds)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(spec['vmin'], spec['vmax']),
                                            cmap=spec['cmap']), cax=axis, orientation='horizontal',
                 ticks=[spec['vmin'], (spec['vmin'] + spec['vmax']) / 2, spec['vmax']])
    axis.tick_params(labelsize=font_size, length=2, pad=2)


def draw_roi_box(axis, roi, offset=0):
    if roi is not None:
        x1, y1, x2, y2 = roi
        axis.add_patch(Rectangle((x1-offset-.5, y1-offset-.5), x2-x1, y2-y1,
                                 fill=False, edgecolor='white', linewidth=1.0))


def paired_roi(arrays, roi, meta):
    x1, y1, x2, y2 = roi
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 2.1))
    for axis, key, title in zip(axes, ('baseline_prediction', 'hphy_prediction'), ('Baseline', 'H-Phy')):
        pixels = colored_pixels(arrays, key, meta)[y1:y2, x1:x2]
        axis.imshow(pixels, interpolation='nearest')
        axis.set_aspect('equal')
        axis.set_axis_off()
        axis.set_title(title, fontsize=9)
    fig.subplots_adjust(left=.03, right=.97, top=.85, bottom=.12, wspace=.12)
    synthetic_label(fig, meta)
    return fig


def group_figure(arrays, meta, roi=None, font_size=9, attenuation_max=None):
    center = crop_arrays(arrays)
    translated = roi_coordinates(roi)
    width, height, left, gap = 7.2, 3.0 if roi else 2.65, .12, .08
    side = (width - 2 * left - 4 * gap) / 5
    bottom = height - .45 - side
    fig = plt.figure(figsize=(width, height))
    axes = []
    for index, (_, key, title) in enumerate(MAIN):
        x = left + index * (side + gap)
        axis = fig.add_axes([x / width, bottom / height, side / width, side / height])
        draw_map(axis, center, key, meta, attenuation_max)
        axis.set_title(title, fontsize=font_size, pad=6)
        axes.append(axis)
        if key in ('teacher_confidence', 'S_phy', 'A_phy'):
            add_bar(fig, [(x+.1)/width, (bottom-.18)/height, (side-.2)/width, .055/height],
                    key, meta, attenuation_max, font_size=7)
    handles = [Patch(color=color / 255., label=name.title()) for name, color in zip(GOJI['GOJI_CLASSES'], PALETTE)]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.015, .05),
               ncol=3, frameon=False, fontsize=6.5, handlelength=.8, columnspacing=.9)
    if roi is not None:
        draw_roi_box(axes[-1], translated)
        x1, y1, x2, y2 = roi
        x = left + 4 * (side + gap)
        for index, key in enumerate(('baseline_prediction', 'hphy_prediction')):
            axis = fig.add_axes([(x+index*(side/2+.015))/width, .16/height,
                                 (side/2-.02)/width, .62/height])
            axis.imshow(colored_pixels(arrays, key, meta)[y1:y2, x1:x2], interpolation='nearest')
            axis.set_aspect('equal')
            axis.set_axis_off()
            axis.set_title('Baseline' if index == 0 else 'H-Phy', fontsize=6.5, pad=2)
    synthetic_label(fig, meta)
    return fig


def save_figure(fig, path, dpi):
    try:
        fig.savefig(str(path), dpi=dpi, facecolor='white')
    finally:
        plt.close(fig)


def export_panels(arrays, meta, folder, dpi, font_size, attenuation_max, roi=None):
    folder.mkdir(parents=True)
    (folder / 'clean').mkdir()
    (folder / 'labeled').mkdir()
    names = list(MAIN) + [('teacher_confidence', 'teacher_confidence', 'Teacher confidence'),
        ('teacher_pseudo_label', 'teacher_pseudo_label', 'Teacher pseudo-label'),
        ('baseline_prediction', 'baseline_prediction', 'Baseline'),
        ('hphy_prediction', 'hphy_prediction', 'H-Phy')]
    names += [(key, key, key) for key in ('ndvi', 'ndre', 'cire', 'R_veg', 'Q_G', 'P_G', 'P_V', 'P_N', 'G_phy', 'W_before', 'W_after')]
    for filename, key, title in names:
        pixels = colored_pixels(arrays, key, meta, attenuation_max)
        info = {'Description': 'SYNTHETIC TEST DATA' if meta.get('synthetic_fixture') else 'H-Phy offline response'}
        for parent in (folder, folder / 'clean'):
            plt.imsave(str(parent / (filename + '.png')), pixels, dpi=dpi, metadata=info)
        fig = plt.figure(figsize=(2.05, 2.4))
        axis = fig.add_axes([.1, .22, .8, .8*2.05/2.4])
        draw_map(axis, arrays, key, meta, attenuation_max)
        axis.set_title(title.replace('\n', ' '), fontsize=font_size, pad=5)
        if display_spec(key, meta, attenuation_max) is not None:
            add_bar(fig, [.16, .12, .68, .035], key, meta, attenuation_max)
        synthetic_label(fig, meta)
        save_figure(fig, folder / 'labeled' / (filename + '.png'), dpi)
    if roi is not None:
        save_figure(paired_roi(arrays, roi, meta), folder / 'baseline_vs_hphy_roi.png', dpi)


def render(input_npz, meta_json, output_dir, roi=None, dpi=600, font_size=9,
           export_group=True, export_individual=True, export_full=True, attenuation_max=None):
    roi_coordinates(roi)
    if dpi <= 0 or font_size <= 0 or (attenuation_max is not None and attenuation_max <= 0):
        raise ValueError('DPI, font size and attenuation color maximum must be positive')
    input_npz, meta_json = Path(input_npz).resolve(), Path(meta_json).resolve()
    original_hash = file_sha256(input_npz)
    meta_hash = file_sha256(meta_json)
    with np.load(input_npz, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    meta = json.loads(meta_json.read_text(encoding='utf-8'))
    if meta.get('arrays_sha256', {}).get('responses_full_256.npz') != original_hash:
        raise ValueError('NPZ hash differs from its inference provenance')
    validate_arrays(arrays, meta)
    center = crop_arrays(arrays)
    out = output_path(output_dir)
    products = ['phy_stats.json', 'phy_stats.csv', 'render_meta.json', 'colorbars', 'logs/render.log']
    if export_group:
        products += ['phy_mechanism_1x5.png', 'phy_mechanism_1x5.pdf']
    if export_individual:
        products.append('panels_196')
    if export_full:
        products.append('panels_256')
    for name in products:
        if (out / name).exists():
            raise FileExistsError('Existing rendering output: ' + str(out / name))
    out.mkdir(parents=True, exist_ok=True)
    # Raw inference files may already be present when replay and rendering share a run.
    import shutil
    for source, name in ((input_npz, 'responses_full_256.npz'), (meta_json, 'mechanism_meta.json')):
        destination = out / name
        if destination.resolve() != source:
            if destination.exists():
                if file_sha256(destination) != file_sha256(source):
                    raise FileExistsError(str(destination))
            else:
                shutil.copyfile(str(source), str(destination))
    center_path = out / 'responses_center_196.npz'
    if center_path.exists():
        with np.load(center_path, allow_pickle=False) as saved:
            if set(saved.files) != set(center) or any(not np.array_equal(saved[key], center[key]) for key in center):
                raise ValueError('Existing center arrays differ from the exact full-grid slice')
    else:
        np.savez_compressed(str(center_path), **center)
    stats = descriptive_statistics(arrays, meta)
    (out / 'phy_stats.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    fields = ['gt', 'teacher', 'pixels', 'high_confidence_errors'] + [key + '_' + stat for key in
              ('S_phy', 'A_phy', 'teacher_confidence') for stat in ('mean', 'std')]
    with (out / 'phy_stats.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in stats['groups']:
            flat = {key: row[key] for key in fields[:4]}
            flat.update({key + '_' + stat: row[key][stat] if row[key] else None
                         for key in ('S_phy', 'A_phy', 'teacher_confidence') for stat in ('mean', 'std')})
            writer.writerow(flat)
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': font_size, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        if export_group:
            for suffix in ('png', 'pdf'):
                save_figure(group_figure(arrays, meta, roi, font_size, attenuation_max),
                            out / ('phy_mechanism_1x5.' + suffix), dpi)
        if export_individual:
            export_panels(center, meta, out / 'panels_196', dpi, font_size, attenuation_max,
                          roi_coordinates(roi))
        if export_full:
            export_panels(arrays, meta, out / 'panels_256', dpi, font_size, attenuation_max, roi)
        (out / 'colorbars').mkdir()
        for key, filename in (('S_phy', 's_phy'), ('A_phy', 'attenuation'), ('teacher_confidence', 'confidence')):
            fig = plt.figure(figsize=(2.1, .5))
            add_bar(fig, [.1, .5, .8, .18], key, meta, attenuation_max)
            save_figure(fig, out / 'colorbars' / (filename + '.png'), dpi)
    settings = dict(meta, render=dict(group_panels=[key for _, key, _ in MAIN],
        roi_original_256=roi, roi_center_196=roi_coordinates(roi),
        dpi=dpi, font_size=font_size, width_inches=7.2, interpolation='nearest',
        S_phy_range=[0, 1], confidence_range=[0, 1],
        attenuation_range=[0, attenuation_max if attenuation_max is not None else 1-meta['gate']['min_weight']],
        colormaps=dict(S_phy='viridis', A_phy='magma', teacher_confidence='viridis'),
        nonapplicable_rgb=GRAY.tolist(), font_family=['Arial', 'DejaVu Sans'],
        color_scale_note='Use matching scales across samples and checkpoint comparisons',
        source_npz_sha256=original_hash, source_metadata_sha256=meta_hash,
        renderer_sha256=file_sha256(__file__)))
    settings['output_files'] = sorted(path.relative_to(out).as_posix() for path in out.rglob('*') if path.is_file())
    settings['output_files'].extend(['render_meta.json', 'logs/render.log'])
    (out / 'render_meta.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    (out / 'logs').mkdir(exist_ok=True)
    with (out / 'logs/render.log').open('x', encoding='utf-8') as log:
        log.write('Pure NPZ rendering completed\nInput SHA256 ' + original_hash + '\n')
        log.write(json.dumps(settings['render'], indent=2) + '\n')
    if file_sha256(input_npz) != original_hash or file_sha256(meta_json) != meta_hash:
        raise RuntimeError('Input arrays or provenance changed during rendering')
    return out


def export_online_views(input_npz, meta_json, output_dir):
    """Render the online layout and native panels from this replay's saved arrays."""
    from tools.build_doris_mechanism_figures import render_online_node
    input_npz = Path(input_npz)
    meta = json.loads(Path(meta_json).read_text(encoding='utf-8'))
    if meta['arrays_sha256']['responses_full_256.npz'] != file_sha256(input_npz):
        raise ValueError('NPZ hash differs from its inference provenance')
    with np.load(input_npz, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    validate_arrays(arrays, meta)
    aliases = dict(w_conf='teacher_confidence', pseudo_label='teacher_pseudo_label',
        teacher_label='teacher_pseudo_label', teacher_confidence='teacher_confidence',
        r_phy='R_veg', r_veg='R_veg', q_goji_given_veg='Q_G',
        p_goji='P_G', p_other_veg='P_V', p_nonveg='P_N',
        s_phy_class_agreement='S_phy', g_phy='G_phy', a_phy='A_phy', attenuation_phy='A_phy',
        w_before_phy='W_before', w_after_phy='W_after', valid_mask='valid_mask')
    maps = {key: arrays[source] for key, source in aliases.items()}
    for key in ('vi_ndvi', 'vi_ndre', 'vi_cire', 'teacher_top2_margin'):
        if key in arrays:
            maps[key] = arrays[key]
    completed = meta['checkpoint_phy']['completed_iter']
    rho = meta['gate']['rho']
    info = dict(meta, iter=completed, rule_iter=meta['gate']['rule_iter'],
        capture_mode='checkpoint_replay', data_source='fixed_probe_recomputed', output_mode='branches',
        modules=dict(phy=True, struct=False, proto=False, edge=False),
        states=dict(phy='warmup' if rho == 0 else 'ramping' if rho < 1 else 'active',
                    struct='disabled', proto='disabled', edge='disabled'),
        input_view=['R', 'G', 'B'], gate_min_weight=meta['gate']['min_weight'],
        state_sha256=meta['checkpoint_phy']['state_sha256_before'],
        config_sha256=meta['config_phy']['resolved_sha256'],
        source_responses_sha256=file_sha256(input_npz))
    maps.update(target_image=arrays['image_rgb'].astype(np.float32)/255,
        teacher_label=arrays['teacher_pseudo_label'].astype(np.float32),
        m_phy=np.isin(arrays['teacher_pseudo_label'], [0, 1]),
        delta_w_phy=arrays['W_after']-arrays['W_before'], rule_iter=np.array(info['rule_iter']),
        proto_active=np.array(False), metadata=np.array(json.dumps(info, sort_keys=True)))
    folder = output_path(output_dir) / 'fixed_probe'
    path = folder / 'selected_arrays' / ('iter_{:06d}.npz'.format(completed))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **maps)
    render_online_node(maps, folder / 'snapshots', completed,
                       array_sha256=file_sha256(path), render_timing='checkpoint_replay')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-npz', required=True)
    parser.add_argument('--meta-json', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--render-only', action='store_true', help='Rendering always uses NPZ only')
    parser.add_argument('--crop-mode', choices=['center196'], default='center196')
    parser.add_argument('--roi', nargs=4, type=int)
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--font-size', type=float, default=9)
    parser.add_argument('--attenuation-vmax', type=float)
    parser.add_argument('--export-group', action='store_true')
    parser.add_argument('--export-panels', action='store_true')
    parser.add_argument('--export-full256', action='store_true')
    args = parser.parse_args()
    if args.attenuation_vmax is not None:
        print('Attenuation color-scale override requested. Use the same scale across compared samples/checkpoints.', file=sys.stderr)
    all_outputs = not (args.export_group or args.export_panels or args.export_full256)
    output = render(args.input_npz, args.meta_json, args.output_dir, args.roi, args.dpi, args.font_size,
                    all_outputs or args.export_group, all_outputs or args.export_panels,
                    all_outputs or args.export_full256, args.attenuation_vmax)
    print('Rendered H-Phy plate and panels in ' + str(output))


if __name__ == '__main__':
    main()

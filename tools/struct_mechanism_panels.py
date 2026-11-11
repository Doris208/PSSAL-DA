"""Shared Structural paper panels for live fixed probes and CPU NPZ rendering."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from tools.build_hphy_mechanism_figure import PALETTE, crop_arrays, file_sha256, roi_coordinates, save_figure
from tools.doris_struct_evidence import CONTRIBUTIONS, RESPONSES, contribution_keys, prepare_struct_evidence, struct_statistics

VERSION = 4
BACKGROUND = np.array([236, 236, 236], dtype=np.uint8)
INVALID = np.array([180, 180, 180], dtype=np.uint8)
COMPONENTS = ('image_rgb',)+RESPONSES+('struct_prediction',)
FORMAL_NAME = 'struct_formal_group_1x7'
FORMAL_PANELS = dict(image_rgb='target_rgb_formal', R_conf='r_conf_struct_only_formal',
    R_loc='r_loc_struct_only_formal', R_adj='r_adj_struct_only_formal',
    R_prob='r_prob_struct_only_formal', R_struct='r_struct_struct_only_formal',
    struct_prediction='struct_student_prediction_formal')
SYMBOLS = dict(R_conf=r'$R_{\mathrm{conf}}$', R_loc=r'$R_{\mathrm{loc}}$',
    R_adj=r'$R_{\mathrm{adj}}$', R_prob=r'$R_{\mathrm{prob}}$', R_struct=r'$R_{\mathrm{struct}}$',
    A_struct=r'$A_{\mathrm{struct}}$', C_loc=r'$C_{\mathrm{loc}}$',
    C_conf=r'$C_{\mathrm{conf}}$',
    C_adj=r'$C_{\mathrm{adj}}$', C_prob=r'$C_{\mathrm{prob}}$')
TITLES = dict(image_rgb='Target RGB', R_conf='Confidence', R_loc='Local', R_adj='Adjacency',
    R_prob='Semantic', R_struct='Structural response', A_struct='Attenuation',
    C_conf='Confidence', C_loc='Local', C_adj='Adjacency', C_prob='Semantic',
    baseline_prediction='Baseline', struct_prediction='Structural Student prediction')
MODES = dict(struct_formal_group_1x7=('formal', COMPONENTS),
    struct_full_raw=('full', COMPONENTS), struct_only_raw=('struct', COMPONENTS),
    struct_only_contrast=('contrast', COMPONENTS), struct_contrib=('struct', CONTRIBUTIONS+('R_struct', 'A_struct')))


def mask_for(arrays, key, mode):
    mask = arrays['valid_mask'].astype(bool).copy()
    if mode != 'full':
        mask &= arrays['struct_display_mask']
    if key == 'A_struct':
        mask &= arrays['A_struct_valid_mask']
    if key in ('R_adj', 'R_prob'):
        mask &= arrays['adjacency_available' if key == 'R_adj' else 'semantic_available']
    return mask


def display_scales(arrays, meta, mode):
    scales = {key: [0., 1.] for key in RESPONSES+('C_conf',)+CONTRIBUTIONS}
    if mode == 'contrast':
        for key in RESPONSES:
            values = arrays[key][mask_for(arrays, key, mode)]
            if values.size:
                low, high = np.quantile(values, [.02, .98]).tolist()
                if high > low:
                    scales[key] = [low, high]
    scales['A_struct'] = meta['attenuation_color_range']
    return scales


def pixels_for(arrays, key, mode, scales):
    values = arrays[key]
    if key == 'image_rgb':
        return values.copy()
    if key.endswith('prediction') or key in ('teacher_pred', 'gt_label'):
        result = np.full((*values.shape, 3), INVALID, dtype=np.uint8)
        for cls, color in enumerate(PALETTE):
            result[values == cls] = color
        return result
    low, high = scales[key]
    cmap = 'magma' if key == 'A_struct' else 'viridis'
    pixels = plt.get_cmap(cmap)(Normalize(low, high)(values), bytes=True)[..., :3]
    pixels[~mask_for(arrays, key, mode)] = INVALID
    if mode != 'full' or key == 'A_struct':
        pixels[arrays['valid_mask'].astype(bool) & ~arrays['struct_class_mask']] = BACKGROUND
    return pixels


def bar(fig, bounds, key, scales, font_size=7):
    low, high = scales[key]
    artist = matplotlib.cm.ScalarMappable(norm=Normalize(low, high), cmap='magma' if key == 'A_struct' else 'viridis')
    axis = fig.add_axes(bounds)
    fig.colorbar(artist, cax=axis, orientation='horizontal', ticks=[low, (low+high)/2, high])
    axis.tick_params(labelsize=font_size, length=2, pad=2)
    axis.xaxis.set_major_formatter(FormatStrFormatter('%.3g' if key == 'A_struct' else '%.2g'))


def panel_figure(arrays, keys, mode, scales, meta, font_size=9, roi=None):
    if 'struct_prediction' in keys and 'struct_prediction' not in arrays:
        raise ValueError('Formal Structural panels require struct_prediction from checkpoint Student test inference')
    width, left, gap = 7.2, .12, .075
    side = (width-2*left-(len(keys)-1)*gap)/len(keys)
    height = side+1.20
    fig = plt.figure(figsize=(width, height))
    for index, key in enumerate(keys):
        x = left+index*(side+gap)
        axis = fig.add_axes([x/width, .65/height, side/width, side/height])
        axis.imshow(pixels_for(arrays, key, mode, scales), interpolation='nearest')
        axis.set_axis_off()
        title = TITLES[key].replace('Structural response', 'Structural\nresponse').replace(
            'Structural Student prediction', 'Structural\nStudent\nprediction')
        symbol = SYMBOLS.get(key, '')
        if symbol:
            title = title+'\n'+symbol
        axis.set_title('({}) {}'.format(chr(97+index), title), fontsize=font_size, pad=5)
        if roi is not None and key.endswith('prediction'):
            x1, y1, x2, y2 = roi
            axis.add_patch(Rectangle((x1-.5, y1-.5), x2-x1, y2-y1, fill=False, edgecolor='white', linewidth=.8))
        if key in scales and (mode == 'contrast' or key == 'A_struct'):
            bar(fig, [(x+.10)/width, .45/height, (side-.20)/width, .045/height], key, scales, 6)
    # Raw response and contribution panels share the same absolute 0..1 scale.
    if mode != 'contrast':
        bar(fig, [.27, .45/height, .31, .045/height], 'R_struct', scales, 6)
    notes = dict(full='Full valid area | Raw values in [0,1]',
        formal='Teacher Building / Water / Road responses | Raw [0,1] | Student five-class prediction',
        struct='Struct classes only | Building / Water / Road',
        contrast='Contrast-enhanced on struct-class pixels only | Visualization only; raw values preserved')
    note = notes[mode]+'\nLight gray = other classes | Dark gray = invalid / unavailable'
    if 'struct_prediction' in keys and mode != 'formal':
        note = notes[mode]+'\nResponses from EMA Teacher | Prediction from Student'
    if meta.get('synthetic_fixture'):
        note += '\nSYNTHETIC TEST DATA'
    fig.text(.5, .075, note, ha='center', va='center', fontsize=5.5)
    return fig


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_formal_panels(arrays, scales, folder, dpi):
    folder.mkdir(parents=True, exist_ok=True)
    for key, filename in FORMAL_PANELS.items():
        plt.imsave(str(folder/(filename+'.png')), pixels_for(arrays, key, 'formal', scales), dpi=dpi)


def export_single(arrays, key, mode, scales, meta, folder, dpi, font_size):
    folder.mkdir(parents=True, exist_ok=True)
    pixels = pixels_for(arrays, key, mode, scales)
    plt.imsave(str(folder/(key+'_no_title.png')), pixels, dpi=dpi,
        metadata={'Description': 'SYNTHETIC TEST DATA' if meta.get('synthetic_fixture') else 'Structural diagnostic v4'})
    plt.imsave(str(folder/(key+'.png')), pixels, dpi=dpi)
    fig = plt.figure(figsize=(2.05, 2.4))
    axis = fig.add_axes([.1, .22, .8, .8*2.05/2.4])
    axis.imshow(pixels, interpolation='nearest')
    axis.set_axis_off()
    axis.set_title(TITLES[key].replace('Structural Student prediction', 'Structural Student\nprediction')+
        '\n'+SYMBOLS.get(key, ''), fontsize=font_size, pad=5)
    if key in scales:
        bar(fig, [.16, .12, .68, .035], key, scales)
    if meta.get('synthetic_fixture'):
        fig.text(.5, .015, 'SYNTHETIC TEST DATA', ha='center', fontsize=6)
    # Both A_struct formats use the fixed physical scale, including in contrast mode.
    if key == 'A_struct':
        fig.savefig(str(folder/'A_struct_raw.pdf'), facecolor='white')
    if key == 'struct_prediction':
        fig.savefig(str(folder/'struct_prediction.pdf'), facecolor='white')
    save_figure(fig, folder/(key+'_raw.png'), dpi)
    if key == 'A_struct':
        fig = plt.figure(figsize=(2.1, .5))
        bar(fig, [.1, .5, .8, .18], key, scales)
        save_figure(fig, folder/'A_struct_colorbar.png', dpi)


def render_struct_bundle(saved, output_dir, metadata=None, config=None, dpi=600, font_size=9,
                         source_sha256='', roi=None):
    """One formal layout and auxiliary diagnostics from an immutable probe payload."""
    roi_coordinates(roi)
    if dpi <= 0 or font_size <= 0:
        raise ValueError('DPI and font size must be positive')
    arrays, meta = prepare_struct_evidence(saved, metadata, config)
    if 'struct_prediction' not in arrays:
        raise ValueError('Formal Structural 1x7 requires struct_prediction from Student test inference. '
                         'Use tools/replay_struct_mechanism.py to recover it from the checkpoint.')
    fingerprint = hashlib.sha256()
    for key in ('image_rgb', 'teacher_pseudo_label', 'valid_mask', 'W_before', 'W_after',
                'adjacency_available', 'semantic_available')+RESPONSES:
        fingerprint.update(np.ascontiguousarray(arrays[key]).tobytes())
    for key in ('baseline_prediction', 'struct_prediction', 'gt_label'):
        if key in arrays:
            fingerprint.update(key.encode())
            fingerprint.update(np.ascontiguousarray(arrays[key]).tobytes())
    options = dict(version=VERSION, dpi=dpi, font_size=font_size, roi=roi, struct_cfg=meta['struct_cfg'],
                   rule_iter=meta['rule_iter'], source_sha256=source_sha256,
                   prediction_checkpoint=meta.get('prediction_checkpoint', meta.get('checkpoint_struct')),
                   student_inference=meta.get('student_inference'),
                   renderer_sha256=file_sha256(__file__), evidence_sha256=file_sha256(Path(__file__).with_name('doris_struct_evidence.py')))
    fingerprint.update(json.dumps(options, sort_keys=True).encode())
    identity = fingerprint.hexdigest()
    out = Path(output_dir).resolve()
    manifest = out/'render_meta.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding='utf-8'))
        if previous.get('render_identity') == identity and all((out/name).is_file() for name in previous['output_files']):
            return out
        raise FileExistsError('Existing Structural visualization has different provenance: '+str(out))
    out.mkdir(parents=True, exist_ok=False)
    validation = meta['struct_validation']
    (out/'validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    np.savez_compressed(str(out/'responses_full_256.npz'), **arrays)
    center = crop_arrays(arrays)
    np.savez_compressed(str(out/'responses_center_196.npz'), **center)
    if not validation['passed']:
        raise ValueError('Structural diagnostic validation failed; raw arrays and audit saved in '+str(out)+': '+', '.join(validation['errors']))
    stats = struct_statistics(arrays, meta)
    stats['gt_available'] = 'gt_label' in arrays
    if 'gt_label' in arrays:
        from tools.build_struct_mechanism_figure import descriptive_statistics
        reference = descriptive_statistics(arrays, meta)
        for key in ('teacher_errors', 'high_confidence_teacher_errors', 'error_detection', 'prediction_iou'):
            stats[key] = reference[key]
    (out/'struct_stats.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    write_csv(out/'struct_stats.csv', stats['attenuation_by_class'])
    write_csv(out/'struct_response_stats.csv', [dict(metric=key, **value) for key, value in stats['responses'].items()])
    write_csv(out/'struct_response_by_class.csv', stats['responses_by_class'])
    write_csv(out/'struct_class_distribution.csv', stats['class_distribution'])
    write_csv(out/'struct_correlations.csv', [dict(pair=key, correlation=value) for key, value in stats['correlations'].items()])
    write_csv(out/'struct_grouped_correlations.csv', stats['grouped_correlations'])
    caption = (
        'R_conf is the EMA Teacher maximum class probability. '
        'R_loc measures eight-neighbor class continuity. '
        'R_adj measures agreement with the saved source class-adjacency prior. '
        'R_prob follows the semantic_score_mode recorded in render_meta.json. '
        'R_struct follows the captured fusion_mode and production availability masks. '
        'The formal seventh panel is the selected-state Student test-mode argmax prediction. '
        'Formal response panels display captured raw values only on valid Teacher Building, Water and Road pixels, '
        'with a shared viridis scale from 0 to 1. '
        'All five classes use the H-Phy palette, including in structural-class response views. '
        'Available additive contribution terms follow the recorded fusion mode. '
        'A_struct = 1 - W_after/W_before measures the subsequent supervision attenuation on '
        'valid Teacher Building, Water and Road pixels with W_before > 0. '
        'The center196 display is the exact [30:226,30:226] slice of the native 256 grid. '
        'Contrast enhancement uses per-response 2nd and 98th percentiles on center196 structural pixels. '
        'A_struct always uses the fixed saved-gate numerical scale. Raw response arrays are preserved.\n')
    (out/'figure_caption.txt').write_text(caption, encoding='utf-8')
    scales_record = {}
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': font_size, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        for stem, (mode, keys) in MODES.items():
            if stem == 'struct_contrib':
                if not contribution_keys(meta):
                    continue
                keys = contribution_keys(meta)+('R_struct', 'A_struct')
            scales = display_scales(center, meta, mode)
            scales_record[stem] = {key: scales[key] for key in keys if key in scales}
            fig = panel_figure(center, keys, mode, scales, meta, font_size)
            fig.savefig(str(out/(stem+'.pdf')), facecolor='white')
            save_figure(fig, out/(stem+'.png'), dpi)
        formal_scales = display_scales(center, meta, 'formal')
        export_formal_panels(center, formal_scales, out, dpi)
        if 'baseline_prediction' in arrays and 'struct_prediction' in arrays:
            keys = ('image_rgb', 'baseline_prediction', 'struct_prediction', 'R_struct', 'A_struct')
            fig = panel_figure(center, keys, 'struct', display_scales(center, meta, 'struct'), meta, font_size, roi_coordinates(roi))
            fig.savefig(str(out/'struct_comparison_1x5.pdf'), facecolor='white')
            save_figure(fig, out/'struct_comparison_1x5.png', dpi)
        for size, current in ((256, arrays), (196, center)):
            folder = out/('panels_'+str(size))
            folder.mkdir()
            export_formal_panels(current, formal_scales, folder/'formal', dpi)
            np.save(str(folder/'A_struct.npy'), current['A_struct'])
            np.save(str(folder/'A_struct_valid_mask.npy'), current['A_struct_valid_mask'])
            raw_scales = display_scales(center, meta, 'full')
            for key in ('image_rgb', 'teacher_pred', 'baseline_prediction', 'gt_label'):
                if key in current:
                    plt.imsave(str(folder/(key+'.png')), pixels_for(current, key, 'full', raw_scales), dpi=dpi)
            export_single(current, 'struct_prediction', 'full', raw_scales, meta, folder, dpi, font_size)
            # Stretch parameters are fitted on center196 and applied identically to full256.
            for mode, stem in (('full', 'full_raw'), ('struct', 'struct_only'), ('contrast', 'contrast')):
                scales = display_scales(center, meta, mode)
                for key in RESPONSES+('A_struct',):
                    export_single(current, key, mode, scales, meta, folder/stem, dpi, font_size)
            for key in contribution_keys(meta):
                export_single(current, key, 'struct', display_scales(center, meta, 'struct'), meta, folder/'contributions', dpi, font_size)
            # Direct A_struct files make the paper panel easy to locate.
            export_single(current, 'A_struct', 'struct', display_scales(center, meta, 'struct'), meta, folder, dpi, font_size)
            if roi is not None and 'baseline_prediction' in current and 'struct_prediction' in current:
                coordinates = roi_coordinates(roi) if size == 196 else roi
                x1, y1, x2, y2 = coordinates
                fig, axes = plt.subplots(1, 2, figsize=(3.3, 2.1))
                for axis, key in zip(axes, ('baseline_prediction', 'struct_prediction')):
                    pixels = pixels_for(current, key, 'full', raw_scales)
                    axis.imshow(pixels[y1:y2, x1:x2], interpolation='nearest')
                    axis.set_title(TITLES[key], fontsize=font_size)
                    axis.set_axis_off()
                save_figure(fig, folder/'baseline_vs_struct_roi.png', dpi)
    meta['source_arrays_sha256'] = meta.get('arrays_sha256', {})
    meta['arrays_sha256'] = {name: file_sha256(out/name) for name in ('responses_full_256.npz', 'responses_center_196.npz')}
    meta.update(render_identity=identity, render_options=options,
        formal_group_panels=list(COMPONENTS),
        formal_version=True, formal_group_file=FORMAL_NAME+'.png',
        formal_response_display_mask='struct_display_mask', formal_mask_source='EMA Teacher class IDs',
        formal_structural_classes=dict(ids=[2, 3, 4], names=['Building', 'Water', 'Road']),
        formal_response_range=[0., 1.], formal_response_colormap='viridis',
        formal_display_crop=[30, 30, 226, 226], formal_display_shape=[196, 196],
        formal_prediction_field='struct_prediction', formal_prediction_identity='Student test-mode prediction',
        formal_prediction_checkpoint=meta.get('prediction_checkpoint', meta.get('checkpoint_struct')),
        formal_prediction_inference=meta.get('student_inference', dict(status='not recorded in source arrays')),
        formal_candidate_figures=[FORMAL_NAME],
        auxiliary_figures=['struct_full_raw', 'struct_only_raw', 'struct_only_contrast',
            'struct_contrib', 'struct_comparison_1x5', 'A_struct panels'],
        render=dict(width_inches=7.2, crop=[30, 30, 226, 226], interpolation='nearest',
            scales=scales_record, quantiles=[.02, .98], contrast_reference='center196 structural pixels',
            attenuation_scale='fixed saved-gate bound; original A_struct values preserved',
            contribution_identity='+'.join(contribution_keys(meta))+' = R_struct' if contribution_keys(meta) else 'Legacy product has no additive decomposition',
            invalid_rgb=INVALID.tolist(), nonstructural_rgb=BACKGROUND.tolist()),
        output_files=sorted(p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()))
    (out/'logs').mkdir()
    (out/'logs/render.log').write_text('Shared Structural v4 rendering completed\n'+identity+'\n', encoding='utf-8')
    meta['output_files'] += ['logs/render.log', 'render_meta.json']
    manifest.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return out


def write_run_summary(folders, output_dir):
    rows = []
    for folder in folders:
        stats = json.loads((Path(folder)/'struct_stats.json').read_text(encoding='utf-8'))
        combined = stats['attenuation_by_class'][-1]
        rows.append(dict(sample_id=stats['sample_id'], selection_score=stats['selection_score'],
            struct_class_ratio=stats['struct_class_ratio'], **combined,
            interior_A_mean=stats['attenuation_by_neighborhood']['interior']['mean'],
            mixed_neighborhood_A_mean=stats['attenuation_by_neighborhood']['mixed_neighborhood']['mean'],
            formal_group_png=str(Path(folder).resolve()/(FORMAL_NAME+'.png')),
            output_dir=str(Path(folder).resolve())))
    rows.sort(key=lambda row: (row['selection_score'] is not None, row['selection_score'] or 0), reverse=True)
    for rank, row in enumerate(rows, 1):
        row['candidate_rank'] = rank if row['selection_score'] is not None else None
    write_csv(Path(output_dir)/'struct_samples_summary.csv', rows)
    (Path(output_dir)/'struct_samples_summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    return rows

"""Render saved mechanism evidence only; no Torch, GPU or inference required."""
import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import runpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

GOJI_PALETTE = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                 'mmseg/datasets/goji_meta.py'))['GOJI_PALETTE']
CLASS_LABEL_KEYS = ('pseudo_label', 'teacher_label', 'reference_label',
                    'teacher_pseudo_label', 'gt_label', 'baseline_prediction', 'hphy_prediction')

try:
    from tools.doris_probe_data import spatial_statistics
except ModuleNotFoundError:  # Direct `python tools/build_doris_mechanism_figures.py`.
    from doris_probe_data import spatial_statistics


plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 7,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False,
    'axes.spines.right': False, 'legend.frameon': False})
TITLES = dict(r_conf=r'$R_{\mathrm{conf}}$', r_loc=r'$R_{\mathrm{loc}}$',
    r_adj=r'$R_{\mathrm{adj}}$', r_prob=r'$R_{\mathrm{prob}}$',
    r_struct=r'$R_{\mathrm{struct}}$', w_conf='Teacher confidence\n(maximum class probability)',
    w_t_ps=r'$W_t^{\mathrm{PS}}$', w_t_proto=r'$W_t^{\mathrm{proto}}$',
    delta_w_proto=r'$\Delta W_{\mathrm{proto}}$',
    w_base=r'$W_{\mathrm{DACS}}$', w_pre_proto=r'$W_{\mathrm{in}}^{\mathrm{proto}}$',
    r_phy='VI-driven response\n'+r'$R_{\mathrm{phy}}$', w_t_phy=r'$W_t^{\mathrm{phy}}$',
    delta_w_phy=r'$\Delta W_{\mathrm{phy}}$', delta_w_ps=r'$\Delta W_{\mathrm{PS}}$',
    e_prob=r'$E_{\mathrm{prob}}$', e_img=r'$E_{\mathrm{img}}$',
    r_pair=r'$R_{\mathrm{pair}}$', r_edge=r'$R_{\mathrm{edge}}$',
    boundary_weight=r'$W_{\mathrm{edge}}$',
    w_before_phy='DACS weight before\nphysiology gate', w_after_phy='Weight after\nphysiology gate',
    w_before_proto=r'$W_{\mathrm{before\ proto}}$', w_after_proto=r'$W_t^{\mathrm{proto}}$',
    w_before_mparc=r'$W_{\mathrm{in}}^{\mathrm{MPARC}}$', w_after_mparc=r'$W_t^{\mathrm{PS}}$',
    delta_w_mparc=r'$\Delta W_{\mathrm{MPARC}}$',
    teacher_pseudo_boundary='Teacher pseudo boundary',
    edge_supervision='Positive-boundary\nsupervision weight')


def read_node(path):
    with np.load(path, allow_pickle=False) as saved:
        maps = {key: saved[key] for key in saved.files}
    maps['description'] = json.loads(str(maps['metadata']))
    return maps


def capture_label(info):
    return {'online_after_train_iter': 'Online fixed probe at recorded iteration',
            'checkpoint_replay': 'checkpoint_replay | restored historical checkpoint',
            'matched_checkpoint': 'Matched-checkpoint probe'}.get(
                info.get('capture_mode'), 'Legacy probe | capture timing unrecorded')


def active_analysis(path):
    root = Path(path).resolve()
    pointer = root / 'active_resume.json'
    if pointer.exists():
        selected = (root / json.loads(pointer.read_text(encoding='utf-8'))['path']).resolve()
        if root not in selected.parents or not selected.is_dir():
            raise ValueError('Invalid active resume analysis directory')
        return selected
    return root


TITLES.update(r_veg='Vegetation response R_veg',
              q_goji_given_veg='Conditional Goji response Q_G|veg',
              s_phy_class_agreement='Class-conditional agreement S_phy',
              g_phy='Physiological gate G_phy', attenuation_phy='Relative attenuation A_phy = 1-G_phy')


def map_availability(maps, key):
    valid = maps['valid_mask'].astype(bool)
    available = maps.get({'r_adj': 'adjacency_available',
                          'r_prob': 'semantic_available'}.get(key, ''), valid)
    if key in ('s_phy_class_agreement', 'g_phy', 'a_phy', 'attenuation_phy'):
        label = maps.get('teacher_label', maps.get('pseudo_label'))
        if label is not None:
            available = np.asarray(available, dtype=bool) & np.isin(label, [0, 1])
        weight = maps.get('w_before_phy', maps.get('w_base'))
        if key != 's_phy_class_agreement' and weight is not None:
            available = np.asarray(available, dtype=bool) & (weight > 0)
    return valid & np.asarray(available, dtype=bool)


def panel(axis, maps, key):
    if key == 'target_image':
        artist = axis.imshow(maps[key], interpolation='nearest')
        view = json.loads(str(maps['metadata'])).get('input_view', [])
        axis.set_title('Target image' + ('\nIRRG' if view and view[0] == 'NIR'
                                        else '\nRGB' if view else ''))
    else:
        values = np.ma.masked_where(~map_availability(maps, key), maps[key])
        cmap = copy.copy(plt.get_cmap('RdBu_r' if key.startswith('delta_') else 'viridis'))
        cmap.set_bad('#d0d0d0')
        artist = axis.imshow(values, cmap=cmap, vmin=-1 if key.startswith('delta_') else 0,
                             vmax=1, interpolation='nearest')
        title = TITLES[key]
        if key == 'r_phy' and 'r_veg' in maps:
            title = TITLES['r_veg']
        info = json.loads(str(maps['metadata']))
        if key == 'w_pre_proto' and 'w_t_ps' in maps:
            title = TITLES['w_t_ps']
        module = {'w_t_proto': 'proto', 'w_after_phy': 'phy', 'w_t_phy': 'phy', 'r_struct': 'struct',
                  'boundary_weight': 'edge'}.get(key)
        state = info.get('states', {}).get(module)
        if key == 'r_pair' and info.get('edge_pair_prior_ready') is False:
            title += '\n(prior uninitialized)'
        if state and state != 'active':
            title += '\n(' + ('gate ' if module == 'phy' else '') + state + ')'
        elif key == 'w_t_proto' and not bool(maps['proto_active']):
            title += '\n(inactive)'
        axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    return artist


def node_modules(maps):
    info = json.loads(str(maps['metadata']))
    return info.get('modules', dict(phy='r_phy' in maps, struct='r_struct' in maps,
                                  proto='w_t_proto' in maps, edge='e_prob' in maps))


def weight_keys(maps):
    if 'w_t_proto' in maps:
        return ['w_conf', 'w_pre_proto' if 'w_pre_proto' in maps else 'w_t_ps',
                'w_t_proto', 'delta_w_proto']
    if 'w_t_phy' in maps and not node_modules(maps)['struct']:
        return ['w_base', 'w_t_phy', 'delta_w_phy']
    if 'w_t_ps' in maps:
        return ['w_base', 'w_t_ps', 'delta_w_ps']
    return []


def kinds(maps):
    enabled = node_modules(maps)
    result = []
    compact = json.loads(str(maps['metadata'])).get('physiology_compact', False)
    if enabled['phy'] and not compact: result.append(('phy', 'physiology'))
    if enabled['struct']: result.append(('struct', 'struct_components'))
    if enabled['proto'] or enabled['struct']: result.append(('weight', 'weight_stages'))
    if enabled['edge']: result.append(('edge', 'edge_components'))
    return result


def strip(maps, kind, completed):
    panels = dict(struct=['r_conf', 'r_loc', 'r_adj', 'r_prob', 'r_struct'],
                  phy=['r_phy', 'w_base', 'w_t_phy', 'delta_w_phy'],
                  edge=['e_prob', 'e_img', 'r_pair', 'r_edge', 'boundary_weight'])
    names = ['target_image'] + (weight_keys(maps) if kind == 'weight' else panels[kind])
    fig, axes = plt.subplots(1, len(names), figsize=(7.2, 2.05), constrained_layout=True)
    normal_axes = []
    for axis, name in zip(axes, names):
        artist = panel(axis, maps, name)
        if name.startswith('delta_'):
            fig.colorbar(artist, ax=axis, shrink=0.65, ticks=[-1, 0, 1])
        elif name != 'target_image':
            normal_axes.append(axis)
    fig.colorbar(normal_axes[0].images[0], ax=normal_axes, shrink=0.65, ticks=[0, 0.5, 1])
    info = json.loads(str(maps['metadata']))
    label = 'SYNTHETIC TEST DATA' if info.get('synthetic_fixture') else capture_label(info)
    fig.suptitle('{} | {} | Iter {:06d}'.format(label, info['sample_id'], completed), fontsize=8)
    fig.text(0.5, 0.015, 'Gray: invalid region or unavailable prior', ha='center', fontsize=6)
    return fig


def save_figure(fig, path, metadata, dpi=300, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        if path.suffix == '.png':
            from PIL import Image
            with Image.open(path) as image:
                if image.info.get('Description') != metadata:
                    raise ValueError('Existing image has different/missing provenance: ' + str(path))
                image.verify()
        return
    tags = {'Description': metadata} if path.suffix == '.png' else {'Subject': metadata}
    temporary = path.with_suffix(path.suffix + '.tmp')
    fig.savefig(temporary, format=path.suffix[1:], dpi=dpi, metadata=tags)
    temporary.replace(path)


def render_node(maps, output, completed, overwrite=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for kind, suffix in kinds(maps):
        fig = strip(maps, kind, completed)
        try:
            save_figure(fig, output / ('iter_{:06d}_{}.png'.format(completed, suffix)),
                        str(maps['metadata']), dpi=160, overwrite=overwrite)
        finally:
            plt.close(fig)


def online_kinds(maps):
    """Layouts depend on effective enabled branches, never configuration filenames."""
    enabled = node_modules(maps)
    mask = tuple(bool(enabled[k]) for k in ('phy','struct','proto','edge'))
    return {
        (False,False,False,False): (), (True,False,False,False): ('phys',),
        (False,True,False,False): ('struct',), (True,True,False,False): ('mparc',),
        (False,False,True,False): ('proto',), (False,False,False,True): ('edge',),
        (False,False,True,True): ('pbcr',), (True,True,True,True): ('mparc','pbcr')
    }[mask]


def phy_weight_statistics(maps):
    before = maps['w_before_phy']
    valid = maps.get('valid_mask',np.ones(before.shape,bool)).astype(bool)
    if not valid.any():
        raise ValueError('Physiology statistics require valid supervision pixels')
    before = before[valid]
    after = maps['w_after_phy'][valid]
    delta = maps['delta_w_phy'][valid]
    return dict(pre_gate_weight_mean=float(before.mean(dtype=np.float64)),
        pre_gate_weight_std=float(before.std(dtype=np.float64)),
        after_gate_weight_mean=float(after.mean(dtype=np.float64)),
        mean_delta=float(delta.mean(dtype=np.float64)),
        maximum_decrease=float(max(0,-delta.min())),
        changed_valid_pixels_ratio=float(np.count_nonzero(delta)/delta.size))


def online_plate(nodes, kind):
    if kind == 'struct' and len(nodes) == 1:
        from tools.doris_struct_evidence import prepare_struct_evidence
        from tools.struct_mechanism_panels import COMPONENTS, display_scales, panel_figure
        from tools.build_hphy_mechanism_figure import crop_arrays
        info = nodes[0].get('description', json.loads(str(nodes[0]['metadata'])))
        if info.get('struct_cfg') or info.get('gate'):
            arrays, metadata = prepare_struct_evidence(nodes[0], info)
            center = crop_arrays(arrays)
            return panel_figure(center, COMPONENTS, 'formal', display_scales(center, metadata, 'formal'), metadata)
    panels = dict(phys=('target_image','w_conf','r_phy','delta_w_phy'),
        struct=('target_image','r_conf','r_loc','r_adj','r_prob','r_struct'),
        mparc=('target_image','r_phy','r_struct','w_before_mparc','w_after_mparc','delta_w_mparc'),
        proto=('target_image','w_before_proto','w_after_proto','delta_w_proto'),
        edge=('target_image','teacher_pseudo_boundary','r_edge','edge_supervision'),
        pbcr=('target_image','w_before_proto','w_after_proto','r_edge','edge_supervision'))
    names = panels[kind]
    if kind == 'phys' and 's_phy_class_agreement' in nodes[0]:
        names = ('target_image', 'w_conf', 's_phy_class_agreement', 'attenuation_phy')
    if kind == 'mparc' and 's_phy_class_agreement' in nodes[0]:
        names = tuple('s_phy_class_agreement' if key == 'r_phy' else key for key in names)
    height, width = nodes[0]['valid_mask'].shape
    horizontal_bars = len(nodes) == 1 and 's_phy_class_agreement' in nodes[0]
    if horizontal_bars:
        fig, axes = plt.subplots(1, len(names), figsize=(3.6*len(names), 5.4), squeeze=False)
        fig.subplots_adjust(left=.025, right=.985, bottom=.31, top=.84, wspace=.18)
    elif kind == 'phys' and len(nodes) == 1:
        fig, axes = plt.subplots(1, 4, figsize=(14.4, 4.9), squeeze=False, constrained_layout=True)
    else:
        fig,axes=plt.subplots(len(nodes),len(names),figsize=(3.0*len(names)+1, (3.0*height/width+1)*len(nodes)+1),
                              squeeze=False,constrained_layout=True)
    normal,delta,edge=[],[],[]
    for row,(maps,row_axes) in enumerate(zip(nodes,axes)):
        meta=maps.get('description',json.loads(str(maps['metadata'])))
        maps=dict(maps)
        if kind=='mparc' and 'w_before_phy' in maps and 'w_after_struct' in maps:
            # Both arrays are actual captured inputs/outputs; do not invent a product gate.
            maps['w_before_mparc']=maps['w_before_phy']
            maps['w_after_mparc']=maps['w_after_struct']
            maps['delta_w_mparc']=maps['w_after_mparc']-maps['w_before_mparc']
        for column,(axis,key) in enumerate(zip(row_axes,names)):
            if key not in maps:
                axis.set_axis_off();axis.text(.5,.5,'N/A',ha='center',va='center',transform=axis.transAxes)
                axis.set_box_aspect(height/width)
                axis.set_title(TITLES.get(key,key),fontsize=10)
                continue
            artist=panel(axis,maps,key)
            if key == 'attenuation_phy':
                cmap = copy.copy(plt.get_cmap('magma'))
                cmap.set_bad('#d0d0d0')
                artist.set_cmap(cmap)
                artist.set_clim(0, 1-meta.get('gate_min_weight', .75))
            axis.title.set_fontsize(10)
            if horizontal_bars and kind == 'phys':
                axis.set_title('({}) {}'.format(chr(97+column), axis.get_title()), fontsize=10)
            if key=='target_image':
                if row:axis.set_visible(False)
            elif key.startswith('delta_'):delta.append(axis)
            elif key=='teacher_pseudo_boundary':artist.set_cmap('gray_r')
            elif key=='edge_supervision':
                scale=max(float(n.get('description',json.loads(str(n['metadata']))).get(
                    'edge_supervision_max',1)) for n in nodes)
                artist.set_clim(0,max(scale,1e-12));edge.append(axis)
            else:normal.append(axis)
        branches = {'phys':('phy',),'struct':('struct',),'proto':('proto',),'edge':('edge',),
                    'mparc':('phy','struct'),'pbcr':('proto','edge')}[kind]
        states=' | '.join(k+': '+meta.get('states',{}).get(k,'unknown') for k in branches)
        row_axes[1].set_ylabel('Iter {}\n{}'.format(meta['iter'],states),fontsize=8)
    if horizontal_bars:
        for axis in normal + delta + edge:
            artist = axis.images[0]
            minimum, maximum = artist.get_clim()
            box = axis.get_position()
            bar_axis = fig.add_axes([box.x0, .235, box.width, .025])
            bar = fig.colorbar(artist, cax=bar_axis, orientation='horizontal',
                              ticks=[minimum, (minimum+maximum)/2, maximum])
            bar.ax.tick_params(labelsize=8, length=3, pad=2)
            bar.set_label(axis.get_title().replace('\n', ' '), fontsize=8, labelpad=3)
    elif normal:
        fig.colorbar(normal[0].images[0],ax=normal,shrink=.6,ticks=[0,.5,1])
    if delta and not horizontal_bars:fig.colorbar(delta[0].images[0],ax=delta,shrink=.6,ticks=[-1,0,1])
    if edge and not horizontal_bars:
        scale=edge[0].images[0].get_clim()[1]
        fig.colorbar(edge[0].images[0],ax=edge,shrink=.6,ticks=[0,scale/2,scale])
    info=nodes[-1].get('description',json.loads(str(nodes[-1]['metadata'])))
    title=kind.upper()+' | '+info['sample_id']
    if info.get('experiment_name'):title=kind.upper()+' | '+info['experiment_name']+' | '+info['sample_id']
    if len(nodes)==1:title+=' | Iter {:06d}'.format(info['iter'])
    if info.get('synthetic_fixture'):title+=' | SYNTHETIC TEST DATA'
    fig.suptitle(title,fontsize=11)
    note=capture_label(info)+' | gray: invalid / unavailable | early inactive weights are unmodified'
    if kind == 'phys':
        note += '\nVI-driven response and pseudo-supervision weight adjustment'
        if all(key in nodes[-1] for key in ('w_before_phy','w_after_phy','delta_w_phy')):
            summary = phy_weight_statistics(nodes[-1])
            note += '\nPre-gate weight mean = {pre_gate_weight_mean:.4f} (spatial std = {pre_gate_weight_std:.1e}); after-gate weight mean = {after_gate_weight_mean:.4f}'.format(**summary)
            note += '\nMean ΔW = {mean_delta:.5f}; maximum decrease = {maximum_decrease:.5f}; changed valid pixels = {changed_valid_pixels_ratio:.1%}'.format(**summary)
    if info.get('historical_state_complete') is False:
        note+='\nPARTIAL replay | unavailable responses are N/A; see replay metadata'
    diagnostics=info.get('diagnostics',{})
    if kind in ('proto','pbcr') and 'prototype_difference' in diagnostics:
        diff=diagnostics['prototype_difference']
        note+='\nPrototype | max absolute delta={:.6g} | changed valid pixels={:.3%}'.format(
            diff['max_abs'],diff['changed_valid_ratio'])
    proto=info.get('prototype')
    if proto and kind in ('proto','pbcr'):
        note+='\nPrototype classes={} | target update={} | tau_proto={} | tau_conf={}'.format(
            proto['valid_classes'],proto['target_update_active'],proto['tau_proto'],proto['tau_conf_proto'])
    fig.supxlabel(note,fontsize=9)
    return fig


def export_spatial_maps(maps, output, completed, stamp, flat=False, subfolder='spatial'):
    """One original-grid PNG per saved map, with no axes, margins or resampling."""
    from PIL import Image, PngImagePlugin
    info = json.loads(str(maps['metadata']))
    shape = tuple(maps['valid_mask'].shape)
    original = tuple(info.get('original_shape',shape))
    crop = info.get('display_crop_xyxy')
    mapped_crop = bool(crop and len(crop)==4 and 0<=crop[0]<crop[2]<=original[1]
        and 0<=crop[1]<crop[3]<=original[0] and (crop[3]-crop[1],crop[2]-crop[0])==shape)
    if (original != shape and not mapped_crop) or maps['target_image'].shape[:2] != shape:
        raise ValueError('Export grid differs from original TIF; an explicit spatial mapping is required')
    statistics = spatial_statistics(maps)
    folder = Path(output) / subfolder
    if not flat:
        folder = folder / ('iter_{:06d}'.format(completed))
    folder.mkdir(parents=True, exist_ok=True)
    for key, value in maps.items():
        value = np.asarray(value)
        if key == 'target_image':
            if not np.isfinite(value).all():
                raise ValueError('Non-finite target image')
            pixels = (np.clip(value, 0, 1) * 255).round().astype(np.uint8)
        elif value.ndim != 2:
            continue
        elif key in CLASS_LABEL_KEYS:
            evidence = info.get('phy_evidence') or info.get('reference_evidence')
            palette = np.asarray(info.get('class_palette') or
                                 (evidence or {}).get('palette') or GOJI_PALETTE, dtype=np.uint8)
            if not np.isin(value, list(range(len(palette))) + [255]).all():
                raise ValueError('Invalid reduced class ids in ' + key)
            pixels = np.full((*shape, 3), 208, dtype=np.uint8)
            labeled = (value >= 0) & (value < len(palette))
            pixels[labeled] = palette[value[labeled].astype(np.int64)]
        elif key == 'phy_semantic_check':
            pixels = np.asarray(info['semantic_palette'],dtype=np.uint8)[value]
        elif value.dtype == bool:
            pixels = value.astype(np.uint8) * 255
        else:
            scale = max(float(info.get('edge_supervision_max', 1)), 1e-12) if key == 'edge_supervision' else 1
            scaled = (value + 1) / 2 if key.startswith('delta_') else value / scale
            style = info.get('spatial_styles', {}).get(key)
            if style:
                scaled = (value-style['vmin']) / (style['vmax']-style['vmin'])
            cmap = plt.get_cmap(style['cmap'] if style else
                               'RdBu_r' if key.startswith('delta_') else 'viridis')
            pixels = cmap(scaled, bytes=True)
            pixels[~map_availability(maps, key)] = [208, 208, 208, 255]
        path = folder / (key + '.png')
        if path.exists():
            save_figure(None, path, stamp)
            with Image.open(path) as image:
                if image.size != shape[::-1]:
                    raise ValueError('Existing spatial map has wrong dimensions')
                if key in CLASS_LABEL_KEYS and (image.mode != 'RGB' or
                        not np.array_equal(np.asarray(image), pixels)):
                    raise ValueError('Existing label PNG uses obsolete or different colors; '
                                     'render saved NPZs to a new output directory: ' + str(path))
            continue
        tags = PngImagePlugin.PngInfo()
        tags.add_text('Description', stamp)
        temporary = path.with_suffix('.png.tmp')
        Image.fromarray(pixels).save(temporary, format='PNG', pnginfo=tags)
        temporary.replace(path)
    metadata = dict(info, statistics=statistics, export_shape=list(shape),
                    render_provenance=json.loads(stamp),
                    interpolation='none; responses already cover the native input grid')
    path = folder / 'metadata.json'
    content = json.dumps(metadata, sort_keys=True, indent=2)
    if path.exists():
        if path.read_text(encoding='utf-8') != content:
            raise ValueError('Existing spatial metadata has different provenance')
        return
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(path)


def export_hphy_native_panels(maps, output, completed, stamp):
    """Two matched exports from the same saved responses, without resizing."""
    if 's_phy_class_agreement' not in maps or maps['valid_mask'].shape != (256, 256):
        return
    info = json.loads(str(maps['metadata']))
    maximum = 1-info.get('gate_min_weight', .75)
    styles = {key: dict(vmin=0, vmax=maximum, cmap='magma')
              for key in ('attenuation_phy', 'a_phy')}
    info.update(spatial_styles=styles, native_panel_version=1)
    full = dict(maps, metadata=np.array(json.dumps(info, sort_keys=True)))
    provenance = dict(json.loads(stamp), spatial_styles=styles, native_panel_version=1)
    export_spatial_maps(full, output, completed, json.dumps(provenance, sort_keys=True),
                        subfolder='full256')
    center = {key: value[30:226, 30:226].copy() if np.asarray(value).shape[:2] == (256, 256)
              else value for key, value in maps.items()}
    info.update(display_crop_xyxy=[30, 30, 226, 226],
                spatial_mapping='Exact center slice [30:226,30:226] of the full 256 grid')
    center['metadata'] = np.array(json.dumps(info, sort_keys=True))
    provenance.update(display_crop_xyxy=[30, 30, 226, 226])
    export_spatial_maps(center, output, completed, json.dumps(provenance, sort_keys=True),
                        subfolder='center196')


def render_online_node(maps, output, completed, array_sha256='', render_timing='after_train_iter',
                       dpi=600, font_size=9):
    """CPU-only PNGs from one already-computed probe; return only verified files."""
    info=json.loads(str(maps['metadata']))
    if int(info['iter'])!=completed:raise ValueError('PNG and NPZ iterations differ')
    if online_kinds(maps) == ('struct',):
        from tools.struct_mechanism_panels import render_struct_bundle
        folder = Path(output)/'struct_v4'/('iter_{:06d}'.format(completed))
        rendered = render_struct_bundle(maps, folder, source_sha256=array_sha256, dpi=dpi, font_size=font_size)
        return [rendered/'struct_formal_group_1x7.png']
    spatial_statistics(maps)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    files=[]
    for kind in online_kinds(maps):
        path=output/('iter_{:06d}_{}.png'.format(completed,kind))
        stamp=json.dumps(dict(info,figure_kind=kind,array_sha256=array_sha256,render_timing=render_timing),sort_keys=True)
        # Check before building the figure on resume.
        if path.exists():
            save_figure(None,path,stamp)
        else:
            fig=online_plate([maps],kind)
            if render_timing!='after_train_iter':
                fig.supxlabel(fig._supxlabel.get_text()+' | '+render_timing,fontsize=6)
            try:save_figure(fig,path,stamp,dpi=300)
            finally:plt.close(fig)
        from PIL import Image
        with Image.open(path) as png:
            if png.info.get('Description')!=stamp:raise ValueError('PNG provenance mismatch: '+str(path))
            png.verify()
        if not path.stat().st_size:raise IOError('Empty PNG: '+str(path))
        files.append(path)
    if files or not any(info.get('modules',{}).values()):
        stamp=json.dumps(dict(info,array_sha256=array_sha256,render_timing=render_timing),sort_keys=True)
        export_spatial_maps(maps,output,completed,stamp)
        export_hphy_native_panels(maps,output,completed,stamp)
    return files


def render_baseline_paper(source, output=None):
    """Export a native-grid Baseline comparison using the same Teacher and GT conventions."""
    from matplotlib.patches import Patch
    from PIL import Image, TiffImagePlugin
    try:
        from tools.check_doris_phy_semantics_7000 import (
            SEMANTIC_PALETTE, VEGETATION_GROUPS, vegetation_confusion_map)
    except ModuleNotFoundError:
        from check_doris_phy_semantics_7000 import (
            SEMANTIC_PALETTE, VEGETATION_GROUPS, vegetation_confusion_map)

    source = Path(source).resolve()
    maps = read_node(source)
    info = json.loads(str(maps['metadata']))
    if (info['capture_mode']!='checkpoint_replay' or any(info['modules'].values())
            or any(info['executed_modules'].values()) or not info['historical_state_complete']):
        raise ValueError('Expected a completely restored Baseline checkpoint replay')
    spatial_statistics(maps)
    if any(key in maps for key in ('r_phy','r_struct','r_edge','delta_w_phy','delta_w_proto')):
        raise ValueError('Baseline responses include an enabled-branch mechanism')
    valid = maps['valid_mask'].astype(bool)
    expected = np.zeros((256,256),bool)
    expected[30:226,30:226] = True
    if not np.array_equal(valid,expected):
        raise ValueError('Baseline comparison requires the same 196x196 valid crop as phy')
    reference = info['reference_evidence']
    if reference['class_names']!=['goji','vegetation','building','water','road']:
        raise ValueError('Unexpected Baseline validation class mapping')
    if not np.isin(maps['reference_label'],[0,1,2,3,4,255]).all():
        raise ValueError('Unexpected Baseline reference labels')
    cropped = {key:value[30:226,30:226].copy() for key,value in maps.items()
        if isinstance(value,np.ndarray) and value.ndim>=2 and value.shape[:2]==valid.shape}
    confusion = vegetation_confusion_map(maps)
    cropped['phy_semantic_check'] = confusion[30:226,30:226].copy()
    rows = []
    for code,name in enumerate(VEGETATION_GROUPS,1):
        confidence = maps['w_conf'][confusion==code].astype(np.float64)
        rows.append(dict(group=name,valid_pixels=int(confidence.size),
            w_conf_mean=float(confidence.mean()) if confidence.size else None,
            w_conf_std=float(confidence.std()) if confidence.size else None))
    settings = dict(layout='1x5',crop_xyxy=[30,30,226,226],export_shape=[196,196],
        original_shape=list(valid.shape),probability_range=[0,1],interpolation='nearest',
        prediction_network='EMA Teacher',semantic_palette=SEMANTIC_PALETTE,
        panel_keys=['target_image','w_conf','pseudo_label','reference_label','phy_semantic_check'],
        semantic_map_definition={'0':'Excluded','1':'GT Goji, Teacher Goji',
            '2':'GT Goji, Teacher other vegetation','3':'GT other vegetation, Teacher Goji',
            '4':'GT other vegetation, Teacher other vegetation'})
    provenance = dict(source_npz=str(source),source_npz_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        analyzer_sha256=hashlib.sha256((Path(__file__).parent/'check_doris_phy_semantics_7000.py').read_bytes()).hexdigest(),
        source_metadata=info,display=settings)
    stamp = json.dumps(provenance,sort_keys=True)
    output = Path(output).resolve() if output else source.parent/'paper'
    if output==source.parent or output in source.parents:
        raise ValueError('Baseline paper figures require a separate output directory')
    output.mkdir(parents=True,exist_ok=True)
    prefix = 'iter_{:06d}_baseline_paper'.format(info['iter'])
    paths = [output/(prefix+suffix) for suffix in ('.png','.pdf')]
    palette = np.asarray(reference['palette'],dtype=np.uint8)
    with plt.rc_context({'font.size':12,'axes.titlesize':14}):
        fig = plt.figure(figsize=(16.8,6.2))
        grid = fig.add_gridspec(1,5,left=.02,right=.985,bottom=.39,top=.90,wspace=.16)
        titles = ('(a) Target RGB','(b) Teacher confidence','(c) Teacher prediction',
            '(d) GT reference','(e) Goji–vegetation check')
        for index,key in enumerate(settings['panel_keys']):
            cell = grid[0,index].subgridspec(2,1,height_ratios=[1,.055],hspace=.08)
            axis,bar_axis = fig.add_subplot(cell[0]),fig.add_subplot(cell[1])
            axis.set_title(titles[index],loc='left',fontweight='bold',pad=8)
            axis.set_axis_off()
            bar_axis.set_axis_off()
            value = cropped[key]
            if key=='w_conf':
                artist = axis.imshow(value,cmap='viridis',vmin=0,vmax=1,interpolation='nearest')
                bar_axis.set_axis_on()
                colorbar = fig.colorbar(artist,cax=bar_axis,orientation='horizontal',ticks=[0,.5,1])
                colorbar.set_label('Teacher top-class probability',fontsize=12,labelpad=2)
            elif key in ('pseudo_label','reference_label'):
                pixels = np.full((*value.shape,3),208,np.uint8)
                labeled = (value>=0)&(value<len(palette))
                pixels[labeled] = palette[value[labeled].astype(np.int64)]
                axis.imshow(pixels,interpolation='nearest')
                bar_axis.text(.5,.5,'GojiDataset class colors',ha='center',va='center',fontsize=10)
            elif key=='phy_semantic_check':
                axis.imshow(np.asarray(SEMANTIC_PALETTE,np.uint8)[value],interpolation='nearest')
                bar_axis.legend(handles=[Patch(facecolor=np.asarray(color)/255,label=label)
                    for color,label in zip(SEMANTIC_PALETTE[1:],
                        ('Goji → Goji','Goji → Veg.','Veg. → Goji','Veg. → Veg.'))],
                    loc='upper center',bbox_to_anchor=(.5,1.1),ncol=2,frameon=False,fontsize=9.2,
                    handlelength=.9,handletextpad=.3,columnspacing=.7)
            else:
                axis.imshow(value,interpolation='nearest')
                bar_axis.text(.5,.5,'196 × 196 pixels',ha='center',va='center')
        title = 'Baseline Teacher predictions | Goji vs. other vegetation | Iter {}'.format(info['iter'])
        if info.get('synthetic_fixture'):
            title += ' | SYNTHETIC TEST DATA'
        fig.suptitle(title,fontsize=15,y=.985)
        fig.text(.5,.27,'Valid supervision area 196 × 196 | rows 30–225, columns 30–225 of the 256 × 256 grid',ha='center',fontsize=11)
        for position,label,correct,error in ((.20,'GT Goji',rows[0],rows[1]),(.13,'GT other vegetation',rows[3],rows[2])):
            parts = []
            for name,row in (('Correct',correct),('Confused',error)):
                mean = '{:.3f}'.format(row['w_conf_mean']) if row['valid_pixels'] else 'N/A'
                parts.append('{} n = {:,}, mean confidence = {}'.format(name,row['valid_pixels'],mean))
            fig.text(.5,position,label+' | '+' | '.join(parts),ha='center',fontsize=11)
        fig.text(.5,.055,'(e) Labels show GT → Teacher prediction. Gray excludes other classes / ignored GT. Single-image descriptive comparison.',ha='center',fontsize=10.5)
    try:
        for path in paths:
            save_figure(fig,path,stamp,dpi=300)
    finally:
        plt.close(fig)
    cropped['metadata'] = np.array(json.dumps(dict(info,display_crop_xyxy=[30,30,226,226],
        semantic_palette=SEMANTIC_PALETTE),sort_keys=True))
    export_spatial_maps(cropped,output,info['iter'],stamp,flat=True)
    tif = output/'spatial/target_image.tif'
    tags = TiffImagePlugin.ImageFileDirectory_v2()
    tags[270] = stamp
    if tif.exists():
        with Image.open(tif) as image:
            if image.tag_v2.get(270)!=stamp:
                raise ValueError('Existing Baseline RGB TIF has different provenance')
    else:
        temporary = tif.with_suffix('.tif.tmp')
        Image.fromarray((cropped['target_image']*255).round().astype(np.uint8)).save(
            temporary,format='TIFF',compression='tiff_deflate',tiffinfo=tags)
        temporary.replace(tif)
    stream = io.StringIO()
    writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    summary = dict(data_source='checkpoint_replay',prediction_network='EMA Teacher',
        groups=rows,scope='GT and Teacher both in {Goji, Vegetation}; within-GT-class confusion counts',
        excluded_pixels=int((valid & (confusion==0)).sum()),provenance=provenance)
    for path,content in ((output/(prefix+'_metadata.json'),json.dumps(provenance,indent=2,sort_keys=True)),
            (output/'semantic_statistics.json',json.dumps(summary,indent=2,sort_keys=True)),
            (output/'semantic_statistics.csv',stream.getvalue())):
        if path.exists():
            if path.read_text(encoding='utf-8')!=content:
                raise ValueError('Existing Baseline paper metadata differs')
            continue
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(content,encoding='utf-8')
        temporary.replace(path)
    return paths


def phy_paper_plate(maps, weight_range=(.45, .60), delta_range=(-.10, 0.0)):
    if 's_phy_class_agreement' in maps:
        raise ValueError('H-Phy uses the four-panel online renderer; legacy paper plates require legacy arrays')
    """Five aligned panels, with a GT-based check of the saved gate responses."""
    from matplotlib.patches import Patch
    try:
        from tools.check_doris_phy_semantics_7000 import (
            SEMANTIC_PALETTE, vegetation_confusion_map, semantic_statistics)
    except ModuleNotFoundError:
        from check_doris_phy_semantics_7000 import (
            SEMANTIC_PALETTE, vegetation_confusion_map, semantic_statistics)

    spatial_statistics(maps)
    valid = maps['valid_mask'].astype(bool)
    rows, columns = np.where(valid)
    if not rows.size:
        raise ValueError('The saved response has no valid supervision pixels')
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(columns.min()), int(columns.max()) + 1
    if not valid[top:bottom, left:right].all():
        raise ValueError('A rectangular valid region is required for the common paper crop')
    names = ('target_image', 'w_conf', 'r_phy', 'w_before_phy', 'w_after_phy', 'delta_w_phy')
    cropped = {key:maps[key][top:bottom, left:right].copy() for key in names}
    for limits, keys in ((weight_range, names[3:5]), (delta_range, names[5:])):
        if len(limits) != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise ValueError('Display limits must be finite and increasing')
        if any(cropped[key].min() < limits[0] or cropped[key].max() > limits[1] for key in keys):
            raise ValueError('Display range excludes saved values; widen the shared limits explicitly')
    info = json.loads(str(maps['metadata']))
    height, width = cropped['w_conf'].shape
    summary = phy_weight_statistics(cropped)
    semantic = semantic_statistics(maps)
    cropped['phy_semantic_check'] = vegetation_confusion_map(maps)[top:bottom,left:right].copy()
    panel_keys = ('target_image','w_conf','r_phy','delta_w_phy','phy_semantic_check')
    settings = dict(crop_xyxy=[left, top, right, bottom], export_shape=[height, width],
        original_shape=list(valid.shape), crop_rule='Bounding rectangle of the saved valid_mask',
        weight_range=list(weight_range), delta_range=list(delta_range),
        probability_range=[0, 1], response_range=[0, 1], interpolation='nearest',
        layout='1x5', panel_keys=list(panel_keys), weight_statistics=summary,
        semantic_check=semantic, semantic_palette=SEMANTIC_PALETTE,
        analysis_focus='Goji versus other vegetation, compared within each GT class',
        response_definition='Actual physiology-head response with a shared positive label for Goji and other vegetation',
        semantic_map_definition={'0':'Excluded from comparison',
            '1':'GT Goji, Teacher Goji','2':'GT Goji, Teacher other vegetation',
            '3':'GT other vegetation, Teacher Goji','4':'GT other vegetation, Teacher other vegetation'},
        color_scaling='Linear display normalization; saved tensor values retained',
        colormaps=dict(probability='viridis', response='viridis', weight='viridis', delta='Blues_r'))
    titles = ('(a) Target RGB', '(b) Teacher confidence', '(c) VI-driven response '+r'$R_{\mathrm{phy}}$',
              '(d) Weight change '+r'$\Delta W$', '(e) Goji–vegetation check')
    limits = ((0,1), (0,1), (0,1), delta_range)
    labels = ('', 'Teacher top-class probability', 'Learned VI-head response',
              'After minus before (zoomed scale)')
    with plt.rc_context({'font.size':12, 'axes.titlesize':14}):
        fig = plt.figure(figsize=(16.8,6.2))
        grid = fig.add_gridspec(1,5,left=.02,right=.985,bottom=.39,top=.90,wspace=.16)
        for index, key in enumerate(panel_keys):
            cell = grid[0,index].subgridspec(2,1,height_ratios=[1,.055],hspace=.08)
            axis = fig.add_subplot(cell[0])
            bar_axis = fig.add_subplot(cell[1])
            axis.set_title(titles[index],loc='left',fontweight='bold',pad=8)
            axis.set_axis_off()
            if key == 'target_image':
                axis.imshow(cropped[key],interpolation='nearest')
                bar_axis.set_axis_off()
                bar_axis.text(.5,.5,'{} × {} pixels'.format(width,height),ha='center',va='center')
                continue
            if key == 'phy_semantic_check':
                axis.imshow(np.asarray(SEMANTIC_PALETTE,dtype=np.uint8)[cropped[key]],
                    interpolation='nearest')
                bar_axis.set_axis_off()
                bar_axis.legend(handles=[Patch(facecolor=np.asarray(color)/255,label=label)
                    for color,label in zip(SEMANTIC_PALETTE[1:],
                        ('Goji → Goji','Goji → Veg.','Veg. → Goji','Veg. → Veg.'))],
                    loc='upper center',bbox_to_anchor=(.5,1.1),ncol=2,frameon=False,fontsize=9.2,
                    handlelength=.9,handletextpad=.3,columnspacing=.7)
                continue
            minimum, maximum = limits[index]
            artist = axis.imshow(cropped[key],cmap='Blues_r' if key=='delta_w_phy' else 'viridis',
                                 vmin=minimum,vmax=maximum,interpolation='nearest')
            colorbar = fig.colorbar(artist,cax=bar_axis,orientation='horizontal',
                                   ticks=[minimum,(minimum+maximum)/2,maximum])
            colorbar.ax.tick_params(labelsize=12,length=3)
            colorbar.set_label(labels[index],fontsize=12,labelpad=2)
        title = 'VI-guided reweighting | Goji vs. other vegetation | Iter {}'.format(info['iter'])
        if info.get('synthetic_fixture'):
            title += ' | SYNTHETIC TEST DATA'
        fig.suptitle(title,fontsize=15,y=.985)
        fig.text(.5,.27,'Valid supervision area {} × {} | rows {}–{}, columns {}–{} of the {} × {} grid'.format(
            height,width,top,bottom-1,left,right-1,*valid.shape),ha='center',fontsize=11)
        fig.text(.5,.225,'Pre-gate weight mean = {pre_gate_weight_mean:.4f} (spatial std = {pre_gate_weight_std:.1e}) | after-gate weight mean = {after_gate_weight_mean:.4f}'.format(**summary),
            ha='center',fontsize=11)
        fig.text(.5,.18,'Mean ΔW = {mean_delta:.5f} | maximum decrease = {maximum_decrease:.5f} | changed valid pixels = {changed_valid_pixels_ratio:.1%}'.format(**summary),
            ha='center',fontsize=11)
        class_rows = semantic['vegetation_class_check']['groups']
        for position,label,correct,error in ((.135,'GT Goji',class_rows[0],class_rows[1]),
                                            (.09,'GT other vegetation',class_rows[3],class_rows[2])):
            pieces = []
            for name,row in (('Correct',correct),('Confused',error)):
                response = '{:.3f}'.format(row['r_phy_mean']) if row['valid_pixels'] else 'N/A'
                delta = '{:.5f}'.format(row['delta_w_phy_mean']) if row['valid_pixels'] else 'N/A'
                pieces.append('{} n = {:,}, mean R_phy = {}, mean ΔW = {}'.format(
                    name,row['valid_pixels'],response,delta))
            fig.text(.5,position,label+' | '+' | '.join(pieces),ha='center',fontsize=11)
        fig.text(.5,.045,'(e) Labels show GT → Teacher prediction. Gray excludes other classes / ignored GT. Single-image descriptive comparison.',
            ha='center',fontsize=10.5)
    return fig, cropped, settings


def render_phy_paper(source, output=None, weight_range=(.45,.60), delta_range=(-.10,0.0)):
    """Render saved responses into a separate paper folder with original-grid crop provenance."""
    from PIL import Image, PngImagePlugin, TiffImagePlugin
    try:
        from tools.check_doris_phy_semantics_7000 import export_semantic_check
    except ModuleNotFoundError:
        from check_doris_phy_semantics_7000 import export_semantic_check

    source = Path(source).resolve()
    maps = read_node(source)
    output = Path(output).resolve() if output else source.parent/'paper'
    if output == source.parent or output in source.parents:
        raise ValueError('Paper figures require a separate output directory')
    fig, cropped, settings = phy_paper_plate(maps,weight_range,delta_range)
    left, top, right, bottom = settings['crop_xyxy']
    exported = dict(cropped)
    exported.update({key:value[top:bottom,left:right].copy() for key,value in maps.items()
        if isinstance(value,np.ndarray) and value.ndim==2 and value.shape==maps['valid_mask'].shape})
    settings['spatial_exports'] = list(exported)
    settings['rgb_tif_role'] = 'Cropped RGB display image in original pixel coordinates'
    info = json.loads(str(maps['metadata']))
    provenance = dict(source_npz=str(source),source_npz_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        semantic_analyzer_sha256=hashlib.sha256((Path(__file__).parent/'check_doris_phy_semantics_7000.py').read_bytes()).hexdigest(),
        source_metadata=info,display=settings)
    stamp = json.dumps(provenance,sort_keys=True)
    output.mkdir(parents=True,exist_ok=True)
    prefix = 'iter_{:06d}_phys_paper'.format(info['iter'])
    metadata_path = output/(prefix+'_metadata.json')
    content = json.dumps(provenance,sort_keys=True,indent=2)
    try:
        if metadata_path.exists() and metadata_path.read_text(encoding='utf-8') != content:
            raise ValueError('Existing paper figure has different provenance; use a new --out-dir')
        export_semantic_check(source,output)
        paths = [output/(prefix+suffix) for suffix in ('.png','.pdf')]
        for path in paths:
            save_figure(fig,path,stamp,dpi=300)
        spatial = output/'spatial'
        spatial.mkdir(exist_ok=True)
        tags = PngImagePlugin.PngInfo()
        tags.add_text('Description',stamp)
        for key,value in exported.items():
            path = spatial/(key+'.png')
            if path.exists():
                save_figure(None,path,stamp)
                continue
            if key == 'target_image':
                pixels = (np.clip(value,0,1)*255).round().astype(np.uint8)
            elif key == 'phy_semantic_check':
                pixels = np.asarray(settings['semantic_palette'],dtype=np.uint8)[value]
            elif key in ('pseudo_label','reference_label'):
                palette = info.get('phy_evidence',{}).get('palette')
                if palette is None:
                    pixels = value.astype(np.uint8)
                else:
                    palette = np.asarray(palette,dtype=np.uint8)
                    pixels = np.full((*value.shape,3),208,dtype=np.uint8)
                    labeled = (value>=0) & (value<len(palette))
                    pixels[labeled] = palette[value[labeled].astype(np.int64)]
            elif value.dtype == bool:
                pixels = value.astype(np.uint8)*255
            else:
                minimum,maximum = settings['delta_range'] if key=='delta_w_phy' else (
                    settings['weight_range'] if key in ('w_before_phy','w_after_phy') else [0,1])
                pixels = plt.get_cmap('Blues_r' if key=='delta_w_phy' else 'viridis')(
                    (value-minimum)/(maximum-minimum),bytes=True)
            temporary = path.with_suffix('.png.tmp')
            Image.fromarray(pixels).save(temporary,format='PNG',pnginfo=tags)
            temporary.replace(path)
        tif_path = spatial/'target_image.tif'
        if tif_path.exists():
            with Image.open(tif_path) as image:
                if image.tag_v2.get(270) != stamp:
                    raise ValueError('Existing cropped RGB TIF has different provenance')
                image.verify()
        else:
            rgb = (np.clip(cropped['target_image'],0,1)*255).round().astype(np.uint8)
            tif_tags = TiffImagePlugin.ImageFileDirectory_v2()
            tif_tags[270] = stamp
            temporary = tif_path.with_suffix('.tif.tmp')
            Image.fromarray(rgb).save(temporary,format='TIFF',compression='tiff_deflate',tiffinfo=tif_tags)
            temporary.replace(tif_path)
        for destination in (metadata_path,spatial/'metadata.json'):
            if destination.exists():
                if destination.read_text(encoding='utf-8') != content:
                    raise ValueError('Existing cropped-map metadata has different provenance')
                continue
            temporary = destination.with_suffix('.json.tmp')
            temporary.write_text(content,encoding='utf-8')
            temporary.replace(destination)
    finally:
        plt.close(fig)
    return paths


def select_stages(nodes, manifest):
    iterations = [int(node['description']['iter']) for node in nodes]
    selected = {0, len(nodes) - 1}
    for threshold in (manifest.get('proto_target_update_start', 0),
                      manifest.get('proto_weight_start', 0),
                      manifest.get('phy_warmup_start'), manifest.get('struct_warmup_start')):
        if threshold is None:
            continue
        before = [i for i, step in enumerate(iterations) if step < threshold]
        after = [i for i, step in enumerate(iterations) if step >= threshold]
        if before: selected.add(before[-1])
        if after: selected.add(after[0])
    active = [i for i, node in enumerate(nodes) if bool(node['proto_active'])]
    if active:
        selected.add(active[0])
        if active[0]: selected.add(active[0] - 1)
    for i in (len(nodes) // 2, 3 * len(nodes) // 4, len(nodes) // 4):
        if len(selected) >= min(6, len(nodes)): break
        selected.add(i)
    # Keep first, final and actual activation transition when a very late cache
    # initialization creates more than six distinct requested stages.
    priority = [0, len(nodes) - 1] + ([active[0] - 1, active[0]] if active and active[0] else [])
    priority += sorted(selected)
    chosen = list(dict.fromkeys(i for i in priority if i >= 0))[:6]
    return [nodes[i] for i in sorted(chosen)]


def build_legacy_figures(analysis, output=None):
    analysis = Path(analysis)
    paths = sorted((analysis / 'fixed_probe/arrays').glob('iter_*.npz'))
    if not paths:
        raise ValueError('No saved fixed-probe arrays; no missing nodes will be invented')
    nodes = [read_node(path) for path in paths]
    manifest = json.loads((analysis / 'manifest.json').read_text(encoding='utf-8'))
    identities = {(n['description']['config_sha256'], n['description']['sample_id'],
                   n['description']['sample_tensor_sha256']) for n in nodes}
    if len(identities) != 1:
        raise ValueError('Snapshots do not describe one fixed sample and configuration')
    if any(n['description']['data_source'] != 'fixed_probe_recomputed' for n in nodes):
        raise ValueError('Image plates require fixed_probe_recomputed snapshots')
    if manifest.get('config_sha256') and manifest['config_sha256'] != nodes[-1]['description']['config_sha256']:
        raise ValueError('Snapshot and experiment manifest configurations differ')
    final = nodes[-1]
    structural = [n for n in nodes if n['valid_mask'].any() and
                  n.get('adjacency_available', np.array(False)).any() and
                  n.get('semantic_available', np.array(False)).any()]
    representative = structural[-1] if structural else final
    stages = select_stages(nodes, manifest)
    enabled = node_modules(final)
    selection_path = analysis / 'metrics/proto_selection_500.csv'
    selection, rows, active_rows, selection_step = [], [], [], None
    if enabled['proto']:
        with selection_path.open(newline='', encoding='utf-8') as stream:
            selection = list(csv.DictReader(stream))
        if not selection or any(row.get('data_source') != 'training_batch_actual' for row in selection):
            raise ValueError('Prototype selection figures require training_batch_actual CSV rows')
        active_rows = [row for row in selection if int(row['update_active'])]
        selection_step = max(int(row['iter']) for row in active_rows or selection)
        rows = sorted([r for r in selection if int(r['iter']) == selection_step], key=lambda r: int(r['class_id']))
    provenance = dict(modules=enabled, node_count=len(nodes), final_iter=final['description']['iter'],
        struct_iter=representative['description']['iter'],
        weight_iterations=[n['description']['iter'] for n in stages],
        selection_iter=selection_step, sample_id=final['description']['sample_id'],
        data_sources=['fixed_probe_recomputed'] + (['training_batch_actual'] if enabled['proto'] else []),
        synthetic_fixture=final['description'].get('synthetic_fixture', False),
        plot_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(b''.join(p.read_bytes() for p in paths) +
            (selection_path.read_bytes() if enabled['proto'] else b'')).hexdigest())
    folder = Path(output) if output else analysis / 'figures'
    if (folder / 'figure_selection.json').exists():
        previous = json.loads((folder / 'figure_selection.json').read_text(encoding='utf-8'))
        if previous != provenance:
            folder = folder / ('through_iter_{:06d}_'.format(provenance['final_iter']) +
                               provenance['source_sha256'][:8] + '_' + provenance['plot_code_sha256'][:8])
    folder.mkdir(parents=True, exist_ok=True)
    stamp = json.dumps(provenance, sort_keys=True)
    descriptions = folder / 'figure_selection.json'
    if descriptions.exists() and json.loads(descriptions.read_text()) != provenance:
        raise ValueError('Existing figure selection differs')
    descriptions.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    figures = []
    for kind, suffix in kinds(final):
        if kind != 'weight':
            figures.append((strip(representative if kind == 'struct' else final,
                                  kind, provenance['struct_iter'] if kind == 'struct' else
                                  provenance['final_iter']), 'fig_' + suffix))
    names = weight_keys(final)
    if not names and enabled['phy']:
        names = ['w_base', 'w_t_phy', 'delta_w_phy']
    if not names and enabled['edge']:
        names = ['e_prob', 'e_img', 'r_pair', 'r_edge', 'boundary_weight']
    fig_b, axes = plt.subplots(len(stages), len(names), squeeze=False,
        figsize=(7.2, 1.30 * len(stages) + 0.5), constrained_layout=True)
    for row_axes, maps in zip(axes, stages):
        for axis, key in zip(row_axes, names):
            panel(axis, maps, key)
        row_axes[0].set_ylabel('Iter {:06d}'.format(maps['description']['iter']))
    for is_delta in (False, True):
        group = [axes[row, col] for row in range(len(stages)) for col, name in enumerate(names)
                 if name.startswith('delta_') == is_delta]
        if group:
            fig_b.colorbar(group[0].images[0], ax=group, shrink=0.35,
                           ticks=[-1, 0, 1] if is_delta else [0, 0.5, 1])
    fig_b.suptitle('Fixed-probe evolution | ' + provenance['sample_id'] +
                  (' | SYNTHETIC TEST DATA' if provenance['synthetic_fixture'] else ''), fontsize=8)
    figures.append((fig_b, 'fig_weight_evolution' if weight_keys(final) or enabled['phy']
                    else 'fig_edge_evolution'))
    if enabled['proto']:
        fig_c, axis = plt.subplots(figsize=(7.2, 2.9), constrained_layout=True)
        x = np.arange(len(rows))
        denominator = np.array([max(int(r['predicted_count']), 1) for r in rows])
        axis.bar(x - 0.18, np.array([int(r['confidence_pass_count']) for r in rows]) / denominator,
                 width=0.36, color='#b4bcc3', label='Teacher confidence passes')
        axis.bar(x + 0.18, np.array([int(r['joint_selected_count']) for r in rows]) / denominator,
                 width=0.36, color='#4d7599', label='Confidence and prototype-input weight pass')
        axis.set_xticks(x)
        axis.set_xticklabels([r['class_name'] + ('\n(no candidates)' if int(r['predicted_count']) == 0 else '')
                             for r in rows])
        for label in axis.get_xticklabels()[:2]: label.set_fontweight('bold')
        axis.set_ylim(0, 1)
        axis.set_ylabel('Fraction of valid predicted class pixels')
        axis.legend(loc='upper right')
        axis.set_title(r'Prototype selection: $\tau_{proto}=$' + rows[0]['tau_proto'] +
                       r', $\tau_{conf}=$' + rows[0]['tau_conf_proto'] +
                       ' | Iter {:06d}'.format(selection_step))
        if not active_rows:
            axis.set_title(axis.get_title() + ' (updates inactive)')
        fig_c.supxlabel('Training-batch feature-grid counts; n={} sampled batches'.format(
            rows[0]['n_sampled_batches']) +
            (' | SYNTHETIC TEST DATA' if provenance['synthetic_fixture'] else ''), fontsize=7)
        figures.append((fig_c, 'fig_proto_selection'))
    for fig, name in figures:
        try:
            for extension in ('pdf', 'png'):
                save_figure(fig, folder / (name + '.' + extension), stamp)
        finally:
            plt.close(fig)
    return folder


def branch_stages(nodes, branch):
    steps = [int(n['description']['iter']) for n in nodes]
    requested = (2000,6000,10000,15000,20000)
    representative = {min(range(len(nodes)),key=lambda i:abs(steps[i]-r)) for r in requested}
    priority = []
    if branch == 'proto':
        active = [i for i,n in enumerate(nodes) if bool(n['proto_active'])]
        if active:
            priority = ([active[0]-1] if active[0] else []) + [active[0]]
    priority += [len(nodes)-1] + sorted(representative)
    chosen = sorted(list(dict.fromkeys(priority))[:5])
    return [nodes[i] for i in chosen]


def draw_vi_summary(fig, grid, path):
    rows = []
    if path.exists():
        with path.open(encoding='utf-8',newline='') as stream:
            rows = [r for r in csv.DictReader(stream) if r['image_group']=='ALL_BALANCED_GROUPS']
    for col,index in enumerate(('ndvi','ndre','cire')):
        axis = fig.add_subplot(grid[0,col])
        current = sorted([r for r in rows if r['index']==index],key=lambda r:int(r['class_id']))
        if current:
            for r in current:
                x=int(r['class_id'])
                if int(r['count']):
                    median=float(r['median']);q1=float(r['Q1']);q3=float(r['Q3'])
                    axis.errorbar(x,median,yerr=[[median-q1],[q3-median]],fmt='o',
                                  color='#547d8e',capsize=3,markersize=3)
                else:
                    axis.text(x,.5,'NA',ha='center',fontsize=6)
            axis.set_xticks(range(5))
            axis.set_xticklabels([r['class_name'] for r in current],rotation=25,ha='right',fontsize=6)
        else:
            axis.text(.5,.5,'Source VI summary unavailable',ha='center',transform=axis.transAxes,fontsize=6)
            axis.set_xticks([])
        axis.set_ylim(0,1)
        axis.set_title(index.upper()+' | source train median / IQR',fontsize=7)
    return bool(rows)


def branch_plate(nodes, branch, source_csv):
    if branch == 'struct':
        ready = [n for n in nodes if n.get('adjacency_available',np.array(False)).any()
                 and n.get('semantic_available',np.array(False)).any() and n['valid_mask'].any()]
        chosen = [ready[-1] if ready else nodes[-1]]
        if chosen[0]['description'].get('struct_cfg') or chosen[0]['description'].get('gate'):
            return online_plate(chosen, 'struct'), [chosen[0]['description']['iter']]
        names = ('target_image','r_conf','r_loc','r_adj','r_prob','r_struct')
        fig,axes = plt.subplots(1,6,figsize=(7.2,2.1),squeeze=False,constrained_layout=True)
    else:
        chosen = branch_stages(nodes,branch)
        names = (('target_image','r_phy','w_before_phy','w_after_phy','delta_w_phy') if branch=='phy' else
                 ('target_image','w_before_proto','w_after_proto','delta_w_proto') if branch=='proto' else
                 ('target_image','teacher_pseudo_boundary','r_edge','edge_supervision'))
        if branch=='phy':
            height=1.5*len(chosen)+2.4
            fig=plt.figure(figsize=(9,height))
            # Separate static evidence from square response panels; colorbars have reserved space.
            static=fig.add_gridspec(1,3,left=.07,right=.92,bottom=1-1.7/height,
                                    top=1-.5/height,wspace=.4)
            draw_vi_summary(fig,static,source_csv)
            grid=fig.add_gridspec(len(chosen),5,left=.06,right=.9,bottom=.035,
                                  top=1-2.4/height,wspace=.24,hspace=.55)
            axes=np.array([[fig.add_subplot(grid[row,col]) for col in range(5)]
                           for row in range(len(chosen))])
        else:
            fig,axes=plt.subplots(len(chosen),len(names),figsize=(7.2,1.35*len(chosen)+.6),
                                  squeeze=False,constrained_layout=True)
    normal,delta,edge_axes=[],[],[]
    for row,(row_axes,maps) in enumerate(zip(axes,chosen)):
        for axis,key in zip(row_axes,names):
            artist=panel(axis,maps,key)
            if key=='target_image':
                if row: axis.set_visible(False)
            elif key.startswith('delta_'):
                delta.append(axis)
            elif key=='edge_supervision':
                scale=float(maps['description'].get('edge_supervision_max',1))
                artist.set_clim(0,max(scale,1e-12));edge_axes.append(axis)
            elif key=='teacher_pseudo_boundary':
                artist.set_cmap('gray_r')
            else:
                normal.append(axis)
        if branch!='struct':
            state=maps['description'].get('states',{}).get(branch,'unknown')
            row_axes[1].set_ylabel('Iter {} | {}'.format(maps['description']['iter'],state),fontsize=6)
        if branch=='phy':
            mask=maps['m_phy'].astype(bool)&maps['valid_mask'].astype(bool)
            if mask.any() and not mask.all():row_axes[1].contour(mask,levels=[.5],colors='white',linewidths=.35)
    if branch=='phy':
        fig.colorbar(normal[0].images[0],cax=fig.add_axes([.93,.40,.012,.19]),ticks=[0,.5,1])
        fig.colorbar(delta[0].images[0],cax=fig.add_axes([.93,.11,.012,.19]),ticks=[-1,0,1])
    else:
        if normal:fig.colorbar(normal[0].images[0],ax=normal,shrink=.35,ticks=[0,.5,1])
        if delta:fig.colorbar(delta[0].images[0],ax=delta,shrink=.35,ticks=[-1,0,1])
    if edge_axes:
        scale=edge_axes[0].images[0].get_clim()[1]
        fig.colorbar(edge_axes[0].images[0],ax=edge_axes,shrink=.35,ticks=[0,scale/2,scale])
    titles=dict(phy='Physiology branch evidence',struct='Structural branch evidence',
                proto='Prototype branch evidence',edge='Boundary branch evidence')
    title=titles[branch]+' | '+nodes[-1]['description']['sample_id']
    if nodes[-1]['description'].get('synthetic_fixture'):title+=' | SYNTHETIC TEST DATA'
    if branch=='phy':fig.suptitle(title,fontsize=8,y=.99)
    else:fig.suptitle(title,fontsize=8)
    return fig,[n['description']['iter'] for n in chosen]


def build_branch_figures(analysis,output=None):
    analysis=Path(analysis)
    paths=sorted((analysis/'fixed_probe/selected_arrays').glob('iter_*.npz'))
    if not paths:raise ValueError('No saved branch arrays; missing nodes are never invented')
    nodes=[read_node(p) for p in paths]
    identities={(n['description']['config_sha256'],n['description']['sample_tensor_sha256'],
                 n['description']['sample_id']) for n in nodes}
    if len(identities)!=1 or any(n['description']['data_source']!='fixed_probe_recomputed' for n in nodes):
        raise ValueError('Branch evidence requires one fixed sample/configuration and probe data source')
    manifest=json.loads((analysis/'manifest.json').read_text(encoding='utf-8'))
    enabled=node_modules(nodes[-1])
    if manifest.get('config_sha256') and manifest['config_sha256']!=nodes[-1]['description']['config_sha256']:
        raise ValueError('Manifest and branch snapshots differ')
    vi=analysis/'physiology/data/phy_vi_statistics.csv'
    source=b''.join(p.read_bytes() for p in paths)+(vi.read_bytes() if enabled['phy'] and vi.exists() else b'')
    provenance=dict(modules=enabled,node_count=len(nodes),sample_id=nodes[-1]['description']['sample_id'],
        final_iter=nodes[-1]['description']['iter'],data_source='fixed_probe_recomputed',
        source_sha256=hashlib.sha256(source).hexdigest(),
        plot_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_vi_available=enabled['phy'] and vi.exists())
    folder=Path(output) if output else analysis/'figures'
    previous=folder/'branch_figure_selection.json'
    if previous.exists():
        old=json.loads(previous.read_text(encoding='utf-8'))
        if any(old.get(k)!=v for k,v in provenance.items()):
            folder=folder/('through_iter_{}_{}_{}'.format(provenance['final_iter'],
                provenance['source_sha256'][:8],provenance['plot_code_sha256'][:8]))
    folder.mkdir(parents=True,exist_ok=True)
    selected={}
    names=dict(phy='fig_phys_branch',struct='fig_struct_branch',proto='fig_proto_branch',edge='fig_edge_branch')
    stamp=json.dumps(provenance,sort_keys=True)
    for branch,active in enabled.items():
        if not active:continue
        fig,steps=branch_plate(nodes,branch,vi)
        try:
            for extension in ('pdf','png'):
                save_figure(fig,folder/(names[branch]+'.'+extension),stamp,dpi=300)
        finally:
            plt.close(fig)
        selected[branch]=steps
    provenance['selected_iterations']=selected
    (folder/'branch_figure_selection.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    return folder


def build_online_summaries(analysis,output=None,iterations=None):
    analysis=Path(analysis)
    paths=sorted((analysis/'fixed_probe/selected_arrays').glob('iter_*.npz'))
    if not paths:raise ValueError('No saved branch arrays; no historical nodes are invented')
    nodes=[read_node(p) for p in paths]
    identities={(n['description']['config_sha256'],n['description']['sample_tensor_sha256'],
                 n['description']['sample_id'],tuple(node_modules(n).items())) for n in nodes}
    if len(identities)!=1 or any(n['description']['data_source']!='fixed_probe_recomputed' for n in nodes):
        raise ValueError('Online summary requires one fixed sample, configuration and data source')
    manifest=json.loads((analysis/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('config_sha256') and manifest['config_sha256']!=nodes[-1]['description']['config_sha256']:
        raise ValueError('Manifest and saved probes differ')
    if iterations:
        saved={n['description']['iter']:n for n in nodes}
        if any(i not in saved for i in iterations):raise ValueError('Requested iteration has no saved probe')
        chosen=[saved[i] for i in sorted(set(iterations))]
    else:
        chosen=branch_stages(nodes,'proto' if node_modules(nodes[-1])['proto'] else 'phy')
    selection=dict(modules=node_modules(nodes[-1]),data_source='fixed_probe_recomputed',
        selected_iterations=[n['description']['iter'] for n in chosen],
        sample_id=nodes[-1]['description']['sample_id'],
        selected_array_paths=['fixed_probe/selected_arrays/iter_{:06d}.npz'.format(n['description']['iter']) for n in chosen],
        source_sha256=hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest(),
        plot_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    folder=Path(output) if output else analysis/'figures'
    previous=folder/'online_figure_selection.json'
    if previous.exists() and json.loads(previous.read_text())!=selection:
        folder=folder/('selection_'+hashlib.sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest()[:12])
    folder.mkdir(parents=True,exist_ok=True)
    stamp=json.dumps(selection,sort_keys=True)
    for kind in online_kinds(nodes[-1]):
        fig=online_plate(chosen,kind)
        try:
            for extension in ('pdf','png'):
                save_figure(fig,folder/('fig_{}_summary.{}'.format(kind,extension)),stamp,dpi=300)
        finally:plt.close(fig)
    (folder/'online_figure_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    return folder


def build_figures(analysis,output=None,iterations=None):
    analysis=active_analysis(analysis)
    manifest=json.loads((Path(analysis)/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('snapshot_policy',{}).get('online_plot') or iterations:
        return build_online_summaries(analysis,output,iterations)
    if manifest.get('output_mode')=='branches':
        return build_branch_figures(analysis,output)
    return build_legacy_figures(analysis,output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', nargs='+', help='one or more independent experiment analysis directories')
    parser.add_argument('--out-dir', help='optional destination for a single experiment')
    parser.add_argument('--iterations',type=int,nargs='+',help='exact saved iterations for the final summary')
    parser.add_argument('--snapshots-only',action='store_true',
                        help='render saved NPZs without inference; requires a separate --out-dir')
    parser.add_argument('--phy-paper',action='store_true',help='crop one saved physiology NPZ for paper display')
    parser.add_argument('--weight-range',type=float,nargs=2,default=(.45,.60),help='shared paper weight color limits')
    parser.add_argument('--delta-range',type=float,nargs=2,default=(-.10,0),help='paper difference color limits')
    args = parser.parse_args()
    if args.phy_paper:
        if len(args.analysis) != 1 or args.snapshots_only or args.iterations:
            parser.error('--phy-paper requires one NPZ and cannot be combined with summary options')
        paths = render_phy_paper(args.analysis[0],args.out_dir,args.weight_range,args.delta_range)
        print('Paper figures:',*[str(path) for path in paths],sep='\n')
        return
    if args.out_dir and len(args.analysis) != 1:
        parser.error('--out-dir requires exactly one experiment')
    if args.snapshots_only and not args.out_dir:
        parser.error('--snapshots-only requires a separate --out-dir to preserve training artifacts')
    for path in args.analysis:
        if args.snapshots_only:
            original=Path(path).resolve();analysis=active_analysis(path);output=Path(args.out_dir).resolve()
            if output==original or original in output.parents or output==analysis or analysis in output.parents:
                parser.error('Offline previews must be outside the original analysis directory')
            sources=sorted((analysis/'fixed_probe/selected_arrays').glob('iter_*.npz'))
            available={int(p.stem.split('_')[1]) for p in sources}
            if not sources or (args.iterations and not set(args.iterations).issubset(available)):
                parser.error('Requested historical node has no saved NPZ; no response can be invented')
            for source in sources:
                maps=read_node(source)
                if not args.iterations or maps['description']['iter'] in args.iterations:
                    render_online_node(maps,output,maps['description']['iter'],
                        array_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        render_timing='offline_from_saved_arrays')
            print('Saved-array previews:',output)
        else:
            print('Figures:', build_figures(path,args.out_dir,args.iterations))


if __name__ == '__main__':
    main()

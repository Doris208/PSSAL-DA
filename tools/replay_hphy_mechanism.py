"""Replay paired Baseline/H-Phy checkpoints and render independent RGB patches."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_hphy_mechanism_figure import (CROP, PALETTE, crop_arrays,
    file_sha256, output_path, roi_coordinates, validate_arrays, render, export_online_views)
from tools.replay_doris_mechanisms import (checkpoint_state,
                                         inference_signature, state_digest)
from tools.doris_probe_data import RGB30_DIRECTIONS, replay_direction

LOGGER = logging.getLogger('hphy_offline')

# Edit these defaults for the RGB26 -> RGB30 experiment.
CONFIG_PHY = ROOT / 'configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_phy_gate.py'
CONFIG_BASELINE = ROOT / 'configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi.py'
PHY_WORK_DIR = Path('/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_phy_gate')
BASELINE_WORK_DIR = Path('/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi')
CHECKPOINT_PHY = Path('latest.pth')
CHECKPOINT_BASELINE = Path('latest.pth')
TARGET_FILE = Path('/root/shared-nvme/mydataset/30/rgb_dir/val/3_1024_1280_1280_1536.tif')
TARGET_FILES = [
    TARGET_FILE,
    TARGET_FILE.with_name('3_256_1152_512_1408.tif'),
    TARGET_FILE.with_name('3_1536_512_1792_768.tif'),
    TARGET_FILE.with_name('3_1536_1024_1792_1280.tif'),
    TARGET_FILE.with_name('3_1792_1280_2048_1536.tif'),
    TARGET_FILE.with_name('3_1536_640_1792_896.tif'),
]
OUTPUT_ROOT = Path('/root/shared-nvme/work_dirs/doris_hphy_early4k/goji26RGB2goji30RGB/hphy_mechanism_26rgb2_30rgb')


def verify_protocol(config, physiology, structural=False):
    from tools.doris_probe_data import replay_direction
    direction = replay_direction(config)
    flags = tuple(bool(config.uda.get(key, {}).get('enable', False)) for key in
                  ('phy_cfg', 'struct_cfg', 'proto_cfg', 'struct_edge_cfg'))
    if flags != (physiology, structural, False, False) or config.uda.get('joint_gate_cfg', {}).get('enable', False):
        raise ValueError('Expected the requested Baseline, phy-only or struct-only configuration')
    if structural and (config.uda.struct_cfg.get('fusion_mode', 'legacy_product') not in
                       ('legacy_product', 'doris_confidence_guided', 'struct_distinct_v1') or
                       not config.uda.struct_cfg.get('use_target_gate', True)):
        raise ValueError('The structural checkpoint must use the enabled DORIS structural gate')
    if (config.runner.max_iters != 15000 or config.model.backbone.type != 'mit_b5' or
            config.model.backbone.in_chans != 6 or config.model.decode_head.num_classes != 5):
        raise ValueError('Both models must use MiT-B5, six channels, five classes and a 15K protocol')
    if physiology and (config.uda.phy_cfg.get('mode') != 'hierarchical_gv' or
                       not config.uda.phy_cfg.get('use_target_gate', True)):
        raise ValueError('The physiology checkpoint must use the enabled hierarchical H-Phy gate')
    for domain in ('source', 'target'):
        data = config.data.train[domain]
        if (list(data.vi_names) != ['ndvi', 'ndre', 'cire'] or not data.use_vi or
                not data.use_indices or not data.vi_as_input or data.vi_norm_for_input):
            raise ValueError('Missing or incompatible six-channel/independent VI input protocol')
    if config.uda.get('class_names') != ['goji', 'vegetation', 'building', 'water', 'road']:
        raise ValueError('Class order differs from the existing Goji mapping')
    return direction


def resolve_iteration(path, meta, explicit_iter=None):
    path = Path(path).resolve()
    stored = meta.get('iter')
    if stored is not None and type(stored) is not int:
        raise ValueError('Checkpoint meta.iter must be an integer')
    candidates, sources = [], []
    completed = meta.get('doris_checkpoint', {}).get('completed_iter')
    if completed is not None:
        if type(completed) is not int:
            raise ValueError('Checkpoint completed_iter must be an integer')
        candidates.append(completed)
        sources.append('doris_checkpoint.completed_iter')
    match = re.search(r'(?:^|_)iter_(\d+)\.pth$', path.name)
    if match:
        candidates.append(int(match[1]))
        sources.append('numbered filename')
    if not candidates:
        record_file = path.parent / 'checkpoint_best.json'
        if record_file.is_file():
            records = json.loads(record_file.read_text(encoding='utf-8'))
            for record in records.values():
                if not isinstance(record, dict) or not record.get('path'):
                    continue
                recorded = Path(record['path'])
                if not recorded.is_file():
                    recorded = path.parent / record['path'].replace('\\', '/').rsplit('/', 1)[-1]
                identical = (recorded.is_file() and (recorded.resolve() == path or
                    (recorded.stat().st_size == path.stat().st_size and file_sha256(recorded) == file_sha256(path))))
                if identical and type(record.get('iter')) is int:
                    candidates.append(record['iter'])
                    sources.append('checkpoint_best.json with verified file identity')
        if not candidates and stored is not None and path.is_file():
            # Limit hashing to the two counter interpretations, never the whole archive.
            for numbered in path.parent.glob('*iter_*.pth'):
                numbered_match = re.search(r'(?:^|_)iter_(\d+)\.pth$', numbered.name)
                if (numbered_match and int(numbered_match[1]) in (stored, stored + 1) and
                        numbered.stat().st_size == path.stat().st_size and file_sha256(numbered) == file_sha256(path)):
                    candidates.append(int(numbered_match[1]))
                    sources.append('byte-identical numbered checkpoint')
    if explicit_iter is not None:
        if type(explicit_iter) is not int:
            raise ValueError('Explicit iteration must be an integer')
        candidates.append(explicit_iter)
        sources.append('explicit --iter')
    if not candidates:
        raise ValueError('Cannot recover an unambiguous completed iteration; supply --iter after checking training records')
    if len(set(candidates)) != 1 or (stored is not None and stored not in (candidates[0], candidates[0]-1)):
        raise ValueError('Checkpoint iteration sources disagree with each other or meta.iter')
    completed = candidates[0]
    if not 0 < completed <= 15000:
        raise ValueError('Checkpoint iteration is outside the required new 15K protocol')
    return dict(completed_iter=completed, checkpoint_meta_iter=stored,
                iteration_source='; '.join(sources),
                mmcv_zero_based_counter_corrected=stored == completed - 1)


def data_protocol(config):
    """Compare configured data before/after GojiDataset appends its mask output."""
    data = deepcopy(dict(config['data']))
    for domain in ('source', 'target'):
        dataset = data['train'][domain]
        if dataset.get('type') == 'GojiDataset' and dataset.get('crop_pseudo_margins') is not None:
            collect = dataset['pipeline'][-1]
            if collect.get('type') == 'Collect':
                collect['keys'] = [key for key in collect['keys'] if key != 'valid_pseudo_mask']
    return data


def load_model(config, checkpoint_path, physiology, device, work_dir, explicit_iter=None,
               structural=False):
    from mmcv import Config
    from mmcv.runner import wrap_fp16_model
    from mmseg.models.builder import build_train_model
    from mmseg.models.uda.phy_reliability import (validate_phy_checkpoint,
        restore_phy_gate_config, phy_gate_parameters)

    digest = file_sha256(checkpoint_path)
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    meta = checkpoint.get('meta', {})
    if not meta.get('config'):
        raise ValueError('Checkpoint lacks its saved training configuration')
    if meta.get('iter') is None and meta.get('doris_checkpoint', {}).get('completed_iter') is None:
        raise ValueError('Checkpoint lacks iteration metadata to verify the completed step')
    historical = Config.fromstring(meta['config'], '.py')
    verify_protocol(historical, physiology, structural)
    runtime = deepcopy(config)
    schedule_info = {}
    if physiology:
        runtime.uda.phy_cfg = restore_phy_gate_config(config.uda.phy_cfg, historical.uda.phy_cfg)
        schedule_info = dict(phy_gate_config_source='checkpoint meta.config',
            requested_phy_gate=phy_gate_parameters(config.uda.phy_cfg),
            effective_phy_gate=phy_gate_parameters(runtime.uda.phy_cfg))
        schedule_info['historical_phy_schedule'] = (
            schedule_info['requested_phy_gate'] != schedule_info['effective_phy_gate'])
    if inference_signature(runtime) != inference_signature(historical):
        raise ValueError('Requested model/input/gate configuration differs from checkpoint provenance')
    if data_protocol(config) != data_protocol(historical):
        raise ValueError('Requested training protocol differs from checkpoint in data')
    for key in ('seed', 'optimizer', 'lr_config', 'fp16'):
        if config.get(key) != historical.get(key):
            raise ValueError('Requested training protocol differs from checkpoint in ' + key)
    iteration = resolve_iteration(checkpoint_path, meta, explicit_iter)
    runtime.model.pretrained = None
    runtime.model.backbone.pretrained = None
    runtime.model.train_cfg.work_dir = str(work_dir)
    model = build_train_model(runtime)
    state = checkpoint_state(checkpoint)
    if physiology:
        validate_phy_checkpoint('hierarchical_gv', state)
    expected = model.state_dict()
    missing, extra = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    if missing or extra:
        raise ValueError('Incomplete or incompatible checkpoint; missing={}, unexpected={}'.format(missing, extra))
    for key, value in state.items():
        if value.shape != expected[key].shape or (value.is_floating_point() and not torch.isfinite(value).all()):
            raise ValueError('Invalid checkpoint tensor: ' + key)
    model.load_state_dict(state, strict=True)
    model.local_iter = iteration['completed_iter']
    model.to(device).float().eval()
    model.replay_amp = device.type == 'cuda' and config.get('fp16') is not None
    if model.replay_amp:
        wrap_fp16_model(model)
    information = dict(iteration, path=str(Path(checkpoint_path).resolve()), sha256=digest,
        saved_config_sha256=hashlib.sha256(meta['config'].encode()).hexdigest(),
        synthetic_fixture=bool(meta.get('synthetic_fixture', False)), key_count=len(state),
        data_runtime_mask_normalized=config.data != historical.data,
        prediction_precision='configured CUDA AMP' if model.replay_amp else 'CPU/device fp32',
        physiology_precision='CUDA autocast fp16 head and fp32 gate' if device.type == 'cuda' else 'fp32 head and gate',
        hphy_probability_dtype='float16' if device.type == 'cuda' else 'float32', **schedule_info)
    del checkpoint, state, expected
    return model, information


def sample_config(config, target_file):
    from mmseg.utils.mechanism import deterministic_probe_config
    from tools.doris_probe_data import replay_direction
    probe = deterministic_probe_config(config)
    if probe.get('direction') != replay_direction(config):
        raise ValueError('Config does not resolve its supported 26-to-RGB30 direction')
    target_file = Path(target_file).resolve()
    if target_file.parent.parent.name != 'rgb_dir' or target_file.suffix.lower() != '.tif':
        raise ValueError('Target must be an RGB TIFF inside DATA_ROOT/rgb_dir/SPLIT/')
    probe['dataset'].update(data_root=str(target_file.parents[2]), img_dir=str(target_file.parent), split=None)
    probe.update(filename=target_file.name, expected_path=str(target_file),
                 selection_policy='explicit_single_target_native_256')
    return probe


def infer_model(model, sample, probe, physiology):
    from mmseg.utils.mechanism import probe_arrays, readonly_probe
    image, vi, metas = sample
    device = next(model.parameters()).device
    image, vi = image.to(device).float(), vi.to(device).float()
    if image.shape != (1, 6, 256, 256) or vi.shape != (1, 3, 256, 256):
        raise ValueError('Replay requires aligned full [1,6,256,256] image and [1,3,256,256] VI')
    before = state_digest(model.state_dict())
    arrays = {}
    with readonly_probe(model), torch.inference_mode():
        if physiology:
            spatial = image.new_ones((1, 1, 256, 256))
            top, bottom, left, right = probe['margins']
            if top: spatial[:, :, :top] = 0
            if bottom: spatial[:, :, -bottom:] = 0
            if left: spatial[:, :, :, :left] = 0
            if right: spatial[:, :, :, -right:] = 0
            maps = probe_arrays(model, image, vi, metas, spatial)
            for destination, source in (('R_veg', 'r_veg'), ('Q_G', 'q_goji_given_veg'),
                    ('P_G', 'p_goji'), ('P_V', 'p_other_veg'), ('P_N', 'p_nonveg'),
                    ('S_phy', 's_phy_class_agreement'), ('G_phy', 'g_phy'),
                    ('A_phy', 'attenuation_phy'), ('W_before', 'w_base'), ('W_after', 'w_t_phy'),
                    ('teacher_pseudo_label', 'pseudo_label'), ('teacher_confidence', 'w_conf'),
                    ('teacher_top2_margin', 'teacher_top2_margin'), ('valid_mask', 'valid_mask')):
                arrays[destination] = maps[source].copy()
            for key, cls in (('teacher_goji_mask', 0), ('teacher_vegetation_mask', 1)):
                arrays[key] = arrays['valid_mask'] & (arrays['teacher_pseudo_label'] == cls)
            arrays['teacher_phy_mask'] = arrays['teacher_goji_mask'] | arrays['teacher_vegetation_mask']
            arrays['positive_base_weight_mask'] = arrays['valid_mask'] & (arrays['W_before'] > 0)
            arrays['positive_phy_weight_mask'] = arrays['teacher_phy_mask'] & arrays['positive_base_weight_mask']
            arrays['gate_active_mask'] = arrays['positive_phy_weight_mask'] & (arrays['G_phy'] < 1)
        # UDADecorator.simple_test delegates to the Student in official evaluation.
        prediction = model.get_model()(img=[image], img_metas=[metas], return_loss=False, rescale=True)
        if len(prediction) != 1 or np.asarray(prediction[0]).shape != (256, 256):
            raise ValueError('Official Student prediction has the wrong shape')
        arrays['hphy_prediction' if physiology else 'baseline_prediction'] = np.asarray(prediction[0], dtype=np.uint8).copy()
    if state_digest(model.state_dict()) != before or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Read-only replay changed model state or created gradients')
    return arrays, before


def replay(config_phy, checkpoint_phy, config_baseline, checkpoint_baseline, target_file,
           out_dir, device='cuda:0', gt_file=None, selection_phy=None, selection_baseline=None, roi=None,
           iteration=None, iter_baseline=None):
    from mmcv import Config
    from mmseg.utils.mechanism import load_probe_sample, hphy_gate_metadata, student_test_metadata
    from mmseg.models.uda.phy_reliability import phy_gate_strength

    roi_coordinates(roi)
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu for a local CPU replay')
    cfg_phy, cfg_base = Config.fromfile(str(config_phy)), Config.fromfile(str(config_baseline))
    direction = verify_protocol(cfg_phy, True)
    verify_protocol(cfg_base, False)
    if data_protocol(cfg_phy) != data_protocol(cfg_base):
        raise ValueError('Baseline and H-Phy training protocol differs in data')
    for key in ('seed', 'optimizer', 'lr_config', 'fp16'):
        if cfg_phy.get(key) != cfg_base.get(key):
            raise ValueError('Baseline and H-Phy training protocol differs in ' + key)
    if cfg_phy.target_img_norm_cfg != cfg_base.target_img_norm_cfg:
        raise ValueError('Baseline and H-Phy target normalization differs')
    target_file = Path(target_file).resolve()
    probes = [sample_config(cfg, target_file) for cfg in (cfg_phy, cfg_base)]
    index_paths = {name: target_file.parents[2] / (name + cfg_phy.data.train.target.get('index_suffix', '_dir')) /
                   target_file.parent.name / target_file.name for name in ('ndvi', 'ndre', 'cire')}
    input_hashes = {str(path): file_sha256(path) for path in
                    [target_file, *index_paths.values(), Path(config_phy).resolve(), Path(config_baseline).resolve()]}
    if file_sha256(checkpoint_phy) == file_sha256(checkpoint_baseline):
        raise ValueError('Two independent Baseline and H-Phy checkpoints are required')
    out = output_path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'logs').mkdir()
    handler = logging.FileHandler(str(out / 'logs/replay.log'), encoding='utf-8')
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    try:
        arrays, infos, reference_sample = {}, [], None
        for physiology, cfg, checkpoint, probe in zip((True, False), (cfg_phy, cfg_base),
                (checkpoint_phy, checkpoint_baseline), probes):
            LOGGER.info('Loading %s checkpoint %s', 'H-Phy' if physiology else 'Baseline', checkpoint)
            model, information = load_model(cfg, checkpoint, physiology, device, out,
                                           iteration if physiology else iter_baseline)
            sample, _ = load_probe_sample(probe, model)
            if reference_sample is None:
                reference_sample = sample
            elif not all(torch.equal(left, right) for left, right in zip(reference_sample[:2], sample[:2])):
                raise ValueError('The two models received different six-channel or VI inputs')
            result, digest = infer_model(model, sample, probe, physiology)
            information.update(state_sha256_before=digest, state_sha256_after=digest,
                               eval=True, inference_mode=True, input_shapes=[list(sample[0].shape), list(sample[1].shape)])
            infos.append(information)
            arrays.update(result)
            if physiology:
                information['student_inference'] = student_test_metadata(model, sample[0], sample[2])
                information['student_inference']['prediction_field'] = 'hphy_prediction'
                image, vi, metas = sample
                norm = metas[0]['img_norm_cfg']
                original = image[0].numpy().transpose(1, 2, 0) * np.asarray(norm['std']) + np.asarray(norm['mean'])
                with Image.open(target_file) as tif:
                    if tif.mode != 'RGB':
                        raise ValueError('Target must be a true three-band uint8 RGB TIFF')
                    rgb = np.asarray(tif).copy()
                if rgb.shape != (256, 256, 3) or not np.allclose(original[..., :3], rgb, atol=1e-3, rtol=0):
                    raise ValueError('Decoded target RGB and the normalized model input differ')
                arrays.update(image_rgb=rgb, image_6ch=image[0].numpy().transpose(1, 2, 0).copy())
                for index, name in enumerate(('ndvi', 'ndre', 'cire')):
                    arrays[name] = original[..., index+3].astype(np.float32)
                    arrays['vi_' + name] = vi[0, index].numpy().copy()
                gate = dict(min_weight=model.phy_min_weight, warmup_iters=model.phy_warmup_iters,
                            rampup_iters=model.phy_rampup_iters, enabled=model.use_target_gate,
                            rule_iter=model.local_iter, implementation='DACS._apply_phy_gate -> hierarchical_phy_gate')
                gate.update(model.phy_gate_params)
                gate['rho'] = phy_gate_strength(model.local_iter, **model.phy_gate_params) if model.use_target_gate else 0.
                gate_metadata = hphy_gate_metadata(model, arrays['S_phy'], arrays['G_phy'],
                    arrays['W_before'], arrays['valid_mask'], arrays['teacher_pseudo_label'])
                threshold = model.pseudo_threshold
            if file_sha256(checkpoint) != information['sha256']:
                raise RuntimeError('Checkpoint changed during replay')
            del model, sample
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        gt_info = None
        if gt_file is not None:
            # GT is read only after both independent network forwards are complete.
            from mmseg.datasets.pipelines import LoadAnnotations
            gt = LoadAnnotations(reduce_zero_label=True)(dict(
                ann_info=dict(seg_map=str(Path(gt_file).resolve())), seg_fields=[]))['gt_semantic_seg']
            if gt.shape != (256, 256) or not np.isin(gt, [0, 1, 2, 3, 4, 255]).all():
                raise ValueError('GT differs from the full grid or standard reduced label mapping')
            arrays['gt_label'] = gt.copy()
            gt_info = dict(path=str(Path(gt_file).resolve()), sha256=file_sha256(gt_file),
                           usage='Offline statistics after both model forwards')
        for path, digest in input_hashes.items():
            if file_sha256(path) != digest:
                raise RuntimeError('Target, VI or configuration input changed during replay')
        for info, selection in zip(infos, (selection_phy, selection_baseline)):
            name = Path(info['path']).name
            info['selection_rule'] = selection or ('best Goji IoU' if name.startswith('best_goji') else
                                                   'best mIoU' if name.startswith('best_mIoU') else 'explicit checkpoint chosen by author')
        source_files = ['tools/replay_hphy_mechanism.py', 'mmseg/models/uda/dacs.py',
            'mmseg/models/uda/phy_reliability.py', 'mmseg/utils/mechanism.py',
            'mmseg/datasets/goji.py', 'mmseg/datasets/goji_meta.py',
            'mmseg/datasets/pipelines/loading.py', 'mmseg/datasets/pipelines/transforms.py',
            'mmseg/models/uda/uda_decorator.py', 'mmseg/vi_utils.py',
            'mmseg/models/segmentors/hrda_encoder_decoder.py', 'mmseg/models/decode_heads/hrda_head.py',
            'tools/replay_doris_mechanisms.py', 'tools/build_hphy_mechanism_figure.py',
            'tools/build_doris_mechanism_figures.py', 'tools/doris_probe_data.py']
        meta = dict(direction=direction, sample_id=target_file.name,
            target_file=str(target_file), target_sha256=input_hashes[str(target_file)],
            vi_files={key: dict(path=str(path), sha256=input_hashes[str(path)]) for key, path in index_paths.items()},
            gt=gt_info, checkpoint_phy=infos[0], checkpoint_baseline=infos[1],
            config_phy=dict(path=str(Path(config_phy).resolve()), sha256=input_hashes[str(Path(config_phy).resolve())],
                            resolved_sha256=hashlib.sha256(cfg_phy.pretty_text.encode()).hexdigest()),
            config_baseline=dict(path=str(Path(config_baseline).resolve()), sha256=input_hashes[str(Path(config_baseline).resolve())],
                                 resolved_sha256=hashlib.sha256(cfg_base.pretty_text.encode()).hexdigest()),
            teacher_network='H-Phy EMA Teacher', prediction_network='Student',
            capture_mode='checkpoint_replay', prediction_mode='test',
            student_inference=infos[0]['student_inference'],
            comparison='independently selected models; checkpoint iterations may differ',
            input_channel_order=['R', 'G', 'B', 'NDVI', 'NDRE', 'CIRE'],
            image_6ch_definition='Normalized segmentation input in HWC order',
            vi_normalization=dict(cfg_phy.data.train.target.vi_norm_cfg),
            vi_definition='ndvi/ndre/cire are pre-VI-normalization model inputs; vi_* are the normalized H-Phy inputs',
            target_normalization=dict(cfg_phy.target_img_norm_cfg),
            reduce_zero_label=True, ignore_index=255, class_names=list(cfg_phy.uda.class_names),
            class_palette=PALETTE.tolist(), original_shape=[256, 256], crop=list(CROP), center_shape=[196, 196],
            roi=dict(coordinate_system='original 256 grid, exclusive x2/y2', requested=roi,
                     translated_center196=roi_coordinates(roi), shift=[-30, -30]),
            render_settings_file='render_meta.json', gate=gate, pseudo_threshold=threshold,
            response_implementation='mmseg.utils.mechanism.probe_arrays shared with online probes',
            colorbars=dict(S_phy=[0, 1], confidence=[0, 1], attenuation=[0, 1-gate['min_weight']]),
            mask_convention='Actual neutral values plus independent applicability masks; gray pixels are excluded from statistics',
            weight_definition='DACS batch-wide confidence fraction recomputed on this B=1 full-grid replay; original training-batch weight is unavailable',
            python=sys.version, torch=torch.__version__,
            git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip(),
            source_sha256={name: file_sha256(ROOT / name) for name in source_files},
            synthetic_fixture=any(info['synthetic_fixture'] for info in infos))
        meta.update(gate_metadata)
        meta['mask_definitions'] = dict(teacher_phy_mask='valid Teacher Goji/Vegetation pixels',
            positive_base_weight_mask='valid pixels with W_before > 0',
            positive_phy_weight_mask='Teacher physiology scope with positive base weight',
            gate_active_mask='positive_phy_weight_mask and G_phy < 1',
            gt_ignore='optional offline reference only, never used in the gate')
        validate_arrays(arrays, meta)
        np.savez_compressed(str(out / 'responses_full_256.npz'), **arrays)
        np.savez_compressed(str(out / 'responses_center_196.npz'), **crop_arrays(arrays))
        meta['arrays_sha256'] = {name: file_sha256(out / name) for name in
                                ('responses_full_256.npz', 'responses_center_196.npz')}
        (out / 'mechanism_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        (out / 'phy_mechanism_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        LOGGER.info('Read-only replay completed; teacher=%s, official predictions=Student', meta['teacher_network'])
        return out
    except Exception:
        LOGGER.exception('H-Phy replay failed')
        raise
    finally:
        LOGGER.removeHandler(handler)
        handler.close()


def resolve_cli_inputs(args, branch, default_work_dir, default_output_root,
                       baseline_default=None, baseline_work_dir=None):
    """Select paired configs and relative checkpoint roots for the actual direction."""
    from mmcv import Config
    key = 'config_' + branch
    if args[key] is None:
        direction = args.get('direction') or RGB30_DIRECTIONS[0]
        args[key] = ROOT / 'configs/hrda/doris' / direction / (
            'doris_' + direction + '_hrda_3vi_' + branch + '_gate.py')
    config = Config.fromfile(str(args[key]))
    direction = replay_direction(config)
    requested = args.pop('direction', None)
    if requested is not None and requested != direction:
        raise ValueError('--direction differs from the resolved configuration')
    if args['config_baseline'] is None:
        args['config_baseline'] = (baseline_default or CONFIG_BASELINE) if direction == RGB30_DIRECTIONS[0] else (
            ROOT / 'configs/hrda/doris' / direction / ('doris_' + direction + '_hrda_3vi.py'))
    baseline = Config.fromfile(str(args['config_baseline']))
    if replay_direction(baseline) != direction:
        raise ValueError('Baseline and branch checkpoints must belong to the same direction')
    legacy = direction == RGB30_DIRECTIONS[0]
    args['checkpoint_' + branch] = (default_work_dir if legacy else Path(config.work_dir)) / args['checkpoint_' + branch]
    args['checkpoint_baseline'] = ((baseline_work_dir or BASELINE_WORK_DIR) if legacy else
                                  Path(baseline.work_dir)) / args['checkpoint_baseline']
    if args['target_file'] is None:
        filenames = [path.name for path in TARGET_FILES]
        if not legacy:
            filenames = filenames[-1:] + filenames[:-1]
        validation = config.data.val
        args['target_file'] = TARGET_FILES if legacy else [
            Path(validation.data_root) / validation.img_dir / name for name in filenames]
    if args['out_dir'] is None:
        root = default_output_root if legacy else Path(config.work_dir).parent / (branch + '_mechanism')
        args['out_dir'] = root / '{}_{}'.format(args['checkpoint_' + branch].stem,
            datetime.now().strftime('%Y%m%d_%H%M%S_%f'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direction', choices=RGB30_DIRECTIONS)
    parser.add_argument('--config-phy', type=Path)
    parser.add_argument('--config-baseline', type=Path)
    parser.add_argument('--checkpoint', '--checkpoint-phy', dest='checkpoint_phy',
                        type=Path, default=CHECKPOINT_PHY,
                        help='H-Phy checkpoint, absolute or relative to PHY_WORK_DIR')
    parser.add_argument('--checkpoint-baseline', type=Path, default=CHECKPOINT_BASELINE,
                        help='Baseline checkpoint, absolute or relative to BASELINE_WORK_DIR')
    for role in ('phy', 'baseline'):
        parser.add_argument('--selection-' + role)
    parser.add_argument('--target-file', type=Path, nargs='+',
                        help='One or more target TIFFs, processed independently with batch size one')
    parser.add_argument('--gt-file')
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--iter', dest='iteration', type=int, help='Explicit H-Phy completed iteration when records are ambiguous')
    parser.add_argument('--iter-baseline', type=int, help='Explicit Baseline completed iteration when records are ambiguous')
    parser.add_argument('--roi', nargs=4, type=int)
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--font-size', type=float, default=9)
    args = vars(parser.parse_args(argv))
    dpi, font_size = args.pop('dpi'), args.pop('font_size')
    if dpi <= 0 or font_size <= 0:
        parser.error('--dpi and --font-size must be positive')
    if args['config_phy'] is None and args['direction'] is None:
        args['config_phy'] = CONFIG_PHY
    resolve_cli_inputs(args, 'phy', PHY_WORK_DIR, OUTPUT_ROOT)
    target_files = args.pop('target_file')
    if len({path.stem for path in target_files}) != len(target_files):
        parser.error('Target image basenames must be unique')
    if len(target_files) > 1 and args['gt_file'] is not None:
        parser.error('--gt-file requires a single --target-file for aligned reference statistics')
    run_dir = output_path(args['out_dir'])
    if len(target_files) > 1:
        run_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    for target_file in target_files:
        image_dir = run_dir / target_file.stem if len(target_files) > 1 else run_dir
        result = replay(**dict(args, target_file=target_file, out_dir=image_dir))
        render(result / 'responses_full_256.npz', result / 'phy_mechanism_meta.json',
               result, roi=args['roi'], dpi=dpi, font_size=font_size)
        export_online_views(result / 'responses_full_256.npz', result / 'phy_mechanism_meta.json', result)
    print('Saved independent H-Phy responses and figures in ' + str(run_dir))


if __name__ == '__main__':
    main()

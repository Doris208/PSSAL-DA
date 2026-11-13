"""Replay paired Baseline/Struct checkpoints and export structural mechanism figures."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import replay_hphy_mechanism as shared
from tools.build_struct_mechanism_figure import (
    CROP, PALETTE, crop_arrays, export_online_views, file_sha256, output_path,
    render, roi_coordinates, validate_arrays)

LOGGER = logging.getLogger('struct_offline')

# Edit these defaults for the RGB26 -> RGB30 experiment.
CONFIG_STRUCT = ROOT / 'configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi_struct_gate.py'
CONFIG_BASELINE = shared.CONFIG_BASELINE
STRUCT_WORK_DIR = Path('/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26RGB2goji30RGB/goji26RGB2goji30RGB_hrda_3vi_struct_gate')
BASELINE_WORK_DIR = shared.BASELINE_WORK_DIR
CHECKPOINT_STRUCT = Path('iter_12000.pth')
CHECKPOINT_BASELINE = Path('iter_12000.pth')
TARGET_FILES = shared.TARGET_FILES
OUTPUT_ROOT = Path('/root/shared-nvme/work_dirs/doris_struct_distinct_v2/goji26RGB2goji30RGB/struct_mechanism_26rgb2_30rgb')


def infer_model(model, sample, probe):
    """Use the saved EMA priors and production gate without updating either."""
    from mmseg.utils.mechanism import readonly_probe
    image, vi, metas = sample
    device = next(model.parameters()).device
    image, vi = image.to(device).float(), vi.to(device).float()
    if image.shape != (1, 6, 256, 256) or vi.shape != (1, 3, 256, 256):
        raise ValueError('Replay requires aligned [1,6,256,256] image and [1,3,256,256] VI')
    before = shared.state_digest(model.state_dict())
    with readonly_probe(model), torch.inference_mode():
        spatial = image.new_ones((1, 1, 256, 256))
        top, bottom, left, right = probe['margins']
        if top: spatial[:, :, :top] = 0
        if bottom: spatial[:, :, -bottom:] = 0
        if left: spatial[:, :, :, :left] = 0
        if right: spatial[:, :, :, -right:] = 0
        with torch.cuda.amp.autocast(enabled=image.is_cuda):
            logits = model.get_ema_model().generate_pseudo_label(image, metas)
            if logits.shape != (1, 5, 256, 256):
                raise ValueError('Teacher logits must cover the full five-class input grid')
            probabilities = logits.softmax(1)
            confidence, label = probabilities.max(1)
            valid = model._doris_target_valid_mask(label, spatial)[:, 0]
            ratio = confidence.ge(model.pseudo_threshold).sum().item() / label.numel()
            weight = torch.ones_like(label, dtype=torch.float32) * ratio * valid
            after, responses = model.struct_gate.apply(
                target_probs=probabilities, pseudo_label=label, pseudo_weight=weight,
                target_confidence=confidence, iter=model.local_iter,
                collect_debug=True, log_stats=False, update_priors=False)
            top_two = probabilities.topk(2, dim=1).values
            tensors = dict(teacher_pseudo_label=label, teacher_confidence=confidence,
                teacher_top2_margin=top_two[:, 0]-top_two[:, 1],
                R_conf=confidence, R_loc=responses['R_continuity'], R_adj=responses['R_adj'],
                R_prob=responses['R_sem'], R_struct=responses['R_struct'],
                G_struct=responses['struct_gate'],
                W_before=weight, W_after=after, delta_W=after-weight, valid_mask=valid,
                adjacency_available=responses['adjacency_available'],
                semantic_available=responses['semantic_available'])
            for key in ('confidence_available', 'continuity_available', 'evidence_count',
                        'target_cross_neighbor_count', 'source_cross_prior_available'):
                if key in responses:
                    tensors[key] = responses[key]
            arrays = {key: value[0].detach().float().cpu().numpy().copy()
                      for key, value in tensors.items()}
        prediction = model.get_model()(img=[image], img_metas=[metas], return_loss=False, rescale=True)
        if len(prediction) != 1 or np.asarray(prediction[0]).shape != (256, 256):
            raise ValueError('Official Student prediction has the wrong shape')
        arrays['struct_prediction'] = np.asarray(prediction[0], dtype=np.uint8).copy()
    if shared.state_digest(model.state_dict()) != before or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Read-only replay changed model state or created gradients')
    for key in ('valid_mask', 'adjacency_available', 'semantic_available', 'confidence_available',
                'continuity_available', 'source_cross_prior_available'):
        if key not in arrays:
            continue
        arrays[key] = arrays[key].astype(bool)
    arrays['teacher_pseudo_label'] = arrays['teacher_pseudo_label'].astype(np.uint8)
    arrays['teacher_struct_mask'] = arrays['valid_mask'] & np.isin(
        arrays['teacher_pseudo_label'], model.struct_gate.struct_positive_classes)
    arrays['positive_base_weight_mask'] = arrays['valid_mask'] & (arrays['W_before'] > 0)
    arrays['positive_struct_weight_mask'] = arrays['positive_base_weight_mask'] & arrays['teacher_struct_mask']
    arrays['gate_active_mask'] = arrays['positive_struct_weight_mask'] & (arrays['G_struct'] < 1)
    arrays['below_threshold_mask'] = arrays['teacher_struct_mask'] & (
        arrays['R_struct'] < model.struct_gate.struct_filter_threshold)
    # Display-only ratio from the captured input/output weights, independent of gate internals.
    from tools.doris_struct_evidence import relative_struct_attenuation
    arrays['A_struct'], arrays['A_struct_valid_mask'] = relative_struct_attenuation(
        arrays['W_before'], arrays['W_after'], arrays['teacher_pseudo_label'], arrays['valid_mask'])
    return arrays, before


def replay(config_struct, checkpoint_struct, config_baseline, checkpoint_baseline, target_file,
           out_dir, device='cuda:0', gt_file=None, selection_struct=None, selection_baseline=None,
           roi=None, iteration=None, iter_baseline=None):
    from mmcv import Config
    from mmseg.utils.mechanism import load_probe_sample, module_states, student_test_metadata

    roi_coordinates(roi)
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu for a CPU replay')
    cfg_struct = Config.fromfile(str(config_struct))
    cfg_base = Config.fromfile(str(config_baseline))
    direction = shared.verify_protocol(cfg_struct, False, structural=True)
    shared.verify_protocol(cfg_base, False)
    if shared.data_protocol(cfg_struct) != shared.data_protocol(cfg_base):
        raise ValueError('Baseline and Struct training protocol differs in data')
    for key in ('seed', 'optimizer', 'lr_config', 'fp16', 'target_img_norm_cfg'):
        if cfg_struct.get(key) != cfg_base.get(key):
            raise ValueError('Baseline and Struct training protocol differs in ' + key)
    target_file = Path(target_file).resolve()
    probes = [shared.sample_config(cfg, target_file) for cfg in (cfg_struct, cfg_base)]
    index_paths = {name: target_file.parents[2] / (name + cfg_struct.data.train.target.get('index_suffix', '_dir')) /
                   target_file.parent.name / target_file.name for name in ('ndvi', 'ndre', 'cire')}
    hashes = {str(path): file_sha256(path) for path in
              [target_file, *index_paths.values(), Path(config_struct).resolve(), Path(config_baseline).resolve()]}
    if file_sha256(checkpoint_struct) == file_sha256(checkpoint_baseline):
        raise ValueError('Two independent Baseline and Struct checkpoints are required')
    out = output_path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'logs').mkdir()
    handler = logging.FileHandler(str(out / 'logs/replay.log'), encoding='utf-8')
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    try:
        arrays, infos, reference = {}, [], None
        for structural, cfg, checkpoint, probe in zip((True, False), (cfg_struct, cfg_base),
                (checkpoint_struct, checkpoint_baseline), probes):
            LOGGER.info('Loading %s checkpoint %s', 'Struct' if structural else 'Baseline', checkpoint)
            model, info = shared.load_model(cfg, checkpoint, False, device, out,
                iteration if structural else iter_baseline, structural=structural)
            for key in ('physiology_precision', 'hphy_probability_dtype'):
                info.pop(key, None)
            sample, _ = load_probe_sample(probe, model)
            if reference is None:
                reference = sample
            elif not all(torch.equal(a, b) for a, b in zip(reference[:2], sample[:2])):
                raise ValueError('The two models received different image or VI inputs')
            result, digest = (infer_model(model, sample, probe) if structural else
                              shared.infer_model(model, sample, probe, False))
            arrays.update(result)
            info.update(state_sha256_before=digest, state_sha256_after=digest,
                eval=True, inference_mode=True, input_shapes=[list(sample[0].shape), list(sample[1].shape)])
            if structural:
                info['struct_probability_dtype'] = 'float16' if device.type == 'cuda' else 'float32'
                info['student_inference'] = student_test_metadata(model, sample[0], sample[2])
            infos.append(info)
            if structural:
                image, vi, metas = sample
                norm = metas[0]['img_norm_cfg']
                original = image[0].numpy().transpose(1, 2, 0) * np.asarray(norm['std']) + np.asarray(norm['mean'])
                with Image.open(target_file) as tif:
                    if tif.mode != 'RGB':
                        raise ValueError('Target must be a true three-band uint8 RGB TIFF')
                    rgb = np.asarray(tif).copy()
                if rgb.shape != (256, 256, 3) or not np.allclose(original[..., :3], rgb, atol=1e-3, rtol=0):
                    raise ValueError('Decoded target RGB and model input differ')
                arrays.update(image_rgb=rgb, image_6ch=image[0].numpy().transpose(1, 2, 0).copy())
                for index, name in enumerate(('ndvi', 'ndre', 'cire')):
                    arrays[name] = original[..., index+3].astype(np.float32)
                    arrays['vi_' + name] = vi[0, index].numpy().copy()
                gate_model = model.struct_gate
                gate = {key: getattr(gate_model, key) for key in (
                    'fusion_mode', 'confidence_weight', 'continuity_weight', 'adjacency_weight', 'semantic_weight',
                    'adjacency_mode', 'semantic_score_mode', 'adj_cross_uniform_smoothing',
                    'adj_min_cross_mass', 'semantic_temperature', 'struct_gate_schedule',
                    'struct_gate_initial_strength', 'struct_gate_full_iter',
                    'confidence_power', 'continuity_power', 'adjacency_power', 'semantic_power',
                    'struct_min_weight', 'struct_filter_threshold', 'struct_filter_mode',
                    'struct_warmup_iters', 'struct_rampup_iters', 'struct_positive_classes',
                    'use_source_adjacency_ema', 'use_semantic_proto', 'use_target_gate', 'eps')}
                gate.update(rule_iter=model.local_iter, state=module_states(model)['struct'],
                    ramp=gate_model.gate_strength(model.local_iter),
                    implementation='StructPriorConsistencyGate.apply(update_priors=False)')
                priors = {key: getattr(gate_model, key).detach().cpu().numpy().copy() for key in (
                    'source_adj_ema', 'semantic_proto_ema', 'source_adj_initialized', 'semantic_proto_initialized')}
                threshold = model.pseudo_threshold
            if file_sha256(checkpoint) != info['sha256']:
                raise RuntimeError('Checkpoint changed during replay')
            del model, sample
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        gt_info = None
        if gt_file is not None:
            # Load reference labels only after both independent network forwards.
            from mmseg.datasets.pipelines import LoadAnnotations
            gt = LoadAnnotations(reduce_zero_label=True)(dict(
                ann_info=dict(seg_map=str(Path(gt_file).resolve())), seg_fields=[]))['gt_semantic_seg']
            if gt.shape != (256, 256) or not np.isin(gt, [0, 1, 2, 3, 4, 255]).all():
                raise ValueError('GT differs from the native grid or reduced Goji mapping')
            arrays['gt_label'] = gt.copy()
            gt_info = dict(path=str(Path(gt_file).resolve()), sha256=file_sha256(gt_file),
                           usage='Offline reference statistics after both model forwards')
        for path, digest in hashes.items():
            if file_sha256(path) != digest:
                raise RuntimeError('Image, VI or configuration changed during replay')
        for info, selection in zip(infos, (selection_struct, selection_baseline)):
            name = Path(info['path']).name
            info['selection_rule'] = selection or ('best Goji IoU' if name.startswith('best_goji') else
                'best mIoU' if name.startswith('best_mIoU') else 'explicit checkpoint chosen by author')
        source_files = ['tools/replay_struct_mechanism.py', 'tools/struct_schedule.py', 'tools/build_struct_mechanism_figure.py',
            'tools/doris_struct_evidence.py', 'tools/struct_mechanism_panels.py',
            'tools/replay_hphy_mechanism.py', 'tools/replay_doris_mechanisms.py',
            'tools/build_hphy_mechanism_figure.py', 'tools/build_doris_mechanism_figures.py',
            'tools/doris_probe_data.py', 'mmseg/models/uda/dacs.py', 'mmseg/models/uda/struct_prior_gate.py',
            'mmseg/utils/mechanism.py', 'mmseg/datasets/goji.py', 'mmseg/datasets/goji_meta.py',
            'mmseg/datasets/pipelines/loading.py', 'mmseg/datasets/pipelines/transforms.py', 'mmseg/vi_utils.py',
            'mmseg/models/uda/uda_decorator.py', 'mmseg/models/segmentors/hrda_encoder_decoder.py',
            'mmseg/models/decode_heads/hrda_head.py']
        meta = dict(direction=direction, sample_id=target_file.name,
            target_file=str(target_file), target_sha256=hashes[str(target_file)],
            vi_files={name: dict(path=str(path), sha256=hashes[str(path)]) for name, path in index_paths.items()},
            gt=gt_info, checkpoint_struct=infos[0], checkpoint_baseline=infos[1],
            config_struct=dict(path=str(Path(config_struct).resolve()), sha256=hashes[str(Path(config_struct).resolve())],
                resolved_sha256=hashlib.sha256(cfg_struct.pretty_text.encode()).hexdigest()),
            config_baseline=dict(path=str(Path(config_baseline).resolve()), sha256=hashes[str(Path(config_baseline).resolve())],
                resolved_sha256=hashlib.sha256(cfg_base.pretty_text.encode()).hexdigest()),
            teacher_network='Struct EMA Teacher', prediction_network='Student',
            capture_mode='checkpoint_replay', prediction_mode='test',
            prediction_identity='Student test-mode prediction',
            student_inference=infos[0]['student_inference'],
            prediction_checkpoint=dict(path=infos[0]['path'], sha256=infos[0]['sha256'],
                state_sha256=infos[0]['state_sha256_before'], iteration=infos[0]['completed_iter'],
                capture_mode='checkpoint_replay'),
            comparison='independently selected models; checkpoint iterations may differ',
            input_channel_order=['R', 'G', 'B', 'NDVI', 'NDRE', 'CIRE'],
            image_6ch_definition='Normalized segmentation input in HWC order',
            vi_normalization=dict(cfg_struct.data.train.target.vi_norm_cfg),
            vi_definition='ndvi/ndre/cire are pre-VI-normalization inputs; vi_* are normalized VI',
            target_normalization=dict(cfg_struct.target_img_norm_cfg),
            reduce_zero_label=True, ignore_index=255, class_names=list(cfg_struct.uda.class_names),
            class_palette=PALETTE.tolist(), original_shape=[256, 256], crop=list(CROP), center_shape=[196, 196],
            roi=dict(coordinate_system='original 256 grid, exclusive x2/y2', requested=roi,
                     translated_center196=roi_coordinates(roi)), gate=gate, pseudo_threshold=threshold,
            response_implementation='Production Teacher forward and StructPriorConsistencyGate.apply',
            response_formula='Recorded fusion_mode with production availability masks',
            weight_definition='DACS confidence fraction recomputed over this B=1 full-grid replay',
            replay_scope='Saved source EMA priors and target gate; training-only source-target feature interaction requires a paired training batch',
            mask_definitions=dict(teacher_struct_mask='valid Teacher predictions in configured structural classes',
                positive_struct_weight_mask='teacher_struct_mask and W_before > 0',
                gate_active_mask='positive_struct_weight_mask and G_struct < 1',
                adjacency_available='saved class initialization flag and adjacency enabled',
                semantic_available='saved class initialization flag and semantic prior enabled'),
            python=sys.version, torch=torch.__version__,
            git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip(),
            source_sha256={name: file_sha256(ROOT / name) for name in source_files},
            synthetic_fixture=any(info['synthetic_fixture'] for info in infos))
        from tools.doris_struct_evidence import prepare_struct_evidence
        enriched, meta = prepare_struct_evidence(arrays, meta)
        arrays.update(enriched)
        (out / 'struct_validation.json').write_text(json.dumps(meta['struct_validation'], indent=2), encoding='utf-8')
        if not meta['struct_validation']['passed']:
            np.savez_compressed(str(out / 'invalid_struct_responses.npz'), **arrays)
            raise ValueError('Structural diagnostic failed: '+', '.join(meta['struct_validation']['errors']))
        validate_arrays(arrays, meta)
        np.savez_compressed(str(out / 'responses_full_256.npz'), **arrays)
        np.savez_compressed(str(out / 'responses_center_196.npz'), **crop_arrays(arrays))
        np.savez_compressed(str(out / 'struct_priors.npz'), **priors)
        meta['arrays_sha256'] = {name: file_sha256(out / name) for name in (
            'responses_full_256.npz', 'responses_center_196.npz', 'struct_priors.npz')}
        meta['prior_initialization'] = {key: priors[key].tolist() for key in (
            'source_adj_initialized', 'semantic_proto_initialized')}
        for name in ('mechanism_meta.json', 'struct_mechanism_meta.json'):
            (out / name).write_text(json.dumps(meta, indent=2), encoding='utf-8')
        LOGGER.info('Read-only replay completed; Teacher structural responses and Student predictions saved')
        return out
    except Exception:
        LOGGER.exception('Struct replay failed')
        raise
    finally:
        LOGGER.removeHandler(handler)
        handler.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direction', choices=shared.RGB30_DIRECTIONS)
    parser.add_argument('--config-struct', type=Path)
    parser.add_argument('--config-baseline', type=Path)
    parser.add_argument('--checkpoint', '--checkpoint-struct', dest='checkpoint_struct', type=Path,
                        default=CHECKPOINT_STRUCT, help='Absolute path or relative to STRUCT_WORK_DIR')
    parser.add_argument('--checkpoint-baseline', type=Path, default=CHECKPOINT_BASELINE,
                        help='Absolute path or relative to BASELINE_WORK_DIR')
    for role in ('struct', 'baseline'):
        parser.add_argument('--selection-' + role)
    parser.add_argument('--target-file', type=Path, nargs='+',
                        help='One or more native 256 RGB TIFFs, each replayed independently')
    parser.add_argument('--gt-file')
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--iter', dest='iteration', type=int, help='Verified Struct iteration if checkpoint records are ambiguous')
    parser.add_argument('--iter-baseline', type=int)
    parser.add_argument('--roi', nargs=4, type=int)
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--font-size', type=float, default=9)
    args = vars(parser.parse_args(argv))
    dpi, font_size = args.pop('dpi'), args.pop('font_size')
    if dpi <= 0 or font_size <= 0:
        parser.error('--dpi and --font-size must be positive')
    roi_coordinates(args['roi'])
    if args['config_struct'] is None and args['direction'] is None:
        args['config_struct'] = CONFIG_STRUCT
        if args['target_file'] is None:
            args['target_file'] = TARGET_FILES
    shared.resolve_cli_inputs(args, 'struct', STRUCT_WORK_DIR, OUTPUT_ROOT,
                              CONFIG_BASELINE, BASELINE_WORK_DIR)
    targets = args.pop('target_file')
    if len({path.stem for path in targets}) != len(targets):
        parser.error('Target image basenames must be unique')
    if len(targets) > 1 and args['gt_file'] is not None:
        parser.error('--gt-file requires a single --target-file')
    run_dir = output_path(args['out_dir'])
    if len(targets) > 1:
        run_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    figure_folders = []
    for target in targets:
        folder = run_dir / target.stem if len(targets) > 1 else run_dir
        result = replay(**dict(args, target_file=target, out_dir=folder))
        figure_folders.append(render(result / 'responses_full_256.npz', result / 'struct_mechanism_meta.json',
               result, roi=args['roi'], dpi=dpi, font_size=font_size))
        export_online_views(result / 'responses_full_256.npz', result / 'struct_mechanism_meta.json', result,
                            dpi=dpi, font_size=font_size)
    from tools.struct_mechanism_panels import write_run_summary
    write_run_summary(figure_folders, run_dir)
    print('Saved independent Struct responses and figures in ' + str(run_dir))


if __name__ == '__main__':
    main()

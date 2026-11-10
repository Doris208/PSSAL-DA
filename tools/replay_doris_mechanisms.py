"""Audit checkpoints and replay saved DORIS states through the shared fixed probe."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROTOTYPE_BUFFERS = ('prototypes', 'valid_proto', 'proto_counts', 'source_prototypes',
                     'target_prototypes', 'source_valid_proto', 'target_valid_proto')
STRUCTURE_BUFFERS = ('source_adj_ema', 'source_adj_initialized',
                     'semantic_proto_ema', 'semantic_proto_initialized')
TRANSIENT_LIMITS = [
    'Original training-batch tensors, ClassMix/crop choices and per-batch prototype update counts are not replayed.',
    'Responses are new fixed-image forwards using the saved state and completed-iteration schedule.',
    'Training-only structural feature interaction is outside the existing fixed_probe response protocol.']


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def state_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_state(checkpoint):
    source = checkpoint.get('state_dict')
    if not isinstance(source, dict) or not source:
        raise ValueError('Checkpoint has no nonempty state_dict')
    state = {}
    for key, value in source.items():
        name = key[7:] if key.startswith('module.') else key
        if name in state or not torch.is_tensor(value):
            raise ValueError('Invalid or duplicate state entry: ' + name)
        state[name] = value
    return state


def checkpoint_iteration(path, meta):
    match = re.fullmatch(r'(?:best_.+_)?iter_(\d+)\.pth', Path(path).resolve().name)
    stored = meta.get('iter')
    if not match:
        raise ValueError('Completed iteration is ambiguous; latest must resolve or match a numbered checkpoint')
    completed = int(match[1])
    if not isinstance(stored, int) or isinstance(stored, bool) or stored not in (completed, completed - 1):
        raise ValueError('Checkpoint filename and meta.iter disagree: {} versus {}'.format(completed, stored))
    return dict(completed_iter=completed, checkpoint_meta_iter=stored,
                iteration_source='numbered_filename_verified_against_meta',
                mmcv_zero_based_counter_corrected=stored == completed - 1)


def audit_checkpoints(work_dir, recursive=False):
    work_dir = Path(work_dir).resolve()
    paths = sorted(work_dir.rglob('*.pth') if recursive else work_dir.glob('*.pth'))
    records = []
    for path in paths:
        row = dict(path=str(path), bytes=path.stat().st_size, completed_iter=None,
                   filename_iter=None, status='unreadable')
        match = re.fullmatch(r'(?:best_.+_)?iter_(\d+)\.pth', path.name)
        if match:
            row['filename_iter'] = int(match[1])
        row['checkpoint_sha256'] = file_digest(path)
        try:
            checkpoint = torch.load(path, map_location='cpu')
            state = checkpoint_state(checkpoint)
            meta = checkpoint.get('meta', {})
            row.update(status='readable', state_sha256=state_digest(state),
                checkpoint_meta_iter=meta.get('iter'), has_saved_config=bool(meta.get('config')),
                student_key_count=sum(k.startswith('model.') for k in state),
                teacher_key_count=sum(k.startswith('ema_model.') for k in state),
                network_completeness='requires comparison with the configured model')
            caches = ['proto_gate.' + k for k in PROTOTYPE_BUFFERS]
            caches += ['struct_gate.' + k for k in STRUCTURE_BUFFERS]
            caches += ['struct_edge_pair_prior', 'struct_edge_pair_prior_ready']
            row['history_buffers'] = {k: dict(present=k in state,
                shape=list(state[k].shape) if k in state else None) for k in caches}
            try:
                row.update(checkpoint_iteration(path, meta))
            except ValueError as error:
                row['iteration_error'] = str(error)
            del checkpoint, state
        except Exception as error:
            row['error'] = '{}: {}'.format(type(error).__name__, error)
        if file_digest(path) != row['checkpoint_sha256']:
            row.update(status='changed_during_read', completed_iter=None)
        records.append(row)
    # A regular-file latest.pth is only assigned an iteration when its bytes match.
    numbered = {r['checkpoint_sha256']: r for r in records if r['completed_iter'] is not None}
    for row in records:
        reference = numbered.get(row['checkpoint_sha256'])
        if row['status'] == 'readable' and row['completed_iter'] is None and reference:
            row.update(completed_iter=reference['completed_iter'],
                       iteration_source='byte_identical_to_numbered_checkpoint', alias_of=reference['path'])
            row.pop('iteration_error', None)
    return records


def select_checkpoints(records, iterations=None, pilot=False):
    by_iteration = {}
    for row in records:
        if row['status'] == 'readable' and row['completed_iter'] is not None:
            by_iteration.setdefault(row['completed_iter'], []).append(row)
    available = {}
    for step, rows in by_iteration.items():
        if len({r['state_sha256'] for r in rows}) != 1:
            for row in rows:
                row['selection_error'] = 'Conflicting model states at the same iteration'
            continue
        available[step] = min(rows, key=lambda r: (not Path(r['path']).name.startswith('iter_'), r['path']))
    if iterations is not None:
        missing = sorted(set(iterations) - set(available))
        if missing:
            raise ValueError('No readable, unambiguous checkpoint for iterations ' + str(missing))
        return [available[i] for i in sorted(set(iterations))]
    steps = sorted(available)
    if pilot:
        if len(steps) < 2:
            raise ValueError('Two distinct readable checkpoints are required for the pilot')
        middle = min(steps[:-1], key=lambda i: abs(i - steps[-1] / 2))
        steps = [middle, steps[-1]]
    if not steps:
        raise ValueError('No readable checkpoint with a verified iteration')
    return [available[i] for i in steps]


def inference_signature(config):
    ignored = {'pretrained', 'init_cfg', 'work_dir', 'output_mode', 'paper_process_logging',
               'doris_process_logging', 'doris_log_interval', 'physiology_collection',
               'log_stats', 'log_class_stats', 'collect_debug'}

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k not in ignored and
                    not k.startswith(('mechanism_', 'fixed_probe_', 'save_')) and 'debug' not in k}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value

    uda = deepcopy(dict(config['uda']))
    uda.pop('model', None)
    uda.pop('max_iters', None)
    target = config['data']['train']['target']
    inputs = {k: target.get(k) for k in ('type', 'use_indices', 'use_vi', 'vi_as_input',
              'index_names', 'vi_names', 'index_suffix', 'vi_norm_cfg', 'vi_norm_for_input',
              'crop_pseudo_margins')}
    return clean(dict(model=dict(config['model']), uda=uda, inputs=inputs,
                      target_img_norm_cfg=config['target_img_norm_cfg'],
                      name_dataset=config.get('name_dataset')))


def restore_response_state(model, state, completed):
    """Load only verified tensors and describe which original responses remain computable."""
    expected = model.state_dict()
    loadable, invalid = {}, {}
    dynamic = {'proto_gate.' + k for k in ('prototypes', 'source_prototypes', 'target_prototypes')}
    for key, value in state.items():
        if key not in expected:
            continue
        reference = expected[key]
        expandable = key in dynamic and reference.numel() == 0 and value.ndim == 2 and value.shape[0] == model.num_classes
        if value.shape != reference.shape and not expandable:
            invalid[key] = 'shape mismatch'
        elif reference.dtype == torch.bool and value.dtype != torch.bool:
            invalid[key] = 'invalid boolean buffer dtype'
        elif value.is_floating_point() and not torch.isfinite(value).all():
            invalid[key] = 'non-finite saved tensor'
        else:
            loadable[key] = value
    for bank, flags in (('prototypes', 'valid_proto'), ('source_prototypes', 'source_valid_proto'),
                        ('target_prototypes', 'target_valid_proto')):
        bank_key, flag_key = 'proto_gate.' + bank, 'proto_gate.' + flags
        if bank_key in loadable and flag_key in loadable:
            if loadable[bank_key].numel() == 0 and loadable[flag_key].any():
                invalid[bank_key] = 'valid class flags with an empty prototype bank'
                loadable.pop(bank_key)
    missing = sorted(set(expected) - set(loadable))
    model.load_state_dict(loadable, strict=False)
    model.local_iter = completed

    def complete(prefix):
        keys = [k for k in expected if k.startswith(prefix)]
        return bool(keys) and all(k in loadable for k in keys)

    requested = dict(phy=model.enable_phy, struct=model.enable_struct,
                     proto=model.enable_proto, edge=model.enable_struct_edge)
    student, teacher = complete('model.'), complete('ema_model.')
    phy = requested['phy'] and complete('phy_head.')
    structural = requested['struct'] and complete('struct_gate.')
    upstream = (not requested['phy'] or phy) and (not requested['struct'] or structural)
    proto = requested['proto'] and student and upstream and all(
        'proto_gate.' + key in loadable for key in ('prototypes', 'valid_proto'))
    edge = requested['edge'] and all(key in loadable for key in
        ('struct_edge_pair_prior', 'struct_edge_pair_prior_ready'))
    restrictions = []
    if getattr(model.get_model(), 'feature_scale', None) == 'all':
        proto = False
        restrictions.append('Prototype features require unsaved last_train_crop_box for feature_scale=all')
    if getattr(model, 'enable_joint_gate', False):
        raise ValueError('Joint-gate configuration is outside the shared fixed_probe protocol')
    if getattr(model, 'struct_edge_use_neighbor_prior', False):
        edge = False
        restrictions.append('Neighbor-patch memory is not persisted in state_dict')
    if requested['edge'] and getattr(model, 'edge_fusion_mode', None) != 'doris_reliable':
        edge = False
        restrictions.append('Legacy boundary fusion is outside the shared DORIS response protocol')
    executed = dict(phy=bool(phy and teacher), struct=bool(structural and teacher),
                    proto=bool(proto and teacher), edge=bool(edge and teacher))
    unavailable_weights = []
    if requested['phy'] and not phy:
        unavailable_weights += ['w_before_struct', 'w_after_struct']
    if not upstream:
        unavailable_weights += ['w_t_ps', 'delta_w_ps', 'w_final']
    if requested['proto'] and not proto:
        unavailable_weights += ['w_final', 'proto_active']
    responses_complete = all(not requested[k] or executed[k] for k in requested)
    return dict(student_complete=student, teacher_complete=teacher, requested_modules=requested,
        executed_modules=executed, historical_state_complete=not missing and responses_complete,
        responses_complete=responses_complete,
        missing_keys=missing, invalid_keys=invalid, unexpected_keys=sorted(set(state)-set(expected)),
        unavailable_weights=unavailable_weights, restrictions=restrictions,
        transient_limits=list(TRANSIENT_LIMITS))


def replay_payload(model, sample, probe_cfg, metadata, recovery):
    from mmseg.utils.mechanism import branch_probe_payload, module_states, readonly_probe
    if not recovery['teacher_complete']:
        raise ValueError('EMA Teacher state is incomplete; label-dependent mechanisms cannot be recovered')
    if any(recovery['requested_modules'].values()) and not any(recovery['executed_modules'].values()):
        raise ValueError('No enabled mechanism can be recovered from this checkpoint')
    image, vi, metas = sample
    device = next(model.parameters()).device
    image, vi = image.to(device), vi.to(device)
    valid = image.new_ones((1, 1, *image.shape[-2:]))
    top, bottom, left, right = probe_cfg['margins']
    if top: valid[:, :, :top] = 0
    if bottom: valid[:, :, -bottom:] = 0
    if left: valid[:, :, :, :left] = 0
    if right: valid[:, :, :, -right:] = 0
    metadata = deepcopy(metadata)
    metadata.update(historical_state_complete=recovery['historical_state_complete'], recovery=recovery,
                    executed_modules=recovery['executed_modules'])
    before = state_digest(model.state_dict())
    attributes = dict(phy='enable_phy', struct='enable_struct', proto='enable_proto', edge='enable_struct_edge')
    flags = {key: getattr(model, attr) for key, attr in attributes.items()}
    try:
        for key, attr in attributes.items():
            setattr(model, attr, recovery['executed_modules'][key])
        with readonly_probe(model):
            metadata['states'] = module_states(model)
            for key, requested in flags.items():
                if requested and not recovery['executed_modules'][key]:
                    metadata['states'][key] = 'unavailable_checkpoint_state'
            payload = branch_probe_payload(model, image, vi, metas, valid, metadata,
                                           unavailable_weights=recovery['unavailable_weights'])
    finally:
        for key, attr in attributes.items():
            setattr(model, attr, flags[key])
    if state_digest(model.state_dict()) != before:
        raise RuntimeError('Replay changed model parameters or buffers; refusing to export')
    return payload


def write_json(path, value):
    path = Path(path)
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    if path.exists():
        if path.read_text(encoding='utf-8') != content:
            raise ValueError('Existing replay metadata differs; use a separate --out-dir: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(path)


def replay_node(config, row, output, device, data_root=None, phy_evidence=False,
                reference_evidence=False, target_filename=None):
    from mmcv import Config
    from mmseg.models.builder import build_train_model
    from mmseg.utils.mechanism import deterministic_probe_config, load_probe_sample, readonly_probe
    from tools.build_doris_mechanism_figures import render_online_node

    checkpoint_path = Path(row['path'])
    if file_digest(checkpoint_path) != row['checkpoint_sha256']:
        raise ValueError('Checkpoint changed since discovery')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    saved_text = checkpoint.get('meta', {}).get('config')
    if not saved_text:
        raise ValueError('Checkpoint lacks meta.config; historical mechanism settings cannot be verified')
    historical = Config.fromstring(saved_text, '.py')
    requested_mode = config.get('uda', {}).get('phy_cfg', {}).get('mode', 'legacy_binary')
    saved_mode = historical.get('uda', {}).get('phy_cfg', {}).get('mode', 'legacy_binary')
    if requested_mode != saved_mode:
        raise ValueError('Physiology checkpoint mode mismatch: saved={}, requested={}. '
                         'Replay legacy checkpoints using their saved historical config.'.format(
                             saved_mode, requested_mode))
    runtime = deepcopy(config)
    schedule_info = {}
    if requested_mode == 'hierarchical_gv':
        from mmseg.models.uda.phy_reliability import restore_phy_gate_config, phy_gate_parameters
        runtime.uda.phy_cfg = restore_phy_gate_config(config.uda.phy_cfg, historical.uda.phy_cfg)
        schedule_info = dict(phy_gate_config_source='checkpoint meta.config',
            requested_phy_gate=phy_gate_parameters(config.uda.phy_cfg),
            effective_phy_gate=phy_gate_parameters(runtime.uda.phy_cfg))
        schedule_info['historical_phy_schedule'] = (
            schedule_info['requested_phy_gate'] != schedule_info['effective_phy_gate'])
    signature = inference_signature(runtime)
    if signature != inference_signature(historical):
        raise ValueError('Provided configuration differs from saved model/branch/input settings')
    runtime.runner = deepcopy(historical.runner)
    # Construction only. No runner, optimizer, init_weights or training entry point is invoked.
    runtime.model.pretrained = None
    runtime.model.backbone.pretrained = None
    runtime.model.train_cfg.work_dir = str(output)
    model = build_train_model(runtime)
    state = checkpoint_state(checkpoint)
    recovery = restore_response_state(model, state, row['completed_iter'])
    del state, checkpoint
    model.to(device)
    if data_root is not None:
        runtime.data.train.target.data_root = str(Path(data_root).resolve())
        runtime.data.val.data_root = str(Path(data_root).resolve())
    probe_cfg = (deterministic_probe_config(runtime,allow_daformer=True) if reference_evidence
                 else deterministic_probe_config(runtime))
    if not probe_cfg.get('filename'):
        raise ValueError('Configuration does not resolve one of the three required fixed target images')
    if target_filename is not None:
        if Path(target_filename).name != target_filename or Path(target_filename).suffix.lower() != '.tif':
            raise ValueError('Offline target filename must be one TIFF basename in the validation split')
        probe_cfg.update(filename=target_filename, selection_policy='explicit_offline_target_val_v1')
        target = probe_cfg['dataset']
        probe_cfg['expected_path'] = str((Path(target['data_root']) / target['img_dir'] / target_filename).resolve())
    sample, sample_meta = load_probe_sample(probe_cfg, model)
    sample_meta['data_source'] = 'checkpoint_replay'
    folder = Path(output) / Path(probe_cfg['filename']).stem / ('iter_{:06d}_{}'.format(
        row['completed_iter'], row['checkpoint_sha256'][:12]))
    source_files = ('mmseg/utils/mechanism.py', 'mmseg/models/uda/dacs.py',
        'mmseg/models/uda/phy_reliability.py',
        'mmseg/datasets/goji.py', 'mmseg/datasets/pipelines/loading.py', 'mmseg/vi_utils.py',
        'mmseg/models/uda/proto_gate.py', 'mmseg/models/uda/struct_prior_gate.py',
        'mmseg/models/segmentors/hrda_encoder_decoder.py', 'mmseg/models/decode_heads/hrda_head.py',
        'tools/replay_doris_mechanisms.py', 'tools/build_doris_mechanism_figures.py')
    if phy_evidence or reference_evidence:
        source_files += ('tools/doris_phy_evidence.py',)
    metadata = dict(iter=row['completed_iter'], rule_iter=row['completed_iter'],
        experiment_name=config.get('name'), direction=probe_cfg['direction'],
        data_source='checkpoint_replay', capture_mode='checkpoint_replay',
        checkpoint=str(checkpoint_path), checkpoint_sha256=row['checkpoint_sha256'],
        checkpoint_meta_iter=row['checkpoint_meta_iter'], iteration_source=row['iteration_source'],
        checkpoint_state_sha256=row['state_sha256'],
        state_sha256=state_digest({k:v for k,v in model.state_dict().items() if k not in recovery['missing_keys']}),
        state_digest_scope='successfully restored keys; unavailable tensors excluded',
        config_sha256=hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest(),
        saved_config_sha256=hashlib.sha256(saved_text.encode()).hexdigest(),
        sample_id=probe_cfg['filename'], target_path=sample_meta['target_path'],
        target_file_sha256=file_digest(sample_meta['target_path']),
        sample_tensor_sha256=sample_meta['tensor_sha256'], original_shape=sample_meta['original_shape'],
        input_shape=sample_meta['spatial_shape'], input_view=probe_cfg['channel_order'][:3],
        modules=recovery['requested_modules'], precision='cuda_autocast' if device.type == 'cuda' else 'cpu_float32',
        source_sha256={name:file_digest(ROOT/name) for name in source_files},
        historical_code_identity='Checkpoint does not provide hashes for all mechanism source files',
        target_update_executed=False, ema_update_executed=False)
    metadata.update(schedule_info)
    if recovery['requested_modules']['proto']:
        gate = model.proto_gate
        metadata['prototype'] = dict(valid_classes=int(gate.valid_proto.sum())
            if 'proto_gate.valid_proto' not in recovery['missing_keys'] else None,
            valid_class_ids=gate.valid_proto.nonzero().flatten().tolist()
            if 'proto_gate.valid_proto' not in recovery['missing_keys'] else None,
            target_valid_class_ids=gate.target_valid_proto.nonzero().flatten().tolist()
            if 'proto_gate.target_valid_proto' not in recovery['missing_keys'] else None,
            source_valid_class_ids=gate.source_valid_proto.nonzero().flatten().tolist()
            if 'proto_gate.source_valid_proto' not in recovery['missing_keys'] else None,
            target_update_active=bool(gate.target_update and model.local_iter >= gate.target_update_start),
            weight_start=gate.weight_start, target_update_start=gate.target_update_start,
            tau_proto=gate.tau_proto, tau_conf_proto=gate.tau_conf_proto)
    if recovery['requested_modules']['edge']:
        metadata['edge_supervision_max'] = float(model.struct_edge_loss_weight_tgt)
    with readonly_probe(model):
        payload = replay_payload(model, sample, probe_cfg, metadata, recovery)
        if phy_evidence:
            from tools.doris_phy_evidence import load_phy_reference
            payload = load_phy_reference(payload, runtime)
        elif reference_evidence:
            from tools.doris_phy_evidence import load_reference_annotation
            reference, annotation = load_reference_annotation(runtime,metadata['sample_id'])
            if reference.shape != payload['valid_mask'].shape:
                raise ValueError('Validation annotation differs from the replay pixel grid')
            payload['reference_label'] = reference
            description = json.loads(str(payload['metadata']))
            description['reference_evidence'] = dict(annotation,
                annotation_usage='Offline interpretation after model forward')
            payload['metadata'] = np.array(json.dumps(description,sort_keys=True))
        if file_digest(checkpoint_path) != row['checkpoint_sha256']:
            raise ValueError('Checkpoint changed during replay; refusing to export')
        write_json(folder/'sample_manifest.json', sample_meta)
        metadata = json.loads(str(payload['metadata']))
        write_json(folder/'replay_metadata.json', metadata)
        if phy_evidence:
            from tools.doris_phy_evidence import class_statistics_csv
            evidence = metadata['phy_evidence']
            write_json(folder/'phy_class_statistics.json', evidence)
            csv_path = folder/'phy_class_statistics.csv'
            content = class_statistics_csv(evidence)
            if csv_path.exists() and csv_path.read_text(encoding='utf-8') != content:
                raise ValueError('Existing physiology class statistics differ')
            if not csv_path.exists():
                csv_path.write_text(content, encoding='utf-8')
        arrays = folder/'responses.npz'
        if arrays.exists():
            with np.load(arrays, allow_pickle=False) as saved:
                if set(saved.files) != set(payload) or any(
                        not np.array_equal(saved[key], payload[key]) for key in payload):
                    raise ValueError('Existing replay arrays differ; use a separate --out-dir')
        else:
            temporary = arrays.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **payload)
            temporary.replace(arrays)
        figures = render_online_node(payload, folder, row['completed_iter'],
                                     array_sha256=file_digest(arrays), render_timing='checkpoint_replay')
    return dict(iter=row['completed_iter'], status='complete' if recovery['historical_state_complete'] else 'partial',
                checkpoint=str(checkpoint_path), output=str(folder),
                figures=[str(p) for p in figures], recovery=recovery)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', help='experiment config; required for replay')
    parser.add_argument('--work-dir', help='defaults to work_dir in the experiment config')
    parser.add_argument('--out-dir', help='defaults to WORK_DIR/analysis/fixed_probe/replayed')
    parser.add_argument('--audit-only', action='store_true', help='inspect files without MMCV, a model or GPU')
    parser.add_argument('--recursive', action='store_true', help='recursive checkpoint inventory, audit-only')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--iterations', nargs='+', type=int)
    selection.add_argument('--pilot', action='store_true', help='two existing middle/late checkpoints (default)')
    selection.add_argument('--all-checkpoints', action='store_true', help='explicit expansion after the pilot')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--target-data-root', help='explicit local mirror of this target domain, ending in 26 or 30')
    parser.add_argument('--phy-evidence', action='store_true', help='Offline reference-label analysis after inference')
    parser.add_argument('--target-filenames', nargs='+', help='Preselected validation TIFF basenames for offline multi-tile evidence')
    args = parser.parse_args()
    if args.recursive and not args.audit_only:
        parser.error('--recursive is limited to --audit-only to avoid mixing experiments')
    config = None
    if not args.audit_only or not args.work_dir:
        if not args.config:
            parser.error('provide a config, or --audit-only with --work-dir')
        from mmcv import Config
        config = Config.fromfile(args.config)
    work_dir = Path(args.work_dir or config.work_dir).resolve()
    if not work_dir.is_dir():
        parser.error('Experiment work directory does not exist: ' + str(work_dir))
    replay_root = work_dir/'analysis/fixed_probe/replayed'
    output = Path(args.out_dir).resolve() if args.out_dir else replay_root
    if output == work_dir or output in work_dir.parents or (
            work_dir in output.parents and output != replay_root and replay_root not in output.parents):
        parser.error('Outputs inside the experiment must stay under analysis/fixed_probe/replayed')
    records = audit_checkpoints(work_dir, args.recursive)
    audit = dict(work_dir=str(work_dir), data_source='checkpoint_replay', checkpoints=records,
                 note='State-key inventory alone does not establish full model compatibility')
    write_json(output/'checkpoint_audit.json', audit)
    print('Checkpoint audit:', output/'checkpoint_audit.json')
    if args.audit_only:
        print('Readable checkpoints: {} / {}'.format(sum(r['status']=='readable' for r in records),len(records)))
        return 0 if records and all(r['status']=='readable' for r in records) else 1
    try:
        selected = select_checkpoints(records, args.iterations, pilot=not args.all_checkpoints and not args.iterations)
    except ValueError as error:
        write_json(output/'selection_failure.json', dict(error=str(error), data_source='checkpoint_replay'))
        print(error, file=sys.stderr)
        return 1
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; select --device cpu explicitly for CPU replay')
    results = []
    for row in selected:
        for filename in args.target_filenames or [None]:
            try:
                result = replay_node(config, row, output, device, args.target_data_root,
                                     phy_evidence=args.phy_evidence, target_filename=filename)
            except Exception as error:
                result = dict(iter=row['completed_iter'], checkpoint=row['path'], status='failed',
                              sample_id=filename, error='{}: {}'.format(type(error).__name__, error))
            results.append(result)
            print(json.dumps(result, ensure_ascii=False))
    steps = '_'.join(str(r['iter']) for r in results)
    write_json(output/('replay_report_'+steps+'.json'), dict(data_source='checkpoint_replay', results=results))
    return 0 if results and all(r['status']=='complete' for r in results) else 1


if __name__ == '__main__':
    sys.exit(main())

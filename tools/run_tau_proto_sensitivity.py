"""Serial, fresh Full runs for the fixed tau_proto sensitivity protocol."""
import argparse
import ast
from contextlib import contextmanager
import csv
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import shlex
import subprocess
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FULL_CONFIG = ('configs/hrda/doris/goji26IRRG2goji30RGB/'
               'doris_goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py')
SWEEP_CONFIG = FULL_CONFIG.replace('proto_struct_edge.py', 'tau_proto_sweep.py')
TAU_VALUES = [0.05, 0.15, 0.25, 0.35, 0.45]
STEPS = list(range(1000, 15001, 1000))
HEADER = ['tau_proto', 'iter', 'mIoU']
DEFAULT_ROOT = '/root/shared-nvme/work_dirs/sensitivity_tau_proto/goji26IRRG2goji30RGB'
IO_PATHS = {
    'work_dir', 'checkpoint_config', 'log_config', 'sensitivity_analysis',
    'model.train_cfg.work_dir', 'evaluation.save_best',
    'evaluation.save_best_goji', 'evaluation.doris_best_only',
}
UDA_IO = {
    'mechanism_logging', 'mechanism_online_plot', 'mechanism_save_raw',
    'mechanism_make_figures', 'fixed_probe_enabled', 'mechanism_detailed_output',
    'mechanism_strict', 'doris_process_logging', 'paper_process_logging',
    'save_class_mix_debug', 'save_teacher_pseudo_debug', 'debug_img_interval',
    'output_mode', 'print_grad_magnitude', 'phy_cfg.save_phy_debug',
    'struct_cfg.save_struct_debug', 'proto_cfg.save_proto_debug',
    'proto_cfg.log_class_stats', 'joint_gate_cfg.save_joint_debug',
    'struct_edge_cfg.debug',
}


def tau_key(tau):
    if tau not in TAU_VALUES:
        raise ValueError('Only the five specified tau_proto values are supported')
    return format(tau, '.2f')


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != HEADER:
            raise ValueError('Invalid sensitivity CSV header')
        rows = list(reader)
    seen = set()
    for row in rows:
        if set(row) != set(HEADER) or any(value is None for value in row.values()):
            raise ValueError('Incomplete sensitivity CSV row')
        tau, step, score = float(row['tau_proto']), int(row['iter']), float(row['mIoU'])
        key = (tau, step)
        if (row['tau_proto'] != tau_key(tau) or step not in STEPS or
                not math.isfinite(score) or not 0 <= score <= 100 or
                row['mIoU'] != format(score, '.4f') or key in seen):
            raise ValueError('Invalid or duplicate sensitivity CSV row')
        seen.add(key)
    # A failed run may contain only an uninterrupted prefix of the 15 nodes.
    for tau in TAU_VALUES:
        steps = [int(row['iter']) for row in rows if row['tau_proto'] == tau_key(tau)]
        if steps != STEPS[:len(steps)]:
            raise ValueError('Sensitivity CSV has missing or unordered evaluation nodes')
    return rows


def group_rows(rows, tau):
    return [row for row in rows if row['tau_proto'] == tau_key(tau)]


def write_rows(path, rows):
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_miou(csv_path, tau, step, score):
    """Called only after the original evaluator populates its runner buffer."""
    if (not isinstance(score, Real) or isinstance(score, bool) or
            not math.isfinite(score) or not 0 <= score <= 1):
        raise ValueError('Validator mIoU must be a finite numeric fraction in [0, 1]')
    path = Path(csv_path)
    rows = read_rows(path)
    completed = group_rows(rows, tau)
    if len(completed) == len(STEPS) or step != STEPS[len(completed)]:
        raise ValueError('Duplicate or out-of-order sensitivity evaluation')
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        if not rows and stream.tell() == 0:
            writer.writerow(HEADER)
        writer.writerow([tau_key(tau), step, format(float(score) * 100, '.4f')])
        stream.flush()
        os.fsync(stream.fileno())
    state_path = path.parent / 'sweep_state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding='utf-8'))
        state['groups'][tau_key(tau)]['completed_iter'] = step
        atomic_json(state_path, state)


def changed_paths(left, right, prefix=''):
    if isinstance(left, dict) and isinstance(right, dict):
        changes = []
        for key in sorted(left.keys() | right.keys()):
            path = prefix + '.' + str(key) if prefix else str(key)
            if key not in left or key not in right:
                changes.append(path)
            else:
                changes.extend(changed_paths(left[key], right[key], path))
        return changes
    return [] if left == right else [prefix]


def check_effective_config(base, cfg):
    """Whitelist output changes while comparing every inherited training field."""
    allowed = IO_PATHS | {'uda.' + path for path in UDA_IO} | {'uda.proto_cfg.tau_proto'}
    changes = changed_paths(base, cfg)
    unexpected = [path for path in changes if not any(
        path == entry or path.startswith(entry + '.') for entry in allowed)]
    if unexpected:
        raise ValueError('Non-sensitivity configuration changes: ' + ', '.join(unexpected))
    proto = cfg['uda']['proto_cfg']
    metadata = cfg['sensitivity_analysis']
    tau_key(proto['tau_proto'])
    if (not metadata['enabled'] or metadata['tau_proto'] != proto['tau_proto'] or
            not metadata['csv_path']):
        raise ValueError('Sensitivity metadata must match the actual prototype threshold and CSV')
    for key, value in dict(tau_conf_proto=0.88, target_update_start=3000,
                           weight_start=6000, loss_start=6000, min_pixels=8,
                           target_weight_thr=None, target_conf_thr=None,
                           ignore_index=255).items():
        if proto.get(key) != value:
            raise ValueError('Unexpected Full prototype setting: ' + key)
    if (base['uda']['proto_cfg']['tau_proto'] != 0.25 or cfg['seed'] != 0 or
            cfg['uda']['pseudo_threshold'] != 0.968 or
            cfg['runner'] != dict(type='IterBasedRunner', max_iters=15000) or
            cfg['model']['backbone']['type'] != 'mit_b5' or
            cfg['model']['backbone']['in_chans'] != 6 or
            cfg['model']['decode_head']['num_classes'] != 5 or
            tuple(cfg['crop_size']) != (256, 256)):
        raise ValueError('Full protocol must remain seed=0, MiT-B5, six channels, five classes, 256px, 15K')
    data = cfg['data']
    if (data['val']['type'] != 'GojiDataset' or
            data['val']['data_root'] != data['train']['target']['data_root'] or
            data['val']['img_dir'] != 'rgb_dir/val' or
            data['val']['ann_dir'] != 'ann_dir/val' or
            data['train']['source']['img_dir'] != 'img_dir/train' or
            data['train']['target']['img_dir'] != 'rgb_dir/train' or
            Path(data['train']['source']['data_root']).name != '26' or
            Path(data['train']['target']['data_root']).name != '30'):
        raise ValueError('Expected ZN_26(IRRG) to HN_30(RGB) and complete target validation')
    for domain in ('source', 'target'):
        dataset = data['train'][domain]
        if not dataset['vi_as_input'] or dataset['vi_names'] != ['ndvi', 'ndre', 'cire']:
            raise ValueError('Full VI input protocol changed')
    if (cfg['hphy_label_mapping'] != {0: 'Goji', 1: 'Vegetation', 2: 'Building',
                                     3: 'Water', 4: 'Road', 255: 'Ignore'} or
            cfg['uda']['source_input_channel_order'] != ['NIR', 'R', 'G', 'NDVI', 'NDRE', 'CIRE'] or
            cfg['uda']['target_input_channel_order'] != ['R', 'G', 'B', 'NDVI', 'NDRE', 'CIRE']):
        raise ValueError('Full class mapping or IRRG/RGB/VI channel order changed')
    if cfg.get('load_from') or cfg.get('resume_from'):
        raise ValueError('Every group must initialize independently from the Full pretrained backbone')
    evaluation = cfg['evaluation']
    if (cfg['checkpoint_config'] is not None or evaluation.get('save_best') is not None or
            evaluation.get('save_best_goji') or evaluation.get('doris_best_only') or
            evaluation['interval'] != 1000 or evaluation['metric'] != 'mIoU' or
            evaluation.get('efficient_test', False) or cfg.get('archive_source_code') or
            cfg.get('custom_hooks')):
        raise ValueError('Sensitivity runs require full 1K evaluation and all checkpoint/extra I/O disabled')
    for path in UDA_IO:
        value = cfg['uda']
        for part in path.split('.'):
            value = value[part]
        expected = 'minimal' if path == 'output_mode' else False
        if value != expected:
            raise ValueError('Observation output is still enabled: uda.' + path)
    hooks = cfg['log_config']['hooks']
    if (cfg['log_config']['interval'] != 1000 or len(hooks) != 1 or
            hooks[0]['type'] != 'PrecisionLoggerHook' or hooks[0].get('log_memory', True) or
            hooks[0].get('paper_names', True) or hooks[0].get('paper_process_logging', True) or
            hooks[0].get('log_diagnostics', True)):
        raise ValueError('Expected a single minimal PrecisionLoggerHook every 1K')
    historical = Path(base['work_dir']).resolve()
    for output in (Path(cfg['work_dir']).resolve(), Path(metadata['csv_path']).resolve()):
        if output == historical or historical in output.parents:
            raise ValueError('Sensitivity output cannot use the historical Full directory')
    return changes


def validate_training_config(cfg):
    # Also protect direct tools/train.py invocations before any model is built.
    from mmcv import Config
    base = Config.fromfile(str(ROOT / FULL_CONFIG))._cfg_dict.to_dict()
    check_effective_config(base, cfg._cfg_dict.to_dict())


def make_command(tau, work_dir, csv_path):
    return [sys.executable, 'tools/train.py', SWEEP_CONFIG,
            '--work-dir', str(work_dir), '--seed', '0', '--options',
            'uda.proto_cfg.tau_proto=' + tau_key(tau),
            'sensitivity_analysis.tau_proto=' + tau_key(tau),
            'sensitivity_analysis.csv_path=' + str(csv_path)]


def preflight(output_root):
    try:
        from mmcv import Config
    except ModuleNotFoundError as error:
        if error.name != 'mmcv':
            raise
        # Syntax and inheritance only. Never substitute this for MMCV parsing.
        for relative in (FULL_CONFIG, SWEEP_CONFIG):
            ast.parse((ROOT / relative).read_text(encoding='utf-8'), filename=relative)
        scope = {}
        exec(compile((ROOT / SWEEP_CONFIG).read_text(encoding='utf-8'), SWEEP_CONFIG, 'exec'), scope)
        if scope['_base_'] != ['./' + Path(FULL_CONFIG).name] or scope['checkpoint_config'] is not None:
            raise ValueError('Sweep inheritance or checkpoint opt-out changed')
        print('MMCV unavailable. Static syntax/inheritance checks only. Effective config NOT verified.')
        print('Expected inherited protocol: Full / MiT-B5 / 6 channels / 256x256 / seed=0 / 15K / eval=1K / rgb_dir/val')
        print('Declared output overrides: checkpoint_config=None; save_best=None; '
              'logging=1K; mechanism/probe/debug outputs disabled. Training variable: uda.proto_cfg.tau_proto.')
        return None
    base = Config.fromfile(str(ROOT / FULL_CONFIG))._cfg_dict.to_dict()
    for tau in TAU_VALUES:
        cfg = Config.fromfile(str(ROOT / SWEEP_CONFIG))
        cfg.work_dir = str(output_root / ('tau_' + tau_key(tau).replace('.', 'p')))
        cfg.merge_from_dict({'uda.proto_cfg.tau_proto': tau,
                             'sensitivity_analysis.tau_proto': tau,
                             'sensitivity_analysis.csv_path': str(output_root / 'sensitivity_miou.csv')})
        changes = check_effective_config(base, cfg._cfg_dict.to_dict())
        print('tau_proto={} effective differences: {}'.format(tau_key(tau), ', '.join(changes)))
    print('Verified Full / {} / {} / 6 channels / {} / seed=0 / batch={} / 15K / eval=1K'.format(
        base['model']['type'], base['model']['backbone']['type'], base['crop_size'],
        base['data']['samples_per_gpu']))
    print('Validation dataset: ' + json.dumps(base['data']['val'], sort_keys=True))
    print('Pretrained initialization: ' + str(base['model']['backbone'].get('pretrained')))
    return base


@contextmanager
def sweep_lock(root):
    """Keep a small lock file. OS locks are released after process death."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.sweep.lock').open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            if stream.tell() == 0:
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # On Linux the training process inherits this descriptor, preventing
        # a second launcher from resetting CSV while an orphaned run is alive.
        yield stream


def initialize_state(root, base, resume):
    state_path = root / 'sweep_state.json'
    signature = hashlib.sha256(json.dumps(base, sort_keys=True).encode('utf-8')).hexdigest()
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()
    metadata = dict(git_sha=sha, full_config=FULL_CONFIG, sweep_config=SWEEP_CONFIG,
                    tau_values=TAU_VALUES, seed=0, batch=base['data']['samples_per_gpu'],
                    max_iters=15000, evaluation_interval=1000,
                    data=base['data'], pretrained=base['model']['backbone'].get('pretrained'),
                    baseline_signature=signature)
    if state_path.exists():
        if not resume:
            raise FileExistsError('Sweep already started. Use --resume-sweep to verify and skip completed groups.')
        state = json.loads(state_path.read_text(encoding='utf-8'))
        if any(state.get(key) != json.loads(json.dumps(value)) for key, value in metadata.items()):
            raise ValueError('Sweep provenance/configuration changed. Select a new output root.')
        return state
    if any(path.name != '.sweep.lock' for path in root.iterdir()):
        raise FileExistsError('Output root contains earlier files without a sweep state. Select a new root.')
    state = dict(metadata, groups={tau_key(tau): dict(status='pending', exit_code=None,
                 completed_iter=0, attempts=[]) for tau in TAU_VALUES})
    atomic_json(state_path, state)
    return state


def run_sweep(root, base, resume=False, retry_failed=False):
    with sweep_lock(root) as lock:
        state = initialize_state(root, base, resume)
        state_path = root / 'sweep_state.json'
        csv_path = root / 'sensitivity_miou.csv'
        rows = read_rows(csv_path)
        # Validate all state/CSV pairs before modifying any incomplete group.
        for tau in TAU_VALUES:
            group = state['groups'][tau_key(tau)]
            records = group_rows(rows, tau)
            if group['status'] == 'completed':
                if len(records) != 15 or group['completed_iter'] != 15000 or group['exit_code'] != 0:
                    raise ValueError('Completed state does not match 15 valid CSV rows')
            elif group['status'] != 'pending' or records:
                if not retry_failed:
                    raise RuntimeError('tau={} incomplete. No checkpoints exist. Use --resume-sweep '
                                       '--retry-failed to discard its CSV rows and restart at zero.'.format(tau_key(tau)))
        for tau in TAU_VALUES:
            key = tau_key(tau)
            group = state['groups'][key]
            if group['status'] == 'completed':
                print('Skipping verified completed tau=' + key)
                continue
            run_dir = root / ('tau_' + key.replace('.', 'p'))
            if group['attempts']:
                run_dir = run_dir / ('retry_{:03d}'.format(len(group['attempts']) + 1))
            if run_dir.exists() and any(run_dir.iterdir()):
                raise FileExistsError('Attempt directory is nonempty; preserving it: ' + str(run_dir))
            # Explicit retry removes only this failed group's metric rows.
            rows = read_rows(csv_path)
            write_rows(csv_path, [row for row in rows if row['tau_proto'] != key])
            command = make_command(tau, run_dir, csv_path)
            group.update(status='running', exit_code=None, completed_iter=0)
            group['attempts'].append(dict(work_dir=str(run_dir), command=command, exit_code=None))
            atomic_json(state_path, state)
            print(shlex.join(command), flush=True)
            try:
                kwargs = dict(pass_fds=(lock.fileno(),)) if os.name != 'nt' else {}
                subprocess.run(command, cwd=str(ROOT), check=True, **kwargs)
                if len(group_rows(read_rows(csv_path), tau)) != 15:
                    raise RuntimeError('Training exited without exactly 15 validation rows')
            except BaseException as error:
                rows = read_rows(csv_path)
                group.update(status='failed', exit_code=getattr(error, 'returncode', None),
                             completed_iter=1000 * len(group_rows(rows, tau)), error=str(error))
                group['attempts'][-1]['exit_code'] = group['exit_code']
                atomic_json(state_path, state)
                raise
            group.update(status='completed', exit_code=0, completed_iter=15000)
            group.pop('error', None)
            group['attempts'][-1]['exit_code'] = 0
            atomic_json(state_path, state)
        rows = read_rows(csv_path)
        if len(rows) != 75:
            raise ValueError('Expected exactly 75 complete validation rows')
        report_trajectories(rows)
        print('Completed 75 validation rows. Report iter=15000 per tau as final validation mIoU.')


def report_trajectories(rows):
    trajectories = [[float(row['mIoU']) for row in group_rows(rows, tau)] for tau in TAU_VALUES]
    if any(max(values) - min(values) > 0.01 for values in zip(*(t[:3] for t in trajectories))):
        warnings.warn('Early 1K/2K/3K results differ by >0.01 percentage points. Check RNG/non-determinism.')
    if all(trajectory[3:] == trajectories[0][3:] for trajectory in trajectories[1:]):
        warnings.warn('All post-3K trajectories match exactly. Check whether tau changes eligible w_t_ps pixels.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default=DEFAULT_ROOT)
    parser.add_argument('--values', type=float, nargs=5, default=TAU_VALUES)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume-sweep', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args(argv)
    if args.values != TAU_VALUES or args.seed != 0:
        parser.error('This protocol fixes --values 0.05 0.15 0.25 0.35 0.45 and --seed 0')
    if args.retry_failed and not args.resume_sweep:
        parser.error('--retry-failed requires --resume-sweep')
    root = Path(args.output_root).expanduser().resolve()
    if any(character.isspace() or character in ',[]()\"\'' for character in str(root)):
        parser.error('Output root must be representable unchanged by MMCV DictAction (no whitespace/quotes/brackets/commas)')
    print('Protocol notice: current Full uses 15K + MiT-B5. Reconcile the paper\'s 20K/multiple backbones manually.')
    base = preflight(root)
    if args.dry_run:
        for tau in TAU_VALUES:
            directory = root / ('tau_' + tau_key(tau).replace('.', 'p'))
            print(shlex.join(make_command(tau, directory, root / 'sensitivity_miou.csv')))
        return
    if base is None:
        raise RuntimeError('MMCV effective-config preflight is mandatory before training')
    run_sweep(root, base, args.resume_sweep, args.retry_failed)


if __name__ == '__main__':
    main()

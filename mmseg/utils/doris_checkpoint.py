"""Transactional storage for four Best models, latest and three fixed snapshots."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import random
import re
import shutil
import uuid

import numpy as np
import torch
import torch.distributed as dist

from .checkpoint_policy import prepare_save
from tools.doris_checkpoint_plan import FIXED_CHECKPOINT_ITERS

POLICY = 'doris_four_best_latest_v1'
MARKER = '.doris_checkpoint_policy.json'
JOURNAL = '.doris_checkpoint_transaction.json'
METRICS = {
    'mIoU': ('best_mIoU', 'best_score', 'best_ckpt', 'best_miou_iter'),
    'IoU.goji': ('best_goji_mIoU', 'best_goji_score', 'best_goji_ckpt', 'best_goji_iter'),
    'Veg.mIoU': ('best_Veg_mIoU', 'best_veg_score', 'best_veg_ckpt', 'best_veg_iter'),
    'NonVeg.mIoU': ('best_NonVeg_mIoU', 'best_nonveg_score', 'best_nonveg_ckpt', 'best_nonveg_iter'),
}


@contextmanager
def checkpoint_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.doris_checkpoint.lock').open('a+b') as stream:
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            if not stream.read(1):
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_identity(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def owns_checkpoint(path, record):
    if not path.is_file() or path.is_symlink():
        return False
    if file_identity(path) == record['identity']:
        return True
    return bool(record.get('sha256') and file_sha256(path) == record['sha256'])


def managed_path(root, name):
    if Path(name).name != name or not re.fullmatch(
            r'(?:best_(?:goji_mIoU|mIoU|Veg_mIoU|NonVeg_mIoU)_iter_\d+\.pth|'
            r'iter_(?:2000|4000|6000)\.pth|'
            r'\.doris-[a-f0-9]+(?:-backup-\d+|-snapshot)?\.tmp\.pth)', name):
        raise ValueError('Invalid checkpoint transaction filename: ' + str(name))
    return root / name


def recover_transaction(root, owner):
    path = root / JOURNAL
    for name in ('checkpoint_best.json.tmp', JOURNAL + '.tmp'):
        (root / name).unlink(missing_ok=True)
    if not path.exists():
        return
    journal = json.loads(path.read_text(encoding='utf-8'))
    if journal.get('owner') != owner:
        raise ValueError('Checkpoint transaction owner differs from this experiment')
    report_path = root / 'checkpoint_best.json'
    report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    committed = report.get('transaction_id') == journal['id']
    moves = journal.get('moves', [])
    backups = {move['source']: managed_path(root, move['backup']) for move in moves}
    if journal['kind'] == 'best' and not committed:
        for name in journal['new_paths']:
            # A journal is durable before any existing destination is moved.
            if name in backups and not backups[name].exists():
                continue
            managed_path(root, name).unlink(missing_ok=True)
        for move in reversed(moves):
            backup = managed_path(root, move['backup'])
            if backup.exists():
                os.replace(backup, managed_path(root, move['source']))
    if journal['kind'] == 'best' and committed:
        active = {Path(record['path']).name for key, record in report.items()
                  if key in METRICS and record.get('path')}
        for record in journal['obsolete']:
            old = managed_path(root, record['name'])
            if old.name not in active and owns_checkpoint(old, record):
                old.unlink()
    managed_path(root, journal['temporary']).unlink(missing_ok=True)
    for name in journal.get('extra_temporaries', []):
        managed_path(root, name).unlink(missing_ok=True)
    path.unlink()


def initialize_policy(runner):
    root = Path(runner.work_dir).resolve()
    with checkpoint_lock(root):
        marker = root / MARKER
        if marker.exists():
            policy = json.loads(marker.read_text(encoding='utf-8'))
            if policy.get('policy') != POLICY:
                raise ValueError('Existing experiment uses a different checkpoint policy')
        else:
            if list(root.glob('*.pth')) or (root / 'checkpoint_best.json').exists():
                raise FileExistsError('Historical checkpoints require a new --work-dir for this saving policy')
            policy = dict(policy=POLICY, owner=uuid.uuid4().hex)
            atomic_json(marker, policy)
        report = root / 'checkpoint_best.json'
        if report.exists() and json.loads(report.read_text(encoding='utf-8')).get('owner') != policy['owner']:
            raise ValueError('Best manifest is not owned by this checkpoint policy')
        recover_transaction(root, policy['owner'])
        runner.meta.setdefault('doris_checkpoint_policy', {}).update(policy)
        return policy


def write_model(runner, temporary, save_optimizer, metadata):
    original = runner.meta
    runner.meta = deepcopy(original)
    runner.meta.update(deepcopy(metadata))
    try:
        local_iter = getattr(getattr(runner.model, 'module', runner.model), 'local_iter', None)
        if local_iter is not None:
            if local_iter != runner.iter + 1:
                raise ValueError('DACS local_iter differs from the completed Runner iteration')
            runner.meta['doris_checkpoint']['local_iter'] = local_iter
        prepare_save(runner, temporary, save_optimizer=save_optimizer)
        runner.save_checkpoint(str(temporary.parent), filename_tmpl=temporary.name,
            save_optimizer=save_optimizer, create_symlink=False)
        if not temporary.is_file() or not temporary.stat().st_size:
            raise IOError('Checkpoint writer did not produce a nonempty file')
        with temporary.open('r+b') as stream:
            os.fsync(stream.fileno())
        saved = torch.load(str(temporary), map_location='cpu')
        if (not saved.get('state_dict') or
                saved.get('meta', {}).get('doris_checkpoint', {}).get('completed_iter') != runner.iter + 1 or
                ('optimizer' in saved) != save_optimizer):
            raise IOError('Saved checkpoint failed model/iteration/optimizer validation')
        expected = getattr(runner.model, 'module', runner.model).state_dict()
        if set(saved['state_dict']) != set(expected):
            raise IOError('Saved checkpoint does not contain the complete model state')
        for key, value in expected.items():
            stored = saved['state_dict'][key]
            if not torch.is_tensor(stored) or stored.shape != value.shape or stored.dtype != value.dtype:
                raise IOError('Saved checkpoint tensor shape/dtype differs: ' + key)
        del saved
    finally:
        runner.meta = original


def new_journal(owner, kind):
    token = uuid.uuid4().hex
    return dict(id=token, owner=owner, kind=kind, temporary='.doris-' + token + '.tmp.pth',
                new_paths=[], moves=[], obsolete=[])


def best_messages(messages, records):
    result = deepcopy(messages)
    for metric, (_, score_key, path_key, iter_key) in METRICS.items():
        record = records[metric]
        if record['score'] is not None:
            result.update({score_key: record['score'], path_key: record['path'], iter_key: record['iter']})
    return result


def save_bests(runner, scores):
    policy = initialize_policy(runner)
    root = Path(runner.work_dir).resolve()
    report_path = root / 'checkpoint_best.json'
    with checkpoint_lock(root):
        recover_transaction(root, policy['owner'])
        previous = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
        records = {key: deepcopy(previous.get(key, dict(score=None, iter=None, path=None))) for key in METRICS}
        step = runner.iter + 1
        improved = [key for key in METRICS if isinstance(scores.get(key), Real) and
            not isinstance(scores[key], bool) and math.isfinite(scores[key]) and 0 <= scores[key] <= 1 and
            (records[key]['score'] is None or scores[key] > records[key]['score'])]
        if not improved:
            return previous
        journal = new_journal(policy['owner'], 'best')
        destinations = [root / (METRICS[key][0] + '_iter_{}.pth'.format(step)) for key in improved]
        journal['new_paths'] = [path.name for path in destinations]
        old_records = [record for record in records.values() if record.get('path')]
        for index, path in enumerate(destinations):
            if not path.exists():
                continue
            owned = [record for record in old_records if record['path'] == str(path)]
            if not owned or not owns_checkpoint(path, dict(identity=owned[0]['file_identity'],
                                                          sha256=owned[0]['sha256'])):
                raise FileExistsError('Refusing to replace an unowned checkpoint: ' + str(path))
            remaining = [key for key in METRICS if key not in improved and records[key]['path'] == str(path)]
            backup = (root / (METRICS[remaining[0]][0] + '_iter_{}.pth'.format(records[remaining[0]]['iter']))
                      if remaining else root / ('.doris-' + journal['id'] + '-backup-{}.tmp.pth'.format(index)))
            if backup.exists():
                raise FileExistsError('Checkpoint rollback destination already exists: ' + str(backup))
            journal['moves'].append(dict(source=path.name, backup=backup.name))
            for key in remaining:
                records[key]['path'] = str(backup)
            journal['obsolete'].append(dict(name=backup.name, identity=owned[0]['file_identity'],
                                             sha256=owned[0]['sha256']))
        for record in old_records:
            journal['obsolete'].append(dict(name=Path(record['path']).name, identity=record['file_identity'],
                                             sha256=record['sha256']))
        atomic_json(root / JOURNAL, journal)
        temporary = managed_path(root, journal['temporary'])
        try:
            proposed = {key: float(scores[key]) for key in improved}
            messages = deepcopy(runner.meta.get('hook_msgs', {}))
            for key in improved:
                _, score_key, path_key, iter_key = METRICS[key]
                messages.update({score_key: proposed[key], path_key: str(destinations[0]), iter_key: step})
            metadata = dict(doris_checkpoint=dict(policy=POLICY, owner=policy['owner'], kind='best',
                resumable=False, save_optimizer=False, validation_metrics=proposed,
                metric_source='GojiDataset validation'), hook_msgs=messages)
            write_model(runner, temporary, False, metadata)
            for move in journal['moves']:
                os.replace(managed_path(root, move['source']), managed_path(root, move['backup']))
            os.replace(temporary, destinations[0])
            storage, link_error = 'hardlink', None
            try:
                for path in destinations[1:]:
                    os.link(destinations[0], path)
            except OSError as error:
                for path in destinations[1:]:
                    path.unlink(missing_ok=True)
                storage, link_error = 'single_file_reference', str(error)
                runner.logger.warning('Hard links unavailable; Best metrics share one JSON-referenced file: %s', error)
            digest = file_sha256(destinations[0])
            for key, path in zip(improved, destinations):
                actual = path if storage == 'hardlink' else destinations[0]
                records[key] = dict(score=float(scores[key]), iter=step, path=str(actual),
                    requested_path=str(path), checkpoint_present=True, save_optimizer=False,
                    resumable=False, owner=policy['owner'], sha256=digest,
                    file_identity=file_identity(actual), storage=storage, link_error=link_error)
            report = dict(records, policy=POLICY, owner=policy['owner'], transaction_id=journal['id'],
                same_iter=records['mIoU']['iter'] == records['IoU.goji']['iter']
                    if all(records[key]['iter'] is not None for key in ('mIoU', 'IoU.goji')) else None,
                metric_source='GojiDataset validation; IoU.goji is the Goji class IoU',
                git_commit=runner.meta.get('git_commit'),
                experiment=runner.meta.get('doris_checkpoint_context', {}),
                config_sha256=hashlib.sha256(runner.meta.get('config', '').encode()).hexdigest())
            report['shared_states'] = {}
            for key in METRICS:
                if records[key].get('sha256'):
                    report['shared_states'].setdefault(records[key]['sha256'], []).append(key)
            atomic_json(report_path, report)
        except BaseException:
            recover_transaction(root, policy['owner'])
            raise
        recover_transaction(root, policy['owner'])
        runner.meta['hook_msgs'] = best_messages(runner.meta.get('hook_msgs', {}), records)
        return report


def collect_rng_states():
    local = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    if dist.is_available() and dist.is_initialized():
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, local)
        return states
    return [local]


def restore_rng_state(runner):
    states = getattr(runner, '_doris_resume_rng', None)
    if states is None:
        return
    if runner.world_size != len(states):
        runner.logger.warning('Distributed world size changed; model/optimizer resume uses the current '
                              'process RNG initialization instead of saved per-rank RNG streams')
        del runner._doris_resume_rng
        return
    state = states[runner.rank]
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])
    del runner._doris_resume_rng


def save_latest(runner, rng_states=None, keep_snapshot=None):
    policy = initialize_policy(runner)
    root = Path(runner.work_dir).resolve()
    step = runner.iter + 1
    if keep_snapshot is None:
        keep_snapshot = step in FIXED_CHECKPOINT_ITERS
    if keep_snapshot and step not in FIXED_CHECKPOINT_ITERS:
        raise ValueError('Fixed retention is limited to Iter 2000, 4000 and 6000')
    with checkpoint_lock(root):
        snapshot = root / 'iter_{}.pth'.format(step)
        snapshot_exists = snapshot.exists() or snapshot.is_symlink()
        if keep_snapshot and snapshot_exists:
            if snapshot.is_symlink():
                raise FileExistsError('Refusing to replace a fixed checkpoint symlink: ' + str(snapshot))
            saved = torch.load(str(snapshot), map_location='cpu')
            record = saved.get('meta', {}).get('doris_checkpoint', {})
            if (record.get('owner') != policy['owner'] or record.get('completed_iter') != step or
                    not record.get('resumable') or not saved.get('optimizer')):
                raise FileExistsError('Existing fixed checkpoint is not owned/resumable: ' + str(snapshot))
            del saved
        journal = new_journal(policy['owner'], 'latest')
        snapshot_temporary = root / ('.doris-' + journal['id'] + '-snapshot.tmp.pth')
        if keep_snapshot and not snapshot_exists:
            journal['extra_temporaries'] = [snapshot_temporary.name]
        atomic_json(root / JOURNAL, journal)
        temporary = managed_path(root, journal['temporary'])
        latest = root / 'latest.pth'
        messages = dict(runner.meta.get('hook_msgs', {}), last_ckpt=str(latest))
        try:
            metadata = dict(hook_msgs=messages,
                doris_checkpoint=dict(policy=POLICY, owner=policy['owner'], kind='latest',
                    resumable=True, save_optimizer=True),
                doris_resume=dict(max_iters=runner.max_iters, rng_states=rng_states or collect_rng_states(),
                    lr_schedule='MMCV initial_lr + saved config + completed iteration',
                    amp_state_source='meta.fp16 and any existing optimizer-hook loss_scaler'))
            if keep_snapshot:
                metadata['doris_checkpoint']['fixed_snapshot_path'] = str(snapshot)
            write_model(runner, temporary, True, metadata)
            if keep_snapshot and not snapshot_exists:
                try:
                    os.link(temporary, snapshot_temporary)
                except OSError as error:
                    runner.logger.warning('Hard link unavailable; storing a full fixed-checkpoint copy: %s', error)
                    if shutil.disk_usage(root).free < temporary.stat().st_size + 64 * 1024**2:
                        raise OSError('Insufficient free space for a full fixed-checkpoint copy') from error
                    shutil.copyfile(temporary, snapshot_temporary)
                    with snapshot_temporary.open('r+b') as stream:
                        os.fsync(stream.fileno())
                    if file_sha256(snapshot_temporary) != file_sha256(temporary):
                        raise IOError('Fixed checkpoint copy failed its checksum validation')
                os.replace(snapshot_temporary, snapshot)
            os.replace(temporary, latest)
        finally:
            recover_transaction(root, policy['owner'])
        runner.meta['hook_msgs'] = messages

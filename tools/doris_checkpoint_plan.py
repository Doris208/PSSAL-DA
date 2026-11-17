"""DORIS checkpoint policy and historical boundary analysis (no training)."""
from copy import deepcopy
from pathlib import Path
import re


FIXED_CHECKPOINT_ITERS = (2000, 4000, 6000)


def applies(config):
    return bool(re.fullmatch(
        r'doris_(goji26RGB2goji30RGB|goji26IRRG2goji30RGB|goji30IRRG2goji26IRRG)'
        r'_hrda_?3vi(?:_phy_gate|_struct_gate|_phy_struct_gate|_ps_proto|_struct_edge|'
        r'_pbcr|_proto_struct_edge)?(?:_hphy15k|_hphy_early3k|_hphy_early4k)?', config.get('name', '')))


def checkpoint_plan(config):
    """Boundaries use zero-based DACS local_iter; filenames count completed steps.

    At local_iter T, iter_T.pth is the state before executing that step. Reuse
    a periodic point on the same side within 500 steps, otherwise save T and
    T+500. Dynamic sample/cache readiness is deliberately not assigned an Iter.
    """
    maximum = config['runner']['max_iters']
    periodic = list(range(2000, maximum + 1, 2000))
    if maximum not in periodic:
        periodic.append(maximum)
    events = []

    def event(branch, kind, boundary):
        if 0 < boundary < maximum:
            before = [n for n in periodic if boundary - 500 <= n <= boundary]
            after = [n for n in periodic if boundary < n <= boundary + 500]
            events.append(dict(branch=branch, kind=kind, local_iter=boundary,
                before=max(before) if before else boundary,
                after=min(after) if after else min(boundary + 500, maximum)))

    uda = config['uda']
    phy, struct, proto, edge = [uda.get(k, {}) for k in
        ('phy_cfg', 'struct_cfg', 'proto_cfg', 'struct_edge_cfg')]
    if phy.get('enable') and phy.get('use_target_gate', True):
        if phy.get('mode') == 'hierarchical_gv' and phy.get('phy_gate_schedule') == 'early_nonzero_linear':
            event('phy', 'ramp_end', phy['phy_gate_full_iter'])
        else:
            start, ramp = phy.get('phy_warmup_iters', 1000), phy.get('phy_rampup_iters', 0)
            event('phy', 'gate_start', start)
            if ramp > 0:
                event('phy', 'ramp_end', start + ramp)
    if struct.get('enable'):
        start, ramp = struct.get('struct_warmup_iters', 6000), struct.get('struct_rampup_iters', 4000)
        early = struct.get('struct_gate_schedule', 'legacy_warmup_linear') == 'early_nonzero_linear'
        if early:
            start, ramp = 0, struct.get('struct_gate_full_iter', 4000)
        if struct.get('use_target_gate', True):
            event('struct', 'gate_start', start)
            event('struct', 'ramp_end', start + max(ramp, 1))
        if struct.get('use_feature_interaction', False):
            if not early:
                start = struct.get('feature_interaction_warmup_iters', start)
                ramp = struct.get('feature_interaction_rampup_iters', ramp)
            event('struct_feature', 'interaction_start', start)
            event('struct_feature', 'ramp_end', start + max(ramp, 1))
    if proto.get('enable'):
        if proto.get('target_update', True):
            event('proto', 'target_update_eligible', proto.get('target_update_start', 3000))
        event('proto', 'weight_eligible', proto.get('weight_start', 0))
        if proto.get('lambda_proto_tgt', 0) > 0:
            event('proto', 'loss_eligible', proto.get('loss_start', 3000))
    if edge.get('enable') and edge.get('use_edge_loss', True):
        event('edge', 'loss_ramp_end', edge.get('warmup_iters', 3000))
        # DORIS valid_boundary_only returns before the legacy pair ramp.
        if edge.get('use_boundary_pair_prior', False) and not (
                edge.get('edge_fusion_mode') == 'doris_reliable' and
                edge.get('pair_response_mode') == 'valid_boundary_only'):
            event('edge_pair', 'ramp_end', edge.get('boundary_pair_warmup_iters',
                                                  edge.get('warmup_iters', 3000)))
    extra = sorted({e[side] for e in events for side in ('before', 'after')} - set(periodic))
    return dict(periodic=periodic, extra=extra, all_iterations=sorted(set(periodic + extra)),
                events=events, dynamic_activation='Not inferred; inspect actual logs/cache statistics')


def configure_checkpoint_policy(config):
    """Unified opt-in by the existing DORIS experiment names; other jobs unchanged."""
    if config.get('sensitivity_analysis', {}).get('enabled', False):
        return False
    if not applies(config):
        return False
    config['checkpoint_config'] = dict(type='DorisCheckpointHook', by_epoch=False,
        interval=2000, max_keep_ckpts=-1, save_optimizer=True, save_last=False,
        keep_iterations=list(FIXED_CHECKPOINT_ITERS))
    evaluation = deepcopy(dict(config['evaluation']))
    evaluation.update(save_best='mIoU', save_best_goji=False, doris_best_only=True)
    config['evaluation'] = evaluation
    return True


def check_output_directory(config):
    """Fail before the training entry point writes config/logs into an old run."""
    if not applies(config):
        return
    directory = Path(config['work_dir'])
    if directory.exists() and (list(directory.glob('*.pth')) or (directory / 'checkpoint_best.json').exists()):
        marker = directory / '.doris_checkpoint_policy.json'
        if not marker.is_file():
            raise FileExistsError('Historical checkpoints use another saving policy; preserve this directory '
                                  'and select a new --work-dir')
    if (directory / 'iter_{}.pth'.format(config['runner']['max_iters'])).exists():
        raise FileExistsError('Work directory already contains the final checkpoint; preserve the completed '
                              'experiment and use a new --work-dir: ' + str(directory))
    if directory.exists() and any(directory.iterdir()) and not config.get('resume_from'):
        raise FileExistsError('DORIS work_dir is not empty; use a new --work-dir or an explicit '
                              '--resume-from for continuation: ' + str(directory))


def check_hphy_training_checkpoint(config):
    """Reject schedule changes before loading or writing a resumed experiment."""
    phy = config.get('uda', {}).get('phy_cfg', {})
    path = config.get('resume_from') or config.get('load_from')
    if not path or not phy.get('enable') or phy.get('mode') != 'hierarchical_gv':
        return
    import torch
    from mmcv import Config
    from mmseg.models.uda.phy_reliability import phy_gate_parameters
    checkpoint = torch.load(path, map_location='cpu')
    meta = checkpoint.get('meta', {})
    saved = meta.get('phy_cfg')
    if saved is None and meta.get('config'):
        saved = Config.fromstring(meta['config'], '.py').get('uda', {}).get('phy_cfg')
    if not saved or saved.get('mode') != 'hierarchical_gv':
        raise ValueError('Checkpoint lacks a verifiable H-Phy schedule and head configuration')
    current_gate, saved_gate = phy_gate_parameters(phy), phy_gate_parameters(saved)
    for key in ('phy_min_weight', 'use_target_gate'):
        current_gate[key] = phy.get(key, .5 if key == 'phy_min_weight' else True)
        saved_gate[key] = saved.get(key, .5 if key == 'phy_min_weight' else True)
    if current_gate != saved_gate:
        raise ValueError('H-Phy gate schedule mismatch. A new schedule requires a fresh experiment. '
                         'Requested {}, saved {}'.format(current_gate, saved_gate))


def check_doris_resume_checkpoint(config, model=None):
    """Validate full resume before files/model state can be changed."""
    path = config.get('resume_from')
    if not applies(config) or not path:
        return None
    import torch
    from mmcv import Config
    from tools.replay_doris_mechanisms import checkpoint_state, inference_signature
    checkpoint = torch.load(path, map_location='cpu')
    meta = checkpoint.get('meta', {})
    saved_policy = meta.get('doris_checkpoint', {})
    if not checkpoint.get('optimizer') or saved_policy.get('resumable') is False:
        raise ValueError('Checkpoint has no complete optimizer/resume state. Use --load-from for '
                         'Best weight initialization, or --resume-from PATH/latest.pth for continuation.')
    if not meta.get('config'):
        raise ValueError('Resume checkpoint lacks the training configuration')
    saved = Config.fromstring(meta['config'], '.py')
    if inference_signature(config) != inference_signature(saved):
        raise ValueError('Resume model, input protocol or module configuration differs from checkpoint')
    for key in ('optimizer', 'lr_config', 'runner', 'seed', 'fp16'):
        if config.get(key) != saved.get(key):
            raise ValueError('Resume training protocol differs in ' + key)
    completed = saved_policy.get('completed_iter')
    if completed is None:
        match = re.fullmatch(r'(?:best_.+_)?iter_(\d+)\.pth', Path(path).resolve().name)
        completed = int(match[1]) if match else None
    if (type(completed) is not int or type(meta.get('iter')) is not int or
            meta['iter'] not in (completed, completed-1) or not 0 < completed <= config['runner']['max_iters']):
        raise ValueError('Resume checkpoint lacks consistent completed-iteration metadata')
    meta = deepcopy(meta)
    meta.setdefault('doris_checkpoint', {})['completed_iter'] = completed
    if saved_policy.get('policy') == 'doris_four_best_latest_v1':
        if saved_policy.get('local_iter', completed) != completed:
            raise ValueError('Saved DACS local_iter disagrees with the completed iteration')
        if meta.get('doris_resume', {}).get('max_iters') != config['runner']['max_iters']:
            raise ValueError('Resume LR scheduling horizon differs from checkpoint')
        if not meta.get('doris_resume', {}).get('rng_states'):
            raise ValueError('Latest checkpoint lacks its saved RNG state')
    if model is not None:
        unwrapped = getattr(model, 'module', model)
        unwrapped.load_state_dict(checkpoint_state(checkpoint), strict=True)
    return meta


def audit_configs(loader):
    root = Path(__file__).resolve().parents[1]
    records = []
    for path in sorted((root/'configs/hrda/doris').glob('*/*.py')):
        config = loader(path)
        if not configure_checkpoint_policy(config):
            continue
        records.append(dict(config=str(path.relative_to(root)), work_dir=config['work_dir'],
            modules={key:dict(config['uda'][key]) for key in
                     ('phy_cfg','struct_cfg','proto_cfg','struct_edge_cfg')},
            checkpoint=dict(config['checkpoint_config']), evaluation=dict(config['evaluation']),
            fixed_checkpoint_iterations=list(FIXED_CHECKPOINT_ITERS),
            plan_analysis_only=True, plan=checkpoint_plan(config),
            latest_iterations=list(range(2000, config['runner']['max_iters'] + 1, 2000))))
    return records


if __name__ == '__main__':
    import argparse
    import json
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mmcv import Config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='new JSON audit file; no training is started')
    args = parser.parse_args()
    records = audit_configs(lambda path: Config.fromfile(str(path)))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(records, stream, indent=2, ensure_ascii=False)
    print('Audited {} experiment plans: {}'.format(len(records), output))

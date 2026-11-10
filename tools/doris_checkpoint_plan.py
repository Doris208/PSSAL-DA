"""Pure configuration planning for DORIS historical checkpoints (no training)."""
from copy import deepcopy
from pathlib import Path
import re


def applies(config):
    return bool(re.fullmatch(
        r'doris_(goji26RGB2goji30RGB|goji26IRRG2goji30RGB|goji30IRRG2goji26IRRG)'
        r'_hrda_?3vi(?:_phy_gate|_struct_gate|_phy_struct_gate|_ps_proto|_struct_edge|'
        r'_pbcr|_proto_struct_edge)?', config.get('name', '')))


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
        start, ramp = phy.get('phy_warmup_iters', 1000), phy.get('phy_rampup_iters', 0)
        event('phy', 'gate_start', start)
        if ramp > 0:
            event('phy', 'ramp_end', start + ramp)
    if struct.get('enable'):
        start, ramp = struct.get('struct_warmup_iters', 6000), struct.get('struct_rampup_iters', 4000)
        if struct.get('use_target_gate', True):
            event('struct', 'gate_start', start)
            event('struct', 'ramp_end', start + max(ramp, 1))
        if struct.get('use_feature_interaction', False):
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
    if not applies(config):
        return False
    plan = checkpoint_plan(config)
    checkpoint = deepcopy(dict(config.get('checkpoint_config', {})))
    checkpoint.update(type='DorisCheckpointHook', by_epoch=False, interval=2000,
        max_keep_ckpts=-1, save_optimizer=True, save_last=True, key_iterations=plan['extra'])
    config['checkpoint_config'] = checkpoint
    evaluation = deepcopy(dict(config['evaluation']))
    evaluation['save_best_goji'] = True
    config['evaluation'] = evaluation
    return True


def check_output_directory(config):
    """Fail before the training entry point writes config/logs into an old run."""
    if not applies(config):
        return
    directory = Path(config['work_dir'])
    if (directory / 'iter_{}.pth'.format(config['runner']['max_iters'])).exists():
        raise FileExistsError('Work directory already contains the final checkpoint; preserve the completed '
                              'experiment and use a new --work-dir: ' + str(directory))
    if directory.exists() and any(directory.iterdir()) and not config.get('resume_from'):
        raise FileExistsError('DORIS work_dir is not empty; use a new --work-dir or an explicit '
                              '--resume-from for continuation: ' + str(directory))


def audit_configs(loader):
    root = Path(__file__).resolve().parents[1]
    records = []
    for path in sorted((root/'configs/hrda/doris').glob('*/*.py')):
        config = loader(path)
        if not applies(config):
            continue
        configure_checkpoint_policy(config)
        records.append(dict(config=str(path.relative_to(root)), work_dir=config['work_dir'],
            modules={key:dict(config['uda'][key]) for key in
                     ('phy_cfg','struct_cfg','proto_cfg','struct_edge_cfg')},
            checkpoint=dict(config['checkpoint_config']), evaluation=dict(config['evaluation']),
            plan=checkpoint_plan(config)))
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

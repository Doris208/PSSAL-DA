"""Parse the real MMCV inheritance tree without starting training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.check_doris_stage1_ablation_configs import STRUCT_V2_DIRECTIONS


DIRECTION = 'goji26RGB2goji30RGB'
EXPERIMENTS = (
    ('Baseline', '', (False, False, False, False)),
    ('Physiology', '_phy_gate', (True, False, False, False)),
    ('Structural', '_struct_gate', (False, True, False, False)),
    ('MPARC', '_phy_struct_gate', (True, True, False, False)),
    ('Prototype', '_ps_proto', (False, False, True, False)),
    ('Edge', '_struct_edge', (False, False, False, True)),
    ('PBCR', '_pbcr', (False, False, True, True)),
    ('Full', '_proto_struct_edge', (True, True, True, True)),
)


def config_path(suffix, direction=DIRECTION):
    return ROOT / 'configs/hrda/doris' / direction / (
        'doris_' + direction + '_hrda_3vi' + suffix + '.py')


def check_config(path, expected=None):
    from mmcv import Config
    cfg = Config.fromfile(str(path))
    direction = cfg.name_dataset.rsplit('_256x256_3vi', 1)[0]
    flags = tuple(bool(cfg.uda.get(key, {}).get('enable', False)) for key in
                  ('phy_cfg', 'struct_cfg', 'proto_cfg', 'struct_edge_cfg'))
    assert expected is None or flags == expected, (path, flags, expected)
    assert cfg.runner.max_iters == 15000, path
    assert cfg.model.backbone.type == 'mit_b5', path
    assert cfg.model.backbone.in_chans == 6, path
    assert cfg.model.decode_head.num_classes == 5, path
    assert cfg.seed == 0 and cfg.lr_config.policy == 'poly', path
    assert cfg.get('resume_from') is None and cfg.get('load_from') is None, path
    assert not cfg.uda.joint_gate_cfg.enable, path
    interaction = cfg.model.decode_head.get('struct_prior_interaction', {})
    assert bool(interaction.get('enable', False)) == flags[1], path
    losses = cfg.model.decode_head.loss_decode
    losses = losses if isinstance(losses, list) else [losses]
    edge_losses = [loss for loss in losses if loss.type == 'StructEdgeLoss']
    assert len(edge_losses) == int(flags[3]), path
    if flags[1]:
        if direction in STRUCT_V2_DIRECTIONS:
            from tools.check_doris_stage1_ablation_configs import STRUCT_V2_OVERRIDES
            for key, value in STRUCT_V2_OVERRIDES.items():
                assert cfg.uda.struct_cfg[key] == value, (path, key)
        else:
            assert cfg.uda.struct_cfg.fusion_mode == 'doris_confidence_guided', path
        assert interaction.gamma_struct == .05, path
    if flags[2]:
        assert cfg.uda.proto_cfg.tau_proto == .25 and cfg.uda.proto_cfg.tau_conf_proto == .88, path
    if flags[3]:
        assert cfg.uda.struct_edge_cfg.edge_fusion_mode == 'doris_reliable', path
        assert edge_losses[0].boundary_loss_mode == 'doris_balanced', path
        assert edge_losses[0].loss_weight == .1, path
    for domain in ('source', 'target'):
        data = cfg.data.train[domain]
        assert list(data.vi_names) == ['ndvi', 'ndre', 'cire'], path
        assert data.use_vi and data.vi_as_input and not data.vi_norm_for_input, path
        annotations = [step for step in data.pipeline if step.type == 'LoadAnnotations']
        assert annotations and annotations[0].reduce_zero_label, path
    if flags[0]:
        required = dict(mode='hierarchical_gv', beta_gv=1.0,
                        lambda_phy_src=0.08, lambda_phy_tgt=0.008,
                        lambda_phy_tgt_gv=0.0, tau_conf_phy=0.968,
                        phy_min_weight=0.75, phy_filter_threshold=None,
                        phy_warmup_iters=0, phy_rampup_iters=4000,
                        phy_gate_schedule='early_nonzero_linear',
                        phy_gate_initial_strength=0.10, phy_gate_full_iter=4000,
                        detach_target_gate=True,
                        phy_target_supervision_mode='balanced_high_conf')
        for key, value in required.items():
            assert cfg.uda.phy_cfg[key] == value, (path, key)
        if direction not in STRUCT_V2_DIRECTIONS:
            assert '/doris_hphy_early4k/' in cfg.work_dir and cfg.work_dir.endswith('_hphy_early4k'), path
    if direction in STRUCT_V2_DIRECTIONS:
        expected_work_dir = '/root/shared-nvme/work_dirs/doris_struct_distinct_v2/' + direction + '/' + Path(path).stem[6:]
        assert cfg.work_dir == expected_work_dir, path
    return cfg, dict(config=Path(path).relative_to(ROOT).as_posix(),
                     iterations=cfg.runner.max_iters, channels=cfg.model.backbone.in_chans,
                     flags=flags, phy_mode=cfg.uda.phy_cfg.get('mode', 'disabled'),
                     phy_gate={key: cfg.uda.phy_cfg.get(key) for key in
                         ('phy_gate_schedule', 'phy_gate_initial_strength', 'phy_gate_full_iter',
                          'phy_min_weight')} if flags[0] else None,
                     config_work_dir=cfg.work_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=[str(i) for i in range(1, 9)] + ['all'], default='all')
    parser.add_argument('--all-directions', action='store_true')
    parser.add_argument('--direction', choices=STRUCT_V2_DIRECTIONS, default=DIRECTION)
    args = parser.parse_args()
    rows, reference = [], None
    for number, (name, suffix, flags) in enumerate(EXPERIMENTS, 1):
        cfg, row = check_config(config_path(suffix, args.direction), flags)
        protocol = dict(data=cfg.data, seed=cfg.seed, optimizer=cfg.optimizer,
                        lr_config=cfg.lr_config, evaluation=cfg.evaluation,
                        backbone=cfg.model.backbone, runner=cfg.runner)
        if reference is None:
            reference = protocol
        assert protocol == reference, ('Table IV protocol differs', name)
        if args.experiment not in ('all', str(number)):
            continue
        row.update(number=number, name=name, launch_work_dir=(
            '/root/shared-nvme/work_dirs/doris_struct_distinct_v2/' + args.direction + '/'
            + args.direction + '_hrda_3vi' + suffix))
        rows.append(row)
    if args.all_directions:
        for direction in ('goji26IRRG2goji30RGB', 'goji30IRRG2goji26IRRG'):
            for suffix in ('_phy_gate', '_phy_struct_gate', '_proto_struct_edge'):
                _, row = check_config(config_path(suffix, direction))
                rows.append(row)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()

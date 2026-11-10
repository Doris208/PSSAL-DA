"""Statically audit the 24 DORIS ablations against each direction's current Full."""
import argparse
from copy import deepcopy
from pathlib import Path
import shlex
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.check_doris_stage1_configs import VARIANTS


DIRECTIONS = ('goji26RGB2goji30RGB', 'goji26IRRG2goji30RGB',
              'goji30IRRG2goji26IRRG')
MODULES = ('phy', 'struct', 'proto', 'struct_edge')
PROTOCOL = ('data', 'optimizer', 'optimizer_config', 'lr_config', 'runner',
            'checkpoint_config', 'evaluation', 'seed', 'fp16', 'crop_size',
            'source_img_norm_cfg', 'target_img_norm_cfg')


def audit(reader):
    rows, commands, outputs = [], [], set()
    assert len(list((ROOT / 'configs/hrda/doris').glob('*/*.py'))) == 24
    assert not list((ROOT / 'configs/hrda').glob('doris_*.py'))
    for direction in DIRECTIONS:
        full_name = 'doris_' + direction + '_hrda_3vi_proto_struct_edge'
        full = reader(str(ROOT / 'configs/hrda/doris' / direction / (full_name + '.py')))
        for suffix, flags in VARIANTS:
            name = 'doris_' + direction + '_hrda_3vi' + suffix
            relative = 'configs/hrda/doris/' + direction + '/' + name + '.py'
            cfg = reader(str(ROOT / relative))
            actual = tuple(cfg.uda.get(module + '_cfg', {}).get('enable', False)
                           for module in MODULES)
            assert actual == flags, (name, actual, flags)
            assert not cfg.uda.get('joint_gate_cfg', {}).get('enable', False), name
            for key in PROTOCOL:
                assert cfg.get(key) == full.get(key), (name, 'protocol', key)
            expected_model = deepcopy(full.model)
            if not flags[1]:
                expected_model.decode_head.pop('struct_prior_interaction', None)
            if not flags[3]:
                losses = expected_model.decode_head.loss_decode
                expected_model.decode_head.loss_decode = next(
                    loss for loss in losses if loss.type == 'CrossEntropyLoss')
            assert cfg.model == expected_model, (name, 'model differs beyond module switches')
            for module, enabled in zip(MODULES, flags):
                if enabled:
                    assert cfg.uda[module + '_cfg'] == full.uda[module + '_cfg'], (name, module)
            # Non-module training options must match too. Observation/channel
            # metadata are allowed to differ for the already-existing Baseline.
            ignored = {module + '_cfg' for module in MODULES}
            ignored.update(('doris_process_logging', 'doris_log_interval',
                            'source_input_channel_order', 'target_input_channel_order', 'physiology_collection'))
            def training_options(uda):
                return {key: value for key, value in uda.items()
                        if key not in ignored and not key.startswith(
                            ('mechanism_', 'fixed_probe_'))}
            assert training_options(cfg.uda) == training_options(full.uda), (name, 'UDA')
            assert cfg.model.type == 'HRDAEncoderDecoder'
            assert cfg.model.backbone.type == 'mit_b5'
            assert cfg.model.backbone.in_chans == 6
            assert cfg.model.decode_head.num_classes == 5
            assert cfg.runner.max_iters == 15000 and cfg.data.samples_per_gpu == 2
            assert cfg.log_config == full.log_config, (name, 'logger')
            assert cfg.uda.get('mechanism_logging', False) == any(flags), name
            assert cfg.uda.get('fixed_probe_enabled', False) == any(flags), name
            if any(flags):
                assert cfg.uda.mechanism_output_mode == 'branches'
                assert not cfg.uda.mechanism_detailed_output
                assert cfg.uda.mechanism_online_plot and cfg.uda.mechanism_strict
                assert cfg.uda.mechanism_summary_interval == cfg.uda.fixed_probe_interval == 500
                assert cfg.uda.mechanism_save_raw
            else:
                assert not cfg.uda.get('mechanism_online_plot', False)
            assert cfg.uda.get('physiology_collection', False) == flags[0], name
            if flags[0]:
                phy = cfg.uda.phy_cfg
                assert phy.phy_target_supervision_mode == 'balanced_high_conf'
                assert (phy.phy_pos_weight, phy.phy_neg_weight, phy.tau_conf_phy) == (.5, .5, .968)
                assert (phy.lambda_phy_src, phy.lambda_phy_tgt) == (.08, .008)
                assert phy.mode == 'hierarchical_gv'
                assert (phy.phy_min_weight, phy.phy_filter_threshold, phy.detach_target_gate) == (.75, None, True)
                assert (phy.phy_gate_schedule, phy.phy_gate_initial_strength,
                        phy.phy_gate_full_iter) == ('early_nonzero_linear', .10, 4000)
            if flags[1]:
                assert cfg.uda.struct_cfg.fusion_mode == 'doris_confidence_guided'
            if flags[2]:
                assert cfg.uda.proto_cfg.tau_proto == 0.25
                assert cfg.uda.proto_cfg.tau_conf_proto == 0.88
            if flags[3]:
                assert cfg.uda.struct_edge_cfg.edge_fusion_mode == 'doris_reliable'
                assert cfg.uda.struct_edge_cfg.pair_response_mode == 'valid_boundary_only'
            if flags[0]:
                assert cfg.work_dir == ('/root/shared-nvme/work_dirs/doris_hphy_early4k/' +
                                       direction + '/' + name[6:] + '_hphy_early4k'), name
            else:
                assert cfg.work_dir == '/root/shared-nvme/work_dirs/doris_stage1/' + direction + '/' + name[6:], name
            assert cfg.work_dir not in outputs, (name, 'duplicate default work_dir')
            outputs.add(cfg.work_dir)
            experiment = name[6:]
            commands.append(['python', 'tools/train.py', relative, '--work-dir',
                             '/root/shared-nvme/work_dirs/doris_stage1/' + direction + '/' + experiment])
            rows.append((name, flags))
    # Preserve the archival launcher identity. Fresh 4K experiments use scripts/run_doris_tableiv_*.
    script = (ROOT / 'train_doris_mitb5_stage1_ablations_20k.sh').read_text(encoding='utf-8').encode('utf-8')
    for launcher in (ROOT / 'scripts').glob('*.sh'):
        assert b'\r' not in launcher.read_bytes(), (launcher, 'Active launchers require LF line endings')
    assert b'set -e\n' in script
    assert b'$RUN_ROOT' not in script, 'Each command must use a standalone absolute work_dir'
    actual_commands = [shlex.split(line) for line in script.decode().splitlines()
                       if line.startswith('python ')]
    assert actual_commands == commands, 'Batch script order/paths differ from the 24 configs'
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-custom-imports', action='store_true',
                        help='Only parse configs when the local model dependencies are unavailable')
    args = parser.parse_args()
    from mmcv import Config
    rows = audit(lambda path: Config.fromfile(
        path, import_custom_modules=not args.skip_custom_imports))
    for name, flags in rows:
        print('{}: PASS phy/struct/proto/edge={}'.format(name, flags))
    print('PASS: 24 DORIS configurations and 24 commands; active module parameters '
          'and training protocols match the corresponding Full. No training executed.')


if __name__ == '__main__':
    main()

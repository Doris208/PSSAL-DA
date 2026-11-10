"""Audit 24 legacy + 6 DORIS + 2 MiT-B5 comparison configs with MMCV."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DIRECTIONS = ('goji30IRRG2goji26IRRG', 'goji26IRRG2goji30RGB',
              'goji26RGB2goji30RGB')
VARIANTS = (
    ('', (False, False, False, False)),
    ('_phy_gate', (True, False, False, False)),
    ('_struct_gate', (False, True, False, False)),
    ('_phy_struct_gate', (True, True, False, False)),
    ('_ps_proto', (False, False, True, False)),
    ('_struct_edge', (False, False, False, True)),
    ('_pbcr', (False, False, True, True)),
    ('_proto_struct_edge', (True, True, True, True)))
COMPARISONS = (
    ('memoryadaptnet', 'doris_goji26IRRG2goji30RGB_memoryadaptnet_3vi',
     'MemoryAdaptEncoderDecoder', 'MemoryAdaptNet'),
    ('daformer', 'doris_goji26RGB2goji30RGB_daformer_3vi',
     'EncoderDecoder', 'DACS'))


def check_config(cfg, expected, doris_full=False):
    uda = cfg['uda']
    actual = tuple(uda.get(name + '_cfg', {}).get('enable', False)
                   for name in ('phy', 'struct', 'proto', 'struct_edge'))
    assert actual == expected, (actual, expected)
    assert not uda.get('joint_gate_cfg', {}).get('enable', False)
    struct_mode = uda.get('struct_cfg', {}).get('fusion_mode', 'legacy_product')
    edge_mode = uda.get('struct_edge_cfg', {}).get('edge_fusion_mode', 'legacy')
    losses = cfg['model']['decode_head'].get('loss_decode', [])
    losses = losses if isinstance(losses, list) else [losses]
    boundary_losses = [x for x in losses if x.get('type') == 'StructEdgeLoss']
    if doris_full:
        assert struct_mode == 'doris_confidence_guided'
        assert edge_mode == 'doris_reliable'
        assert len(boundary_losses) == 1
        assert boundary_losses[0]['boundary_loss_mode'] == 'doris_balanced'
        assert boundary_losses[0]['loss_weight'] == 0.1
        assert (boundary_losses[0]['bce_weight'], boundary_losses[0]['ce_weight'],
                boundary_losses[0]['dice_weight']) == (1.0, 0.1, 0.0)
        assert cfg['model']['decode_head']['struct_prior_interaction']['gamma_struct'] == 0.05
        edge = uda['struct_edge_cfg']
        assert edge['pair_response_mode'] == 'valid_boundary_only'
        assert uda['proto_cfg']['tau_proto'] == 0.25
        assert uda['proto_cfg']['tau_conf_proto'] == 0.88
        assert uda['proto_cfg']['target_weight_thr'] is None
        assert uda['proto_cfg']['target_conf_thr'] is None
        assert not uda['doris_process_logging']
        assert uda['mechanism_logging'] and uda['fixed_probe_enabled']
        assert uda['mechanism_summary_interval'] == uda['fixed_probe_interval'] == 500
        assert uda['mechanism_sample_interval'] == 100
        assert uda['fixed_probe_max_samples'] == 1
        assert uda['mechanism_save_raw'] and uda['mechanism_make_figures']
        assert not cfg['log_config']['hooks'][0]['paper_names']
        for flag in ('use_struct_gate', 'use_spec_gate', 'use_conf_weight',
                     'use_proto_gate', 'use_spatial_layout_prior', 'use_neighbor_patch_prior'):
            assert not edge.get(flag, False), flag
        assert (edge['prob_edge_weight'], edge['img_edge_weight'],
                edge['pair_edge_weight']) == (0.45, 0.35, 0.20)
    else:
        assert struct_mode == 'legacy_product'
        assert edge_mode == 'legacy'
        assert all(x.get('boundary_loss_mode', 'legacy') == 'legacy' for x in boundary_losses)
    assert cfg['runner']['max_iters'] == 20000
    assert cfg['data']['samples_per_gpu'] == 2
    assert cfg['seed'] == 0
    assert cfg['uda']['pseudo_threshold'] == 0.968
    assert cfg['model']['backbone']['in_chans'] == 6
    return actual, struct_mode, edge_mode


def audit_configs(reader):
    rows, old_dirs, new_dirs = [], set(), set()
    for direction in DIRECTIONS:
        for suffix, flags in VARIANTS:
            name = direction + '_hrda_3vi' + suffix
            cfg = reader(str(ROOT / 'configs/hrda' / (name + '.py')))
            actual, struct_mode, edge_mode = check_config(cfg, flags)
            old_dirs.add(cfg['work_dir'])
            rows.append(dict(config=name, modules=actual, struct=struct_mode, edge=edge_mode))
        for full in (False, True):
            name = 'doris_' + direction + '_hrda_3vi' + ('_proto_struct_edge' if full else '')
            cfg = reader(str(ROOT / 'configs/hrda/doris' / direction / (name + '.py')))
            actual, struct_mode, edge_mode = check_config(cfg, (full,) * 4, full)
            assert cfg['log_config']['hooks'][0]['log_memory']
            if not full:
                assert not cfg['uda'].get('mechanism_logging', False)
            assert cfg['work_dir'] not in new_dirs
            new_dirs.add(cfg['work_dir'])
            rows.append(dict(config=name, modules=actual, struct=struct_mode, edge=edge_mode,
                             work_dir=cfg['work_dir']))
    assert not old_dirs.intersection(new_dirs)
    return rows


def audit_comparison_configs(reader):
    rows = []
    for folder, name, segmentor, wrapper in COMPARISONS:
        cfg = reader(str(ROOT / 'configs' / folder / (name + '.py')))
        model, uda = cfg['model'], cfg['uda']
        assert model['type'] == segmentor
        assert uda['type'] == wrapper
        assert model['backbone']['type'] == 'mit_b5'
        assert model['backbone']['in_chans'] == 6
        assert model['backbone']['pretrained'] == '/root/shared-nvme/pretrained/mit_b5.pth'
        assert not model.get('pretrained')
        assert not any(k in model['backbone'] for k in ('depth', 'in_channels', 'dilations'))
        assert cfg['runner']['max_iters'] == 20000
        assert cfg['seed'] == 0 and cfg['data']['samples_per_gpu'] == 2
        assert cfg['optimizer']['lr'] == 6e-5
        assert cfg['checkpoint_config']['interval'] == cfg['evaluation']['interval'] == 1000
        assert cfg['data']['train']['source']['data_root'] == '/root/shared-nvme/mydataset/26'
        assert cfg['data']['train']['target']['data_root'] == '/root/shared-nvme/mydataset/30'
        assert cfg['work_dir'] == '/root/shared-nvme/work_dirs/doris_stage1/' + name[6:]
        assert cfg['log_config']['hooks'][0]['type'] == 'PrecisionLoggerHook'
        for module in ('phy', 'struct', 'proto', 'struct_edge', 'joint_gate'):
            assert not uda.get(module + '_cfg', {}).get('enable', False)
        if folder == 'memoryadaptnet':
            assert model['stage1_head']['in_channels'] == 512
            assert model['memory_bottleneck']['in_channels'] == 512
            assert model['memory_bottleneck']['out_channels'] == model['memory_module']['memory_dim']
            assert model['memory_module']['output_channels'] == model['stage2_head']['in_channels']
            assert model['feature_upsample_scale'] == 4
            assert cfg['optimizer']['paramwise_cfg']['custom_keys']['discriminator']['lr_mult'] == 0
            assert cfg['custom_hooks'][0]['type'] == 'MemoryAdaptNetOptimizerStateHook'
        else:
            assert model['decode_head']['in_channels'] == [64, 128, 320, 512]
            assert model['decode_head']['type'] == 'DAFormerHead'
        rows.append(dict(config=name, modules=(False,) * 4, struct='n/a', edge='n/a',
                         backbone='mit_b5', uda_type=wrapper, work_dir=cfg['work_dir']))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', help='optional output path for the audit table')
    args = parser.parse_args()
    from mmcv import Config
    rows = audit_configs(Config.fromfile)
    comparisons = audit_comparison_configs(Config.fromfile)
    assert not {r['work_dir'] for r in rows if 'work_dir' in r}.intersection(
        r['work_dir'] for r in comparisons)
    rows.extend(comparisons)
    print('config | phy struct proto edge | structural fusion | edge fusion')
    for row in rows:
        print('{} | {} | {} | {}'.format(row['config'],
              ' '.join(str(int(v)) for v in row['modules']), row['struct'], row['edge']))
    for row in comparisons:
        print('{}: backbone={}, UDA={}'.format(row['config'], row['backbone'], row['uda_type']))
    print('PASS: 24 legacy and 8 new effective configs (6 HRDA/Full + 2 comparisons); '
          'output directories are distinct.')
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()

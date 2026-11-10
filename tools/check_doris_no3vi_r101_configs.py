"""Audit the three input ablations with real MMCV Config; no Torch/GPU needed."""
from copy import deepcopy
from pathlib import Path
import shlex
import sys

from mmcv import Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PAIRS = (
    ('configs/hrda/baseline_resnet/',
     'doris_goji30IRRG2goji26IRRG_hrda_r101_no3vi.py',
     'goji30IRRG2goji26IRRG_baseline_resnet101.py', 'HRDAEncoderDecoder'),
    ('configs/memoryadaptnet/',
     'doris_goji26IRRG2goji30RGB_memoryadaptnet_r101_no3vi.py',
     'goji26IRRG2goji30RGB_memoryadaptnet_r101_3vi.py',
     'MemoryAdaptEncoderDecoder'),
    ('configs/daformer/',
     'doris_goji26RGB2goji30RGB_daformer_r101_no3vi.py',
     'goji26RGB2goji30RGB_daformer_3vi.py', 'EncoderDecoder'),
)


def without_vi(value):
    """Compare effective data semantics after removing only VI input fields."""
    if isinstance(value, list):
        return [without_vi(item) for item in value
                if not isinstance(item, dict)
                or item.get('type') != 'LoadGojiIndices']
    if not isinstance(value, dict):
        return value
    vi_fields = {'use_indices', 'use_vi', 'index_names', 'vi_names',
                 'vi_as_input', 'vi_norm_for_input', 'vi_norm_cfg'}
    result = {key: without_vi(item) for key, item in value.items()
              if key not in vi_fields}
    if result.get('type') == 'Collect':
        result['keys'] = [key for key in result['keys'] if key != 'vi_img']
    if result.get('type') == 'Normalize':
        result['mean'] = result['mean'][:3]
        result['std'] = result['std'][:3]
        order = result.get('channel_order') or [0, 1, 2]
        order = order[:3]
        result['channel_order'] = order[::-1] if result['to_rgb'] else order
        result['to_rgb'] = False
        result.pop('vis_channel_order', None)
    return result


def check_pipeline(pipeline):
    for step in pipeline:
        assert step['type'] != 'LoadGojiIndices'
        if step['type'] == 'Normalize':
            assert len(step['mean']) == len(step['std']) == 3
            assert sorted(step.get('channel_order', [0, 1, 2])) == [0, 1, 2]
        if step['type'] == 'Collect':
            assert 'vi_img' not in step['keys']
        if 'transforms' in step:
            check_pipeline(step['transforms'])


def main():
    work_dirs = set()
    expected_commands = []
    for directory, filename, reference, model_type in PAIRS:
        path = directory + filename
        # Skip runtime custom_imports, which need the cloud Torch environment.
        cfg = Config.fromfile(str(ROOT / path), import_custom_modules=False)
        old = Config.fromfile(str(ROOT / directory / reference),
                              import_custom_modules=False)
        assert cfg.model.type == model_type
        backbone = cfg.model.backbone
        assert backbone.type == 'ResNetV1c' and backbone.depth == 101
        assert backbone.in_channels == 3 and 'in_chans' not in backbone
        assert backbone.pretrained == 'open-mmlab://resnet101_v1c'
        expected_model = deepcopy(old.model)
        expected_model.backbone.in_channels = 3
        expected_model.backbone.pretrained = backbone.pretrained
        assert cfg.model == expected_model, path + ': network changed'
        heads = ('stage1_head', 'stage2_head') if model_type == \
            'MemoryAdaptEncoderDecoder' else ('decode_head',)
        assert all(cfg.model[head].num_classes == 5 for head in heads)
        expected_uda = deepcopy(old.uda)
        if 'input_channel_order' in cfg.uda:
            expected_uda['input_channel_order'] = cfg.uda.input_channel_order
            assert len(cfg.uda.input_channel_order) == 3
        assert cfg.uda == expected_uda, path + ': UDA changed'
        for key in ('phy_cfg', 'struct_cfg', 'proto_cfg', 'struct_edge_cfg',
                    'joint_gate_cfg'):
            assert not cfg.uda.get(key, {}).get('enable', False)
        expected_optimizer = deepcopy(old.optimizer)
        if model_type == 'MemoryAdaptEncoderDecoder':
            expected_optimizer['paramwise_cfg'] = dict(custom_keys=dict(
                discriminator=dict(lr_mult=0.0, decay_mult=0.0)))
            comparison = Config.fromfile(str(ROOT / 'configs/memoryadaptnet/'
                'doris_goji26IRRG2goji30RGB_memoryadaptnet_3vi.py'),
                import_custom_modules=False)
            assert cfg.optimizer == comparison.optimizer
        assert cfg.optimizer == expected_optimizer, path + ': optimizer changed'
        for key in ('optimizer_config', 'lr_config', 'evaluation',
                    'checkpoint_config', 'seed', 'fp16', 'custom_imports',
                    'custom_hooks'):
            assert cfg.get(key) == old.get(key), path + ': ' + key
        assert cfg.runner == dict(type='IterBasedRunner', max_iters=20000)
        assert cfg.data.samples_per_gpu == 2
        assert without_vi(cfg.data) == without_vi(old.data), \
            path + ': split/augmentation/normalization changed'
        for dataset in (cfg.data.train.source, cfg.data.train.target,
                        cfg.data.val, cfg.data.test):
            assert dataset.type == 'GojiDataset'
            assert not dataset.get('use_indices', False)
            assert not dataset.get('use_vi', False)
            check_pipeline(dataset.pipeline)
        assert cfg.evaluation.save_best == 'mIoU'
        assert cfg.log_config.interval == 50
        assert all(hook.type == 'PrecisionLoggerHook' and not hook.by_epoch
                   for hook in cfg.log_config.hooks)
        assert cfg.evaluation.interval == cfg.checkpoint_config.interval == 1000
        assert cfg.work_dir.startswith(
            '/root/shared-nvme/work_dirs/doris_no3vi_r101/')
        assert cfg.work_dir not in work_dirs
        work_dirs.add(cfg.work_dir)
        expected_commands.append(['python', 'tools/train.py', path])
        print('{}: PASS, {}, R101/3ch/5classes, 20k, batch=2, lr={}'.format(
            filename, cfg.uda.type, cfg.optimizer.lr))
    script = (ROOT / 'train_doris_no3vi_r101_20k.sh').read_bytes()
    assert b'\r' not in script and b'set -e\n' in script
    commands = [shlex.split(line) for line in script.decode().splitlines()
                if line.startswith('python ')]
    assert commands == expected_commands
    print('PASS: original networks/UDA/protocol retained except VI input, '
          'matching V1c pretraining, user-confirmed 20k override and '
          'MemoryAdaptNet discriminator optimizer isolation. '
          'Cloud initialization/training not executed.')


if __name__ == '__main__':
    main()

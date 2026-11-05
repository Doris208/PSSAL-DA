import ast
import copy
import math
import os
import runpy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

TASKS = [
    (
        'goji30IRRG2goji26IRRG',
        'configs/advent/goji30IRRG2goji26IRRG_advent_r101_3vi.py',
        'configs/_base_/datasets/uda_goji30IRRG_to_goji26IRRG_256x256_3vi.py',
        'configs/dacs/goji30IRRG2goji26IRRG_dacs_r101_3vi.py',
        ('30', 'IRRG'),
        ('26', 'IRRG'),
    ),
    (
        'goji26IRRG2goji30RGB',
        'configs/advent/goji26IRRG2goji30RGB_advent_r101_3vi.py',
        'configs/_base_/datasets/uda_goji26IRRG_to_goji30RGB_256x256_3vi.py',
        'configs/dacs/goji26IRRG2goji30RGB_dacs_r101_3vi.py',
        ('26', 'IRRG'),
        ('30', 'RGB'),
    ),
    (
        'goji26RGB2goji30RGB',
        'configs/advent/goji26RGB2goji30RGB_advent_r101_3vi.py',
        'configs/_base_/datasets/uda_goji26RGB_to_goji30RGB_256x256_3vi.py',
        'configs/dacs/goji26RGB2goji30RGB_dacs_r101_3vi.py',
        ('26', 'RGB'),
        ('30', 'RGB'),
    ),
]

PRETRAINED = '/root/HRDA-master/pretrained/resnet101-5d3b4d8f.pth'
EXPECTED_WORK_DIR_ROOT = '/root/shared-nvme/work_dirs/advent_r101'
ALLOWED_MAIN_DIFFS = {'auxiliary_head'}


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def public_config_vars(namespace):
    return {
        key: value
        for key, value in namespace.items()
        if not key.startswith('__') and key not in {'Path', 'os'}
    }


def deep_merge(base, child):
    result = copy.deepcopy(base)
    for key, value in child.items():
        if key == '_base_':
            continue
        if (isinstance(value, dict) and isinstance(result.get(key), dict)
                and not value.pop('_delete_', False)):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path):
    path = (REPO_ROOT / path).resolve()
    namespace = public_config_vars(runpy.run_path(str(path)))
    bases = namespace.get('_base_', [])
    if isinstance(bases, str):
        bases = [bases]
    merged = {}
    for base in bases:
        base_path = (path.parent / base).resolve()
        merged = deep_merge(merged, load_config(base_path.relative_to(REPO_ROOT)))
    return deep_merge(merged, namespace)


def top_level_keys(path):
    namespace = public_config_vars(runpy.run_path(str(REPO_ROOT / path)))
    return set(namespace.keys()) - {'_base_'}


def check_no_duplicate_base_keys(config):
    path = (REPO_ROOT / config).resolve()
    namespace = public_config_vars(runpy.run_path(str(path)))
    bases = namespace.get('_base_', [])
    if isinstance(bases, str):
        bases = [bases]
    seen = {}
    for base in bases:
        rel = (path.parent / base).resolve().relative_to(REPO_ROOT)
        for key in top_level_keys(rel):
            if key in seen:
                raise AssertionError(
                    f'Duplicate base key "{key}" in {seen[key]} and {rel}')
            seen[key] = rel


def count_input_channels(pipeline):
    for step in pipeline:
        if step.get('type') == 'Normalize':
            return len(step['mean'])
    return None


def domain_and_modality(dataset_cfg):
    root = dataset_cfg['data_root'].replace('\\', '/')
    domain = root.rstrip('/').split('/')[-1]
    modality = 'RGB' if dataset_cfg['img_dir'].startswith('rgb_dir') else 'IRRG'
    return domain, modality


def compact_decode_head(head):
    keys = ('type', 'in_channels', 'in_index', 'dilations', 'num_classes',
            'align_corners')
    return {key: head.get(key) for key in keys}


def check_model(cfg, dacs_cfg):
    model = cfg['model']
    backbone = model['backbone']
    decode_head = model['decode_head']
    aux = model['auxiliary_head']
    dacs_head = dacs_cfg['model']['decode_head']

    assert_true(model['type'] == 'EncoderDecoder', 'model.type mismatch')
    assert_true(backbone['type'] == 'ResNetV1c', 'backbone.type mismatch')
    assert_true(backbone['depth'] == 101, 'backbone.depth mismatch')
    assert_true(backbone['in_channels'] == 6, 'backbone.in_channels mismatch')
    assert_true(backbone['pretrained'] == PRETRAINED, 'pretrained mismatch')
    assert_true(decode_head['num_classes'] == 5, 'decode_head classes mismatch')
    assert_true(decode_head['in_channels'] == 2048, 'decode_head channels mismatch')
    assert_true(decode_head['in_index'] == 3, 'decode_head in_index mismatch')
    assert_true(compact_decode_head(decode_head) == compact_decode_head(dacs_head),
                'main decode head differs from DACS-R101+3VI baseline')
    assert_true(aux['type'] == 'FCNHead', 'auxiliary head type mismatch')
    assert_true(aux['in_channels'] == 1024, 'auxiliary head channels mismatch')
    assert_true(aux['in_index'] == 2, 'auxiliary head in_index mismatch')
    assert_true(aux['num_classes'] == 5, 'auxiliary head classes mismatch')
    assert_true(model['test_cfg']['mode'] == 'slide', 'test mode mismatch')
    assert_true(tuple(model['test_cfg']['crop_size']) == (256, 256),
                'test crop size mismatch')
    assert_true(tuple(model['test_cfg']['stride']) == (128, 128),
                'test stride mismatch')


def check_uda(cfg):
    uda = cfg['uda']
    expected = dict(
        type='AdvEnt',
        num_classes=5,
        multi_level=True,
        lambda_seg_main=1.0,
        lambda_seg_aux=0.1,
        lambda_adv_main=0.001,
        lambda_adv_aux=0.0002,
        source_domain_label=0.0,
        target_domain_label=1.0,
    )
    for key, value in expected.items():
        assert_true(uda.get(key) == value, f'uda.{key} mismatch')
    discriminator = uda['discriminator']
    assert_true(discriminator['type'] == 'FCDiscriminator',
                'discriminator type mismatch')
    assert_true(discriminator['in_channels'] == 5,
                'discriminator channels mismatch')
    assert_true(discriminator['base_channels'] == 64,
                'discriminator base channels mismatch')
    assert_true(discriminator['negative_slope'] == 0.2,
                'discriminator slope mismatch')
    opt = uda['discriminator_optimizer']
    assert_true(opt['type'] == 'Adam', 'discriminator optimizer type mismatch')
    assert_true(opt['lr'] == 1e-4, 'discriminator optimizer lr mismatch')
    assert_true(tuple(opt['betas']) == (0.9, 0.99),
                'discriminator optimizer betas mismatch')


def check_data(cfg, task, expected_src, expected_tgt):
    data = cfg['data']
    source = data['train']['source']
    target = data['train']['target']
    src_domain, src_modality = domain_and_modality(source)
    tgt_domain, tgt_modality = domain_and_modality(target)
    assert_true((src_domain, src_modality) == expected_src,
                f'{task} source mismatch: {(src_domain, src_modality)}')
    assert_true((tgt_domain, tgt_modality) == expected_tgt,
                f'{task} target mismatch: {(tgt_domain, tgt_modality)}')
    for split_name, split in [('source', source), ('target', target)]:
        assert_true(split['use_indices'] is True, f'{task} {split_name} no 3VI')
        assert_true(split['index_names'] == ['ndvi', 'ndre', 'cire'],
                    f'{task} {split_name} indices mismatch')
        assert_true(count_input_channels(split['pipeline']) == 6,
                    f'{task} {split_name} input channels mismatch')
    print(f'{task}: source={src_domain}{src_modality}+3VI target={tgt_domain}{tgt_modality}+3VI')


def check_names_and_work_dir(cfg, task):
    assert_true(cfg['name_encoder'] == 'r101', 'name_encoder mismatch')
    assert_true(cfg['name_architecture'] == 'advent_r101_3vi',
                'name_architecture mismatch')
    assert_true(cfg['name_uda'] == 'advent', 'name_uda mismatch')
    assert_true(cfg['work_dir'] == f'{EXPECTED_WORK_DIR_ROOT}/{task}_advent_r101_3vi',
                'work_dir mismatch')


def check_entropy_source():
    source = (REPO_ROOT / 'mmseg/models/utils/advent_entropy.py').read_text()
    assert_true('F.softmax(logits, dim=1)' in source, 'entropy softmax mismatch')
    assert_true('eps=1e-30' in source, 'entropy eps missing')
    assert_true('math.log2(num_classes)' in source, 'entropy normalization missing')
    assert_true('.detach()' not in source, 'entropy function must not detach')
    try:
        import torch
        from mmseg.models.utils.advent_entropy import prob_2_entropy
        logits = torch.randn(2, 5, 8, 8)
        entropy = prob_2_entropy(logits, 5)
        assert_true(tuple(entropy.shape) == (2, 5, 8, 8),
                    'entropy tensor shape mismatch')
        assert_true(torch.isfinite(entropy).all().item(), 'entropy non-finite')
    except ModuleNotFoundError:
        print('SKIP runtime entropy tensor check: torch/mmcv unavailable')


def check_discriminator_source():
    source = (REPO_ROOT / 'mmseg/models/discriminators/fc_discriminator.py').read_text()
    tree = ast.parse(source)
    assert_true('nn.Sigmoid' not in source and 'Sigmoid(' not in source,
                'discriminator must not end with sigmoid')
    assert_true('nn.LeakyReLU' in source and 'negative_slope=negative_slope' in source,
                'LeakyReLU(0.2) wiring missing')
    conv_defs = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call)
                 and getattr(getattr(node.func, 'attr', ''), 'lower', lambda: '')() == 'conv2d']
    assert_true(len(conv_defs) == 5, 'FCDiscriminator must have five conv layers')
    assert_true('self.classifier = nn.Conv2d(base_channels * 8, 1' in source,
                'classifier output channel must be 1')


def check_advent_source():
    source = (REPO_ROOT / 'mmseg/models/uda/advent.py').read_text()
    required = [
        'self.d_main = FCDiscriminator',
        'self.d_aux = FCDiscriminator',
        'self._set_discriminator_grad(False)',
        'self._set_discriminator_grad(True)',
        'entropy_src_main = self._entropy(pred_src',
        'entropy_tgt_main = entropy_tgt_main.detach()',
        'optimizer.step()',
        'self.optimizer_d_main.step()',
        'self.optimizer_d_aux.step()',
        'advent_optimizer_state_dict',
        'load_advent_optimizer_state_dict',
    ]
    for needle in required:
        assert_true(needle in source, f'AdvEnt source missing: {needle}')


def check_optimizer_hook(cfg):
    hooks = cfg.get('custom_hooks', [])
    assert_true(any(hook.get('type') == 'AdvEntOptimizerStateHook'
                    for hook in hooks), 'AdvEnt optimizer state hook missing')
    source = (REPO_ROOT / 'mmseg/core/optimizer/advent_optimizer_hook.py').read_text()
    assert_true('advent_optimizer_state_dict' in source,
                'optimizer sidecar save missing')
    assert_true('load_advent_optimizer_state_dict' in source,
                'optimizer sidecar load missing')


def check_exclusions(cfg):
    forbidden = [
        'MinEnt', 'lambda_ent', 'EMA', 'ClassMix', 'MIC', 'SePiCo', 'ProCA',
        'HRDAEncoderDecoder', 'DAFormerHead', 'phy_gate', 'struct_gate',
        'proto_gate', 'edge', 'PBCR'
    ]
    text = repr(cfg)
    for item in forbidden:
        assert_true(item not in text, f'forbidden mechanism present: {item}')


def check_manifest():
    manifest = (REPO_ROOT / 'benchmark_external_methods.json').read_text()
    for task, config, *_ in TASKS:
        assert_true(f'"task": "{task}"' in manifest,
                    f'manifest task missing: {task}')
        assert_true(config in manifest, f'manifest config missing: {config}')
    assert_true(manifest.count('"name": "AdvEnt"') == 3,
                'manifest must contain three AdvEnt entries')


def main():
    for task, config, dataset_base, dacs_config, expected_src, expected_tgt in TASKS:
        check_no_duplicate_base_keys(config)
        cfg = load_config(config)
        dacs_cfg = load_config(dacs_config)
        config_text = (REPO_ROOT / config).read_text()
        assert_true(Path(dataset_base).name in config_text,
                    f'{task} does not inherit expected 3VI dataset base')
        check_model(cfg, dacs_cfg)
        check_uda(cfg)
        check_data(cfg, task, expected_src, expected_tgt)
        check_names_and_work_dir(cfg, task)
        check_optimizer_hook(cfg)
        check_exclusions(cfg)
        print(f'{task}: SAME main model/data baseline fields, DIFF-ALLOWED AdvEnt auxiliary/discriminator fields')

    check_entropy_source()
    check_discriminator_source()
    check_advent_source()
    check_manifest()
    print('PASS: AdvEnt-R101-3VI configs are ready.')


if __name__ == '__main__':
    main()

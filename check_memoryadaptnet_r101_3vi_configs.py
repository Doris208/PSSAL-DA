import ast
import copy
import runpy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
PRETRAINED = '/root/HRDA-master/pretrained/resnet101-5d3b4d8f.pth'
WORK_DIR_ROOT = '/root/shared-nvme/work_dirs/memoryadaptnet_r101'

TASKS = [
    (
        'goji30IRRG2goji26IRRG',
        'configs/memoryadaptnet/goji30IRRG2goji26IRRG_memoryadaptnet_r101_3vi.py',
        'configs/_base_/datasets/uda_goji30IRRG_to_goji26IRRG_256x256_3vi.py',
        ('30', 'IRRG'),
        ('26', 'IRRG'),
    ),
    (
        'goji26IRRG2goji30RGB',
        'configs/memoryadaptnet/goji26IRRG2goji30RGB_memoryadaptnet_r101_3vi.py',
        'configs/_base_/datasets/uda_goji26IRRG_to_goji30RGB_256x256_3vi.py',
        ('26', 'IRRG'),
        ('30', 'RGB'),
    ),
    (
        'goji26RGB2goji30RGB',
        'configs/memoryadaptnet/goji26RGB2goji30RGB_memoryadaptnet_r101_3vi.py',
        'configs/_base_/datasets/uda_goji26RGB_to_goji30RGB_256x256_3vi.py',
        ('26', 'RGB'),
        ('30', 'RGB'),
    ),
]


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def public_config_vars(namespace):
    return {
        k: v
        for k, v in namespace.items()
        if not k.startswith('__') and k not in {'Path', 'os'}
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
        rel = (path.parent / base).resolve().relative_to(REPO_ROOT)
        merged = deep_merge(merged, load_config(rel))
    return deep_merge(merged, namespace)


def top_level_keys(path):
    return set(public_config_vars(runpy.run_path(str(REPO_ROOT / path)))) - {
        '_base_'
    }


def check_no_duplicate_base_keys(config):
    path = (REPO_ROOT / config).resolve()
    bases = public_config_vars(runpy.run_path(str(path))).get('_base_', [])
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


def check_data(cfg, task, expected_src, expected_tgt):
    source = cfg['data']['train']['source']
    target = cfg['data']['train']['target']
    src = domain_and_modality(source)
    tgt = domain_and_modality(target)
    assert_true(src == expected_src, f'{task} source mismatch: {src}')
    assert_true(tgt == expected_tgt, f'{task} target mismatch: {tgt}')
    for name, split in [('source', source), ('target', target)]:
        assert_true(split['use_indices'] is True, f'{task} {name} no 3VI')
        assert_true(split['index_names'] == ['ndvi', 'ndre', 'cire'],
                    f'{task} {name} index mismatch')
        assert_true(count_input_channels(split['pipeline']) == 6,
                    f'{task} {name} input channel mismatch')
    print(f'{task}: source={src[0]}{src[1]}+3VI target={tgt[0]}{tgt[1]}+3VI')


def check_model(cfg):
    model = cfg['model']
    backbone = model['backbone']
    stage1 = model['stage1_head']
    bottleneck = model['memory_bottleneck']
    memory = model['memory_module']
    stage2 = model['stage2_head']
    assert_true(model['type'] == 'MemoryAdaptEncoderDecoder',
                'model type mismatch')
    assert_true(backbone['type'] == 'ResNetV1c', 'backbone type mismatch')
    assert_true(backbone['depth'] == 101, 'backbone depth mismatch')
    assert_true(backbone['in_channels'] == 6, 'backbone input mismatch')
    assert_true(backbone['pretrained'] == PRETRAINED, 'pretrained mismatch')
    assert_true(stage1['in_channels'] == 2048, 'stage1 input mismatch')
    assert_true(stage1['num_classes'] == 5, 'stage1 classes mismatch')
    assert_true(bottleneck['in_channels'] == 2048, 'bottleneck input mismatch')
    assert_true(bottleneck['out_channels'] == 1024,
                'bottleneck output mismatch')
    assert_true(memory['num_classes'] == 5, 'memory classes mismatch')
    assert_true(memory['memory_dim'] == 1024, 'memory dim mismatch')
    assert_true(memory['transform_channels'] == 512,
                'memory transform mismatch')
    assert_true(memory['num_prototypes_per_class'] == 1,
                'memory prototypes mismatch')
    assert_true(memory['output_channels'] == 2048, 'memory output mismatch')
    assert_true(stage2['in_channels'] == 2048, 'stage2 input mismatch')
    assert_true(stage2['num_classes'] == 5, 'stage2 classes mismatch')
    assert_true(tuple(model['test_cfg']['crop_size']) == (256, 256),
                'test crop mismatch')
    assert_true(tuple(model['test_cfg']['stride']) == (128, 128),
                'test stride mismatch')


def check_uda(cfg):
    uda = cfg['uda']
    assert_true(uda['type'] == 'MemoryAdaptNet', 'uda type mismatch')
    assert_true(uda['num_classes'] == 5, 'uda classes mismatch')
    assert_true(uda['ignore_index'] == 255, 'ignore index mismatch')
    assert_true(uda['memory_start_iter'] == 3000,
                'memory start iter mismatch')
    assert_true(uda['target_entropy_threshold'] == 0.3,
                'target entropy threshold mismatch')
    assert_true(uda['lambda_stage1'] == 1.0, 'lambda stage1 mismatch')
    assert_true(uda['lambda_stage2'] == 1.0, 'lambda stage2 mismatch')
    assert_true(uda['lambda_adv'] == 0.001, 'lambda adv mismatch')
    discriminator = uda['discriminator']
    assert_true(discriminator['type'] == 'FCDiscriminator',
                'discriminator type mismatch')
    assert_true(discriminator['in_channels'] == 5,
                'discriminator input mismatch')
    assert_true(cfg['uda'].get('d_aux') is None,
                'MemoryAdaptNet must not define d_aux')
    opt = uda['discriminator_optimizer']
    assert_true(opt['type'] == 'Adam', 'discriminator optimizer mismatch')
    assert_true(opt['lr'] == 1e-4, 'discriminator lr mismatch')


def check_names(cfg, task):
    assert_true(cfg['name_encoder'] == 'r101', 'name_encoder mismatch')
    assert_true(cfg['name_architecture'] == 'memoryadaptnet_r101_3vi',
                'name_architecture mismatch')
    assert_true(cfg['name_uda'] == 'memoryadaptnet', 'name_uda mismatch')
    assert_true(cfg['work_dir'] == f'{WORK_DIR_ROOT}/{task}_memoryadaptnet_r101_3vi',
                'work_dir mismatch')


def check_sources():
    segmentor = (REPO_ROOT / 'mmseg/models/segmentors/memory_adapt_encoder_decoder.py').read_text()
    uda = (REPO_ROOT / 'mmseg/models/uda/memory_adaptnet.py').read_text()
    memory = (REPO_ROOT / 'mmseg/models/utils/domain_prototype_memory.py').read_text()
    attention = (REPO_ROOT / 'mmseg/models/utils/memory_attention.py').read_text()
    discriminator = (REPO_ROOT / 'mmseg/models/discriminators/fc_discriminator.py').read_text()

    required_segmentor = [
        'MemoryAdaptStage1Head', 'memory_bottleneck',
        'DomainPrototypeMemory', 'stage2_head', 'encode_decode',
        "outputs['stage2_logits']"
    ]
    for item in required_segmentor:
        assert_true(item in segmentor, f'segmentor missing {item}')
    for item in ['register_buffer', "'memory'", 'dist.all_reduce(sums)',
                 'dist.all_reduce(counts)', 'copy_(', 'allow_uninitialized=False']:
        assert_true(item in memory or item in uda, f'memory update missing {item}')
    for item in ['MemorySelfAttentionBlock', 'torch.matmul', 'F.softmax']:
        assert_true(item in attention, f'attention missing {item}')
    assert_true('F.softmax(tgt[\'stage1_logits\'], dim=1)' in uda,
                'discriminator input must be stage-1 probability')
    assert_true('prob_2_entropy' not in uda and 'advent_entropy' not in uda,
                'must not use AdvEnt weighted self-information')
    assert_true('self.discriminator = FCDiscriminator' in uda,
                'single discriminator missing')
    assert_true('d_aux' not in uda, 'must not create auxiliary discriminator')
    assert_true('target_pseudo' in uda and 'target_entropy_threshold' in uda,
                'target entropy-filtered pseudo update missing')
    assert_true('gt_semantic_seg' in uda and 'target_img_metas' in uda,
                'source/target train API missing')
    assert_true('target_gt' not in uda and 'target_label' not in uda,
                'target GT appears in training source')
    assert_true('nn.Sigmoid' not in discriminator and 'Sigmoid(' not in discriminator,
                'discriminator must not use sigmoid')
    assert_true('BCEWithLogitsLoss' in uda, 'BCEWithLogitsLoss missing')
    assert_true('memoryadaptnet_optimizer_state_dict' in uda,
                'discriminator optimizer state save missing')


def check_exclusions(cfg):
    forbidden = [
        'ClassMix', 'MIC', 'SePiCo', 'ProCA', 'HRDAEncoderDecoder',
        'DAFormerHead', 'MiT-B5', 'mit_b5', 'phy_gate', 'struct_gate',
        'proto_gate', 'PBCR', 'AdvEnt'
    ]
    text = repr(cfg)
    for item in forbidden:
        assert_true(item not in text, f'forbidden mechanism present: {item}')


def check_manifest():
    manifest = (REPO_ROOT / 'benchmark_external_methods.json').read_text()
    assert_true(manifest.count('"name": "MemoryAdaptNet"') == 3,
                'manifest must contain three MemoryAdaptNet entries')
    for _, config, *_ in TASKS:
        assert_true(config in manifest, f'manifest missing {config}')


def main():
    for task, config, dataset_base, expected_src, expected_tgt in TASKS:
        check_no_duplicate_base_keys(config)
        cfg = load_config(config)
        config_text = (REPO_ROOT / config).read_text()
        assert_true(Path(dataset_base).name in config_text,
                    f'{task} does not inherit expected dataset base')
        check_data(cfg, task, expected_src, expected_tgt)
        check_model(cfg)
        check_uda(cfg)
        check_names(cfg, task)
        check_exclusions(cfg)
        print(f'{task}: SAME baseline data/schedule, DIFF-ALLOWED memory modules/discriminator')
    check_sources()
    check_manifest()
    print('PASS: MemoryAdaptNet-R101-3VI configs are ready.')


if __name__ == '__main__':
    main()

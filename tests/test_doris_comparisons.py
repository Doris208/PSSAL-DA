"""Checks for the two requested MiT-B5 comparison runs; no GPU required."""
import ast
import copy
from collections import OrderedDict
import hashlib
import importlib.util
import json
import math
import shlex
from types import SimpleNamespace
import unittest

import test_minimal_output as helpers
from test_minimal_output import ROOT, torch

MEMORY = 'configs/memoryadaptnet/doris_goji26IRRG2goji30RGB_memoryadaptnet_3vi.py'
DAFORMER = 'configs/daformer/doris_goji26RGB2goji30RGB_daformer_3vi.py'


def load_torch_source(path, scope):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8-sig'))
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, path, 'exec'), scope)
    return scope


def memory_components():
    registry = SimpleNamespace(register_module=lambda: lambda cls: cls)
    class Base(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
    class BackboneStandin(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Conv2d(6, 512, 1)
        def forward(self, image):
            return [self.proj(torch.nn.functional.avg_pool2d(image, 32))]
    scope = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, dist=torch.distributed,
                 math=math, optim=torch.optim, UDA=registry, MODELS=registry,
                 SEGMENTORS=registry, BaseSegmentor=Base,
                 auto_fp16=lambda **kw: lambda fn: fn,
                 builder=SimpleNamespace(build_backbone=lambda cfg: BackboneStandin()),
                 resize=lambda input, size, **kw: torch.nn.functional.interpolate(
                     input, size=size, mode='bilinear', align_corners=False))
    for path in ('mmseg/models/utils/memory_attention.py',
                 'mmseg/models/utils/domain_prototype_memory.py',
                 'mmseg/models/segmentors/memory_adapt_encoder_decoder.py',
                 'mmseg/models/discriminators/fc_discriminator.py'):
        load_torch_source(path, scope)
    parser = helpers.load_methods('mmseg/models/segmentors/base.py', 'BaseSegmentor',
        ['_parse_losses'], dict(torch=torch, dist=torch.distributed, OrderedDict=OrderedDict))
    class UdaBase(Base):
        _parse_losses = staticmethod(parser._parse_losses)
        def __init__(self, **cfg):
            super().__init__()
            self.model = cfg['model']
            self.num_classes = self.model.num_classes
        def forward(self, **data):
            return self.forward_train(**data)
    scope['UDADecorator'] = UdaBase
    load_torch_source('mmseg/models/uda/memory_adaptnet.py', scope)
    return scope


class ComparisonConfigTests(unittest.TestCase):
    def test_exact_two_methods_directions_channels_paths_and_20k(self):
        from tools.check_doris_stage1_configs import audit_comparison_configs
        rows = audit_comparison_configs(helpers.static_config)
        self.assertEqual(len(rows), 2)
        self.assertEqual({row['config'] for row in rows},
                         {MEMORY.split('/')[-1][:-3], DAFORMER.split('/')[-1][:-3]})
        for path, source in ((MEMORY, 'configs/memoryadaptnet/goji26IRRG2goji30RGB_memoryadaptnet_r101_3vi.py'),
                             (DAFORMER, 'configs/daformer/goji26RGB2goji30RGB_daformer_3vi.py')):
            cfg, old = helpers.static_config(path), helpers.static_config(source)
            self.assertEqual(cfg['data'], old['data'])
            self.assertEqual(cfg['lr_config'], old['lr_config'])
            self.assertEqual(cfg['checkpoint_config'], old['checkpoint_config'])
            self.assertEqual(cfg['evaluation'], old['evaluation'])
            self.assertEqual(cfg['model']['backbone']['type'], 'mit_b5')
            self.assertNotIn('depth', cfg['model']['backbone'])
            self.assertEqual(cfg['model']['backbone']['in_chans'], 6)
        cfg = helpers.static_config(MEMORY)
        uda = dict(cfg['uda'])
        self.assertTrue(uda.pop('log_train_accuracy'))
        self.assertEqual(uda, helpers.static_config(
            'configs/memoryadaptnet/goji26IRRG2goji30RGB_memoryadaptnet_r101_3vi.py')['uda'])
        self.assertEqual(cfg['model']['memory_module']['memory_dim'], 1024)
        self.assertEqual(cfg['model']['memory_module']['output_channels'], 2048)

    def test_batch_has_eight_jobs_and_additions_script_only_two(self):
        def commands(filename):
            raw = (ROOT / filename).read_bytes()
            self.assertNotIn(b'\r', raw)
            self.assertIn(b'set -e\n', raw)
            return [line for line in raw.decode().splitlines()
                    if line.startswith('python tools/train.py')]
        all_jobs = commands('train_doris_mitb5_stage1_20k.sh')
        added = commands('train_doris_mitb5_stage1_comparisons_20k.sh')
        self.assertEqual(len(all_jobs), 8)
        self.assertEqual(len(added), 2)
        self.assertEqual(all_jobs[-2:], added)
        self.assertEqual(len({shlex.split(line)[4] for line in all_jobs}), 8)
        self.assertEqual([shlex.split(line)[2] for line in added], [MEMORY, DAFORMER])
        for line in added:
            args = shlex.split(line)
            self.assertEqual(args[4], '$RUN_ROOT/' + args[2].split('/')[-1][6:-3])

    def test_original_configs_unchanged(self):
        path = ROOT / 'outputs/doris_comparisons_before.json'
        if not path.is_file():
            self.skipTest('Local comparison-task snapshot unavailable')
        snapshot = json.loads(path.read_text())
        for name, digest in snapshot['hashes'].items():
            if name.startswith('configs/'):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)

    @unittest.skipUnless(importlib.util.find_spec('mmcv'), 'MMCV unavailable locally')
    def test_real_mmcv_comparison_configs(self):
        from mmcv import Config
        from tools.check_doris_stage1_configs import audit_comparison_configs
        self.assertEqual(len(audit_comparison_configs(Config.fromfile)), 2)


@unittest.skipIf(torch is None, 'Torch CPU environment unavailable')
class MemoryCompatibilityTests(unittest.TestCase):
    def test_mit_feature_grid_matches_existing_heads_and_discriminator(self):
        cls = helpers.load_methods('mmseg/models/segmentors/memory_adapt_encoder_decoder.py',
                                   'MemoryAdaptEncoderDecoder', ['_stage_features'],
                                   dict(F=torch.nn.functional))
        model = cls()
        model.feature_upsample_scale = 4
        feature = torch.randn(2, 512, 8, 8)
        model.extract_feat = lambda image: [feature]
        _, stage4 = model._stage_features(torch.randn(2, 6, 256, 256))
        self.assertEqual(stage4.shape, (2, 512, 32, 32))

    def test_legacy_feature_path_is_identical_and_does_not_interpolate(self):
        cls = helpers.load_methods('mmseg/models/segmentors/memory_adapt_encoder_decoder.py',
                                   'MemoryAdaptEncoderDecoder', ['_stage_features'],
                                   dict(F=torch.nn.functional))
        model = cls()
        model.feature_upsample_scale = 1
        features = [torch.randn(2, 2048, 32, 32)]
        model.extract_feat = lambda image: features
        raw, stage4 = model._stage_features(None)
        self.assertIs(raw, features)
        self.assertIs(stage4, features[-1])

    def test_optional_accuracy_does_not_change_loss_or_gradient(self):
        cls = helpers.load_methods('mmseg/models/uda/memory_adaptnet.py',
            'MemoryAdaptNet', ['_stage_loss'], dict(F=torch.nn.functional,
                resize=lambda input, size, **kw: torch.nn.functional.interpolate(
                    input, size=size, mode='bilinear', align_corners=False)))
        labels = torch.arange(64).reshape(1, 1, 8, 8).remainder(5)
        labels[:, :, 0] = 255
        logits = torch.randn(1, 5, 8, 8)
        results = []
        for enabled in (False, True):
            model = cls()
            model.ignore_index, model.log_train_accuracy = 255, enabled
            values = logits.clone().requires_grad_()
            losses = model._stage_loss(values, labels, (8, 8), 1.0, 'loss_seg_stage1')
            losses['loss_seg_stage1'].backward()
            results.append((losses['loss_seg_stage1'], values.grad))
            self.assertEqual('stage1.acc_seg' in losses, enabled)
            if enabled:
                self.assertFalse(losses['stage1.acc_seg'].requires_grad)
        for old, new in zip(*results):
            torch.testing.assert_close(old, new, rtol=0, atol=0)

    def test_actual_memory_heads_discriminator_and_both_training_phases(self):
        scope = memory_components()
        config = helpers.static_config(MEMORY)
        model_cfg = copy.deepcopy(config['model'])
        model_cfg.pop('type')
        # Keep the real 512-channel MiT interface and real modules/control flow;
        # reduce only hidden widths for the CPU stand-in test.
        model_cfg['stage1_head']['channels'] = 4
        model_cfg['memory_bottleneck']['out_channels'] = 16
        model_cfg['memory_module'].update(memory_dim=16, transform_channels=8, output_channels=32)
        model_cfg['stage2_head']['in_channels'] = 32
        for step in (0, 3000):
            torch.manual_seed(67)
            segmentor = scope['MemoryAdaptEncoderDecoder'](**model_cfg)
            image = torch.randn(2, 6, 256, 256)
            output = segmentor.forward_memory(image)
            self.assertEqual(output['stage1_logits'].shape, (2, 5, 32, 32))
            self.assertEqual(output['stage2_logits'].shape, (2, 5, 32, 32))
            uda_cfg = copy.deepcopy(config['uda'])
            uda_cfg.pop('type')
            uda_cfg['discriminator']['base_channels'] = 4
            uda = scope['MemoryAdaptNet'](**uda_cfg, model=segmentor, max_iters=20000)
            uda.local_iter = step
            with self.assertRaises(RuntimeError):
                uda.discriminator(torch.rand(2, 5, 8, 8))
            optimizer = torch.optim.AdamW([
                dict(params=segmentor.parameters(), lr=6e-5),
                dict(params=uda.discriminator.parameters(), lr=0, weight_decay=0)], lr=6e-5)
            original_step = optimizer.step
            checked = []
            def step_without_double_update(*args, **kwargs):
                before = [p.detach().clone() for p in uda.discriminator.parameters()]
                result = original_step(*args, **kwargs)
                for old, param in zip(before, uda.discriminator.parameters()):
                    torch.testing.assert_close(old, param, rtol=0, atol=0)
                checked.append(True)
                return result
            optimizer.step = step_without_double_update
            before_discriminator = [p.detach().clone() for p in uda.discriminator.parameters()]
            label = torch.arange(256 * 256).reshape(1, 1, 256, 256).remainder(5).expand(2, -1, -1, -1)
            result = uda.train_step(dict(img=image, img_metas=[{}, {}],
                gt_semantic_seg=label, target_img=torch.randn_like(image),
                target_img_metas=[{}, {}]), optimizer)
            self.assertTrue(checked)
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in result['log_vars'].values()))
            self.assertEqual('loss_seg_stage2' in result['log_vars'], step >= 3000)
            self.assertIn('stage1.acc_seg', result['log_vars'])
            self.assertEqual('stage2.acc_seg' in result['log_vars'], step >= 3000)
            self.assertEqual(segmentor.memory_module.has_valid_memory(), step >= 3000)
            self.assertTrue(any(not torch.equal(old, new) for old, new in
                                zip(before_discriminator, uda.discriminator.parameters())))
            self.assertTrue(all(torch.isfinite(p).all() for p in uda.parameters()))
            segmentor.test_cfg = SimpleNamespace(**model_cfg['test_cfg'])
            segmentor.eval()
            with torch.no_grad():
                predictions = segmentor.simple_test(image[:1], [dict(ori_shape=(256, 256, 6))])
            self.assertEqual(predictions[0].shape, (256, 256))
            self.assertTrue(((predictions[0] >= 0) & (predictions[0] < 5)).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)

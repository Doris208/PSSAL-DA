"""Small-tensor mechanism tests. No dataset, checkpoint or GPU is required."""
import copy
import ast
import csv
import importlib.util
import json
from pathlib import Path
import random
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import test_minimal_output as helpers
from test_minimal_output import ROOT, torch


@unittest.skipIf(torch is None, 'Torch unavailable')
class MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('mechanism_test', ROOT / 'mmseg/utils/mechanism.py')
        cls.observation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.observation)

    def model(self):
        from test_doris_stage1 import new_model
        model = new_model()
        class Student(torch.nn.Module):
            last_train_crop_box = {0: (0, 16, 0, 16)}
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(6, 4, 1)
                self.bn = torch.nn.BatchNorm2d(4)
                self.drop = torch.nn.Dropout2d(0.2)
                self.classifier = torch.nn.Conv2d(4, 5, 1)
            def extract_feat(self, image):
                return [self.drop(self.bn(self.conv(image)))]
            def generate_pseudo_label(self, image, metas):
                type(self).last_train_crop_box = {1: (2, 4, 2, 4)}
                return self.classifier(self.extract_feat(image)[0])
        model.model = Student()
        model.ema_model = copy.deepcopy(model.model)
        model.model.drop.eval()  # Restore heterogeneous modes, not just root.training.
        label = torch.arange(256).reshape(1, 16, 16) % 5
        model.proto_gate.update_source(torch.randn(1, 4, 16, 16), label)
        model.struct_gate.update_source_adjacency(label)
        model.struct_gate.update_semantic_prototype(torch.randn(1, 5, 16, 16).softmax(1), label)
        model.local_iter = 16000
        return model

    def test_probe_preserves_parameters_buffers_gradients_optimizer_modes_and_rng(self):
        model = self.model()
        image, vi = torch.randn(1, 6, 16, 16), torch.rand(1, 3, 16, 16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        sum(p.sum() for p in model.parameters()).backward()
        optimizer.step()
        state = {key: value.clone() for key, value in model.state_dict().items()}
        gradients = {key: value.grad.clone() for key, value in model.named_parameters()}
        opt = copy.deepcopy(optimizer.state_dict())
        modes = [m.training for m in model.modules()]
        crops = copy.deepcopy(type(model.model).last_train_crop_box)
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        # Any attempt to refresh training priors in a probe must fail this test.
        model.struct_gate.update_source_adjacency = lambda *a: self.fail('adjacency update in probe')
        model.struct_gate.update_semantic_prototype = lambda *a: self.fail('semantic update in probe')
        model.proto_gate._refresh_fused_prototypes = lambda: self.fail('prototype refresh in probe')
        maps = self.observation.probe_arrays(model, image, vi, [{}], torch.ones(1, 1, 16, 16))
        np.testing.assert_allclose(maps['delta_w_proto'], maps['w_t_proto'] - maps['w_t_ps'])
        self.assertEqual(maps['w_t_ps'].shape, (16, 16))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        for key, value in model.named_parameters():
            torch.testing.assert_close(value.grad, gradients[key], rtol=0, atol=0)
        for key in opt['state']:
            for name, value in opt['state'][key].items():
                torch.testing.assert_close(value, optimizer.state_dict()['state'][key][name], rtol=0, atol=0)
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertEqual(crops, type(model.model).last_train_crop_box)
        self.assertEqual(rng[0], random.getstate())
        np.testing.assert_equal(rng[1], np.random.get_state())
        torch.testing.assert_close(rng[2], torch.get_rng_state(), rtol=0, atol=0)

    def test_probe_handles_half_physiology_response_with_float_pseudo_weights(self):
        model = self.model()
        with torch.no_grad():
            model.ema_model.classifier.weight.zero_()
            model.ema_model.classifier.bias.fill_(-5)
            model.ema_model.classifier.bias[0] = 5
        # Reproduce AMP's FP16 head output without requiring a CUDA convolution.
        handle = model.phy_head.register_forward_hook(
            lambda module, inputs, output: output.half())
        try:
            maps = self.observation.probe_arrays(model, torch.randn(1, 6, 16, 16),
                torch.rand(1, 3, 16, 16), [{}], torch.ones(1, 1, 16, 16))
        finally:
            handle.remove()
        self.assertEqual(maps['w_t_phy'].dtype, np.float32)
        self.assertTrue(np.isfinite(maps['w_t_phy']).all())
        self.assertTrue((maps['w_t_phy'] <= maps['w_base']).all())
        self.assertTrue((maps['w_t_phy'] < maps['w_base']).any())

    def test_rng_and_modes_restore_even_on_exception(self):
        model = self.model()
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        modes = [m.training for m in model.modules()]
        with self.assertRaises(RuntimeError):
            with self.observation.readonly_probe(model):
                random.random(); np.random.rand(); torch.rand(3)
                raise RuntimeError('fixture')
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertEqual(rng[0], random.getstate())
        np.testing.assert_equal(rng[1], np.random.get_state())
        torch.testing.assert_close(rng[2], torch.get_rng_state(), rtol=0, atol=0)

    def test_lazy_prototype_banks_restore_exactly_from_checkpoint(self):
        model = self.model()
        original = model.proto_gate
        saved = copy.deepcopy(original.state_dict())
        cls = helpers.component('mmseg/models/uda/proto_gate.py', 'ProtoGate')
        restored = cls(num_classes=5, separate_domain_prototypes=True)
        self.assertEqual(restored.prototypes.numel(), 0)
        restored.load_state_dict(saved, strict=True)
        for key, value in restored.state_dict().items():
            torch.testing.assert_close(value, saved[key], rtol=0, atol=0)

    def test_resume_restores_dacs_iteration_without_resetting_ema_schedule(self):
        tree = ast.parse((ROOT / 'mmseg/apis/train.py').read_text(encoding='utf-8'))
        method = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'train_segmentor')
        block = next(n for n in method.body if isinstance(n, ast.If)
                     and ast.dump(n.test) == ast.dump(ast.parse('cfg.resume_from', mode='eval').body))
        class Config(dict):
            __getattr__ = dict.__getitem__
        model = SimpleNamespace(local_iter=0)
        runner = SimpleNamespace(iter=0, model=SimpleNamespace(module=model))
        runner.resume = lambda path: setattr(runner, 'iter', 8500)
        runner.load_checkpoint = lambda path: None
        cfg = Config(resume_from='iter_8500.pth', load_from=None, uda=dict(type='DACS'))
        exec(compile(ast.Module(body=[block], type_ignores=[]), 'resume_fixture', 'exec'),
             dict(cfg=cfg, runner=runner))
        self.assertEqual(model.local_iter, 8500)

    def test_resume_corrects_mmcv_checkpoint_counter_for_numbered_and_latest_paths(self):
        tree = ast.parse((ROOT / 'mmseg/apis/train.py').read_text(encoding='utf-8'))
        method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_segmentor')
        block = next(n for n in method.body if isinstance(n, ast.If) and
                     ast.dump(n.test) == ast.dump(ast.parse('cfg.resume_from', mode='eval').body))
        class Config(dict):
            __getattr__ = dict.__getitem__
        class Runner:
            def __init__(self, saved_iter):
                self._iter = saved_iter
                self.model = SimpleNamespace(module=SimpleNamespace(local_iter=0))
            @property
            def iter(self):
                return self._iter
            def resume(self, path):
                pass
        for path, stored_iter in (('iter_5000.pth', 4999), ('latest.pth', 4999), ('iter_5000.pth', 5000)):
            with self.subTest(path=path, stored_iter=stored_iter):
                runner = Runner(stored_iter)
                cfg = Config(resume_from=path, load_from=None, uda=dict(type='DACS'),
                             runner=dict(type='IterBasedRunner'))
                path_api = SimpleNamespace(realpath=lambda p: '/work/iter_5000.pth',
                                           basename=lambda p: p.rsplit('/', 1)[-1])
                exec(compile(ast.Module(body=[block], type_ignores=[]), 'resume_fixture', 'exec'),
                     dict(cfg=cfg, runner=runner, osp=path_api, re=re,
                          logger=SimpleNamespace(info=lambda *a: None)))
                self.assertEqual(runner.iter, 5000)
                self.assertEqual(runner.model.module.local_iter, 5000)

    def test_proto_diagnostic_spatial_mask_excludes_invalid_cells_without_update_changes(self):
        cls = helpers.component('mmseg/models/uda/proto_gate.py', 'ProtoGate')
        a = cls(tau_proto=.25, tau_conf_proto=.88, min_pixels=8, target_update_start=0)
        b = copy.deepcopy(a)
        b.collect_diagnostics = True
        b.diagnostic_spatial_mask = torch.ones(1, 1, 4, 8)
        b.diagnostic_spatial_mask[:, :, :, :2] = 0
        feat, label = torch.randn(1, 3, 4, 8), torch.zeros(1, 4, 8).long()
        weight = b.diagnostic_spatial_mask[:, 0].clone()
        for gate in (a, b):
            gate.update_target(feat, label, torch.ones_like(weight), weight, iter=10)
        self.assertEqual(b.last_target_diagnostics['class_0.candidate_count'], 24)
        self.assertEqual(b.last_target_diagnostics['class_0.updated_count'], 24)
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)

    def test_collector_uses_cli_workdir_deterministic_unlabeled_sample_and_resume(self):
        model = self.model()
        image, vi = torch.ones(6, 16, 16), torch.zeros(3, 16, 16)
        captured = []
        class Dataset:
            img_infos = [dict(filename='z.tif'), dict(filename='a.tif')]
            def __getitem__(self, index):
                captured.append(index)
                return dict(img=SimpleNamespace(data=image), vi_img=SimpleNamespace(data=vi),
                    img_metas=SimpleNamespace(data=dict(filename=self.img_infos[index]['filename'])))
        def build_dataset(config):
            self.assertIsNone(config['ann_dir'])
            self.assertFalse(any('Random' in step['type'] or step['type'] == 'LoadAnnotations'
                                 for step in config['pipeline']))
            return Dataset()
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as folder:
            config = dict(model=dict(train_cfg={}), uda=dict(fixed_probe_index=0),
                target_img_norm_cfg=dict(mean=[0]*6, std=[1]*6, to_rgb=False),
                data=dict(train=dict(target=dict(type='GojiDataset', pipeline=[]))), seed=0)
            runner = SimpleNamespace(work_dir=folder, iter=0, mechanism_config=config,
                meta=dict(git_commit='fixture'), logger=SimpleNamespace(info=lambda *args: None))
            with patch.dict(sys.modules, {'mmseg.datasets': SimpleNamespace(build_dataset=build_dataset)}):
                collector = self.observation.MechanismCollector(runner, model)
                self.assertEqual(collector.root, Path(folder) / 'analysis')
                self.assertEqual(captured, [1])  # Stable lexicographic sample a.tif.
                runner.iter = 500
                config['resume_from'] = 'iter_500.pth'
                self.observation.MechanismCollector(runner, model)
                description = json.loads((collector.root / 'fixed_probe/sample_manifest.json').read_text())
                self.assertEqual(description['sample_id'], 'a.tif')
                config['seed'] = 99
                with self.assertRaises(ValueError):
                    self.observation.MechanismCollector(runner, model)

    def test_baseline_cannot_create_fake_mechanism_outputs(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as folder:
            baseline = SimpleNamespace(enable_phy=False)
            with self.assertRaises(ValueError):
                self.observation.MechanismCollector(SimpleNamespace(work_dir=folder), baseline)
            self.assertFalse((Path(folder) / 'analysis').exists())

    def test_full_only_schema_is_preserved_when_collecting_module_schema(self):
        model = self.model()
        model.mechanism_cfg.update(probe_enabled=False, online_plot=False)
        config = dict(model=dict(train_cfg={}), uda={}, target_img_norm_cfg={},
                      data=dict(train=dict(target={})))
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as folder:
            root = Path(folder) / 'analysis'
            root.mkdir()
            previous = root / 'manifest.json'
            previous.write_text('{"schema_version": 1, "historical_data": true}')
            content = previous.read_bytes()
            runner = SimpleNamespace(work_dir=folder, iter=0, mechanism_config=config,
                meta={}, logger=SimpleNamespace(info=lambda *args: None))
            collector = self.observation.MechanismCollector(runner, model)
            self.assertEqual(collector.root, root / 'branches_v4')
            self.assertEqual(previous.read_bytes(), content)
            self.assertEqual(json.loads((collector.root / 'manifest.json').read_text())['schema_version'], 4)

    def test_compact_physiology_snapshots_have_required_fields_without_generic_duplicates(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as folder:
            obj = self.collector(folder)
            obj.model, obj.identity, obj.physiology = self.model(), 'compact-fixture', True
            obj.model.enable_struct = obj.model.enable_proto = obj.model.enable_struct_edge = False
            obj.model.struct_gate = obj.model.proto_gate = None
            image = torch.randn(1, 6, 16, 16)
            obj.sample = (image, torch.rand(1,3,16,16), [dict(filename='fixed.tif',
                img_norm_cfg=dict(mean=[0]*6,std=[1]*6))])
            obj.probe_cfg = dict(margins=[1,1,1,1])
            (obj.root/'fixed_probe').mkdir()
            (obj.root/'fixed_probe/sample_manifest.json').write_text(json.dumps(dict(tensor_sha256='fixed')))
            for step in (500,1000):
                obj.model.local_iter = step
                obj.snapshot(step)
            paths = sorted((obj.root/'physiology/arrays').glob('*_phy.npz'))
            self.assertEqual(len(paths),2)
            self.assertEqual(len(list((obj.root/'fixed_probe/arrays').glob('*.npz'))),0)
            self.assertEqual(len(list(obj.root.rglob('*.png'))),0)
            with np.load(paths[0]) as a, np.load(paths[1]) as b:
                for key in ('r_phy','m_phy','w_before_phy','w_t_phy','delta_w_phy','valid_mask'):
                    self.assertEqual(a[key].shape,(16,16))
                np.testing.assert_array_equal(a['target_image'],b['target_image'])
                np.testing.assert_array_equal(a['delta_w_phy'],a['w_t_phy']-a['w_before_phy'])
                self.assertTrue(json.loads(str(a['metadata']))['physiology_compact'])
            obj.snapshot(1000)
            self.assertEqual(len(list((obj.root/'physiology/arrays').glob('*_phy.npz'))),2)

    def test_physiology_csv_has_correct_group_denominators_and_actual_loss(self):
        model=self.model()
        model.mechanism_cfg['output_mode']='detailed'
        model.enable_struct=model.enable_proto=model.enable_struct_edge=False
        model.struct_gate=model.proto_gate=None
        model.mechanism_phy_source={'vegetation':(.3,1),'nonvegetation':(.8,2)}
        label=torch.tensor([[[0,1,2,255]]])
        before=torch.tensor([[[1.,1.,1.,0.]]]); after=torch.tensor([[[.8,1.,1.,0.]]])
        valid=label.ne(255)
        record=self.observation.training_record(model,torch.ones_like(before),label,after,after,
            valid,None,None,None,w_base=before,w_t_phy=after,
            r_phy=torch.tensor([[[[.2,.6,.8,.5]]]]),log_vars={'loss_phy_src':.5})
        record['phy_losses']['loss_phy_tgt']=.008
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            obj=self.collector(folder);obj.pending=[record];obj.summarize(500)
            with (obj.root/'metrics/mechanism_500.csv').open() as stream:
                row=next(csv.DictReader(stream))
            self.assertAlmostEqual(float(row['r_phy_source_vegetation_mean']),.3)
            self.assertAlmostEqual(float(row['r_phy_source_nonvegetation_mean']),.4)
            self.assertAlmostEqual(float(row['r_phy_target_vegetation_pred_mean']),.4)
            self.assertAlmostEqual(float(row['r_phy_target_nonvegetation_pred_mean']),.8)
            self.assertEqual(float(row['phy_downweight_ratio']),.5)
            self.assertEqual(float(row['loss_phy_tgt']),.008)
            self.assertEqual(row['phy_target_supervision_mode'],'balanced_high_conf')

    def test_all_module_combinations_collect_only_enabled_evidence(self):
        from tools.check_doris_stage1_configs import VARIANTS
        for suffix, flags in VARIANTS[1:]:
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as folder:
                model = self.model()
                model.mechanism_cfg['output_mode']='detailed'
                for name, enabled in zip(('phy', 'struct', 'proto', 'struct_edge'), flags):
                    setattr(model, 'enable_' + name, enabled)
                if not flags[1]: model.struct_gate = None
                if not flags[2]:
                    model.proto_gate = None
                    model.model.extract_feat = lambda *a: self.fail('unneeded Student forward')
                if not flags[3]:
                    del model.struct_edge_ignore_index, model.struct_edge_num_classes
                    model.doris_edge_responses = lambda *a: self.fail('disabled boundary forward')
                image, vi = torch.randn(1, 6, 16, 16), torch.rand(1, 3, 16, 16)
                valid = torch.ones(1, 1, 16, 16)
                valid[:, :, :2] = 0
                if flags[2]:
                    gate = model.proto_gate
                    gate.collect_diagnostics = True
                    gate.update_target(torch.randn(1, 4, 16, 16),
                        torch.zeros(1, 16, 16).long(), torch.ones(1, 16, 16),
                        torch.ones(1, 16, 16), iter=model.local_iter)
                    model.local_iter = gate.weight_start - 1
                    before = self.observation.probe_arrays(model, image, vi, [{}], valid)
                    np.testing.assert_array_equal(before['w_t_proto'], before['w_pre_proto'])
                    self.assertFalse(before['proto_active'])
                    model.local_iter = gate.weight_start + 1
                state = {k: v.clone() for k, v in model.state_dict().items()}
                rng = torch.get_rng_state().clone()
                maps = self.observation.probe_arrays(model, image, vi, [{}], valid)
                for key, enabled in zip(('r_phy', 'r_struct', 'w_t_proto', 'r_edge'), flags):
                    self.assertEqual(key in maps, enabled)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, state[key], rtol=0, atol=0)
                torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
                tensor = lambda key: torch.from_numpy(maps[key]).unsqueeze(0) if key in maps else None
                structural = {'R_struct': tensor('r_struct')} if flags[1] else None
                components = {source: tensor(key) for key, source in
                    (('e_prob', 'E_prob'), ('e_img', 'E_img'), ('r_pair', 'R_pair'))} if flags[3] else None
                record = self.observation.training_record(model, tensor('w_conf'),
                    tensor('pseudo_label'), tensor('w_pre_proto') if flags[2] else
                    tensor('w_t_ps') if flags[0] or flags[1] else tensor('w_base'),
                    tensor('w_final'), tensor('valid_mask'), structural,
                    tensor('r_edge'), tensor('boundary_weight'),
                    w_base=tensor('w_base'), w_t_phy=tensor('w_t_phy'),
                    r_phy=tensor('r_phy'), edge_components=components)
                obj = self.collector(folder)
                obj.pending = [record]
                obj.summarize(500)
                self.assertEqual((Path(folder) / 'metrics/proto_selection_500.csv').exists(), flags[2])
                with (Path(folder) / 'metrics/mechanism_500.csv').open() as stream:
                    row = next(csv.DictReader(stream))
                for key, enabled in zip(('r_phy', 'r_struct', 'w_t_proto', 'r_edge'), flags):
                    self.assertEqual(row[key + '_mean_valid'] != '', enabled)
                if flags[2] and not flags[0] and not flags[1]:
                    self.assertEqual(row['w_t_ps_mean_valid'], '')
                    np.testing.assert_array_equal(maps['w_pre_proto'], maps['w_base'])

    def test_deterministic_pipeline_collect_does_not_require_random_flip_metadata(self):
        cfg = dict(uda={}, data=dict(train=dict(target={})),
                   target_img_norm_cfg=dict(mean=[0]*6, std=[1]*6, to_rgb=False))
        pipeline = self.observation.deterministic_probe_config(cfg)['dataset']['pipeline']
        args = dict(pipeline[-1])
        args.pop('type')
        collect = helpers.load_methods('mmseg/datasets/pipelines/formating.py',
            'Collect', ['__init__', '__call__'],
            dict(DC=lambda value, **kw: SimpleNamespace(data=value)))(**args)
        values = {key: 'fixture' for key in args['meta_keys']}
        values.update(img=object(), vi_img=object())
        result = collect(values)
        self.assertIn('img_norm_cfg', result['img_metas'].data)
        self.assertNotIn('flip', result['img_metas'].data)

    def collector(self, root):
        obj = object.__new__(self.observation.MechanismCollector)
        obj.root, obj.pending, obj.window_start = Path(root), [], 0
        obj.cfg = dict(sample_interval=100, summary_interval=500, probe_interval=500,
                       probe_enabled=False, save_raw=True, make_figures=False, output_mode='detailed')
        obj.runner = SimpleNamespace(iter=0, logger=SimpleNamespace(info=lambda *a: None))
        obj.model = SimpleNamespace(mechanism_records=[])
        return obj

    def record(self, step, valid, predicted, selected):
        return dict(iter=step, valid_count=valid, total_count=10, high_conf_count=valid // 2,
            sums={key: valid * 0.4 for key in ('w_conf', 'w_t_ps', 'w_t_proto',
                                               'r_struct', 'r_edge', 'boundary_weight')},
            tau_proto=0.25, tau_conf_proto=0.88, proto_valid_class_count=5,
            proto_update_active=1, proto_weight_active=1,
            classes=[dict(class_id=cls, class_name=str(cls), prototype_initialized=1,
                predicted_count=predicted, confidence_pass_count=selected,
                weight_pass_count=selected, joint_selected_count=selected,
                updated_count=selected) for cls in range(5)])

    def test_500_1000_csv_weighted_counts_and_resume_do_not_duplicate(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as folder:
            obj = self.collector(folder)
            for step in range(100, 1001, 100):
                obj.model.mechanism_records.append(self.record(step, 5 if step == 100 else 10,
                                                               1 if step == 100 else 10, 1))
                obj.runner.iter = step - 1
                obj.consume()
            path = Path(folder) / 'metrics/proto_selection_500.csv'
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 10)
            self.assertAlmostEqual(float(rows[0]['selected_ratio']), 5 / 41)
            with (Path(folder) / 'metrics/mechanism_500.csv').open() as stream:
                mechanism = list(csv.DictReader(stream))
            self.assertEqual([r['iter'] for r in mechanism], ['500', '1000'])
            self.assertEqual(int(mechanism[0]['n_sampled_batches']), 5)
            self.assertAlmostEqual(float(mechanism[0]['w_conf_mean_valid']), 0.4)
            self.observation.append_unique_csv(path, rows, ['iter', 'class_id'])
            with path.open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 10)
            for step in range(1100, 20001, 100):
                obj.model.mechanism_records.append(self.record(step, 10, 10, 1))
                obj.runner.iter = step - 1
                obj.consume()
            with path.open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 200)
            with (Path(folder) / 'metrics/mechanism_500.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 40)
            changed = dict(rows[0], updated_count='99')
            with self.assertRaises(ValueError):
                self.observation.append_unique_csv(path, [changed], ['iter', 'class_id'])
            with path.open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 200)

    def test_snapshot_same_sample_iteration_state_and_changed_state_conflict(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as folder:
            obj = self.collector(folder)
            obj.model, obj.identity = self.model(), 'fixture-config'
            image = torch.randn(1, 6, 16, 16)
            metas = [dict(filename='synthetic_target.tif', img_norm_cfg=dict(mean=[0]*6, std=[1]*6))]
            obj.sample = (image, torch.rand(1, 3, 16, 16), metas)
            obj.probe_cfg = dict(margins=[1, 1, 1, 1])
            (obj.root / 'fixed_probe').mkdir()
            (obj.root / 'fixed_probe/sample_manifest.json').write_text(json.dumps(dict(tensor_sha256='fixed')))
            for completed in (500, 1000):
                obj.model.local_iter = completed
                obj.snapshot(completed)
            paths = sorted((obj.root / 'fixed_probe/arrays').glob('*.npz'))
            self.assertEqual(len(paths), 2)
            with np.load(paths[0]) as a, np.load(paths[1]) as b:
                np.testing.assert_array_equal(a['target_image'], b['target_image'])
                self.assertFalse(a['valid_mask'][0].any())
                self.assertEqual(json.loads(str(a['metadata']))['iter'], 500)
            obj.snapshot(1000)  # Same state is safely reused.
            with torch.no_grad(): next(obj.model.parameters()).add_(1)
            with self.assertRaises(ValueError): obj.snapshot(1000)


if __name__ == '__main__':
    unittest.main()

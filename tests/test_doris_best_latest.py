"""Real MMCV 1.4 CPU serialization, replacement failures and interrupted training."""
import ast
import copy
import json
import logging
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import IterBasedRunner
from mmcv.runner.iter_based_runner import IterLoader

from mmseg.core.evaluation.eval_hooks import EvalHook, DistEvalHook
from mmseg.utils import doris_checkpoint as storage
from mmseg.utils.checkpoint_policy import DorisCheckpointHook
from tools.doris_checkpoint_plan import configure_checkpoint_policy, check_doris_resume_checkpoint
from test_minimal_output import ROOT, dacs_component


class Dataset(torch.utils.data.Dataset):
    CLASSES = ('goji', 'vegetation', 'building', 'water', 'road')
    def __len__(self): return 2
    def __getitem__(self, index): return dict(index=torch.tensor(index))
    def evaluate(self, results, **kwargs):
        self.evaluations = getattr(self, 'evaluations', 0) + 1
        return results


class BestLatestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / 'outputs')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runner = self.make_runner(self.root)
        self.hook = EvalHook(torch.utils.data.DataLoader(Dataset()), interval=1000,
            save_best='mIoU', doris_best_only=True)
        self.hook.before_run(self.runner)

    def make_runner(self, root, model=None, meta=None):
        model = model or torch.nn.Linear(2, 2)
        model.train_step = lambda *args, **kwargs: dict(log_vars={}, num_samples=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5)
        return IterBasedRunner(model, optimizer=optimizer, work_dir=str(root),
            logger=logging.getLogger('checkpoint-test'), max_iters=15000,
            meta=meta or dict(config='name = "synthetic"\n', git_commit='synthetic'))

    def validate(self, step, scores, hook=None, runner=None):
        runner = runner or self.runner
        hook = hook or self.hook
        runner._iter = step - 1
        runner.log_buffer.clear()
        key = hook.evaluate(runner, scores)
        hook._save_ckpt(runner, key)
        path = Path(runner.work_dir) / 'checkpoint_best.json'
        return json.loads(path.read_text()) if path.exists() else {}

    def scores(self, value):
        return {key: value for key in storage.METRICS}

    def files(self):
        return {path.name: path.read_bytes() for path in self.root.glob('*.pth')}

    def test_four_independent_metrics_zero_ties_missing_and_nonfinite(self):
        with patch.object(self.runner, 'save_checkpoint', wraps=self.runner.save_checkpoint) as save:
            report = self.validate(1000, self.scores(np.float32(0)))
            self.assertEqual(save.call_count, 1)
            self.assertEqual(len(list(self.root.glob('best_*.pth'))), 4)
            paths = [Path(report[key]['path']) for key in storage.METRICS]
            self.assertTrue(all(os.path.samefile(paths[0], path) for path in paths))
            for step, key in enumerate(storage.METRICS, 2):
                self.validate(step * 1000, {key: np.float64(.1 * step)})
            self.assertEqual(save.call_count, 5)
            report = self.validate(6000, {key: float('nan') for key in storage.METRICS})
            self.validate(7000, self.scores(float('inf')))
            self.validate(8000, {key: record['score'] for key, record in report.items() if key in storage.METRICS})
            self.validate(9000, {})
            self.assertEqual(save.call_count, 5)
        self.assertEqual([report[key]['iter'] for key in storage.METRICS], [2000, 3000, 4000, 5000])
        self.assertEqual(len(self.files()), 4)
        for path in self.root.glob('*.pth'):
            checkpoint = torch.load(path, map_location='cpu')
            self.assertNotIn('optimizer', checkpoint)
            self.assertFalse(checkpoint['meta']['doris_checkpoint']['resumable'])
            for key, value in self.runner.model.state_dict().items():
                torch.testing.assert_close(checkpoint['state_dict'][key], value)

    def test_fifteen_k_has_at_most_four_bests_latest_and_three_fixed_snapshots(self):
        latest = DorisCheckpointHook(interval=2000, by_epoch=False)
        self.runner.register_hook(latest, priority='LOW')
        latest.before_run(self.runner)
        with patch.object(self.runner, 'save_checkpoint', wraps=self.runner.save_checkpoint) as save:
            for step in range(1, 15001):
                self.runner._iter = step - 1
                if step % 1000 == 0:
                    self.validate(step, self.scores(step / 20000))
                latest.after_train_iter(self.runner)
                self.assertLessEqual(len(list(self.root.glob('*.pth'))), 8)
            self.assertEqual(save.call_count, 15 + 7)
        self.assertEqual({path.name for path in self.root.glob('iter_*.pth')},
                         {'iter_2000.pth', 'iter_4000.pth', 'iter_6000.pth'})
        for step in (2000, 4000, 6000):
            saved = torch.load(self.root / 'iter_{}.pth'.format(step))
            self.assertEqual(saved['meta']['doris_checkpoint']['completed_iter'], step)
            self.assertTrue(saved['meta']['doris_checkpoint']['resumable'])
            self.assertIn('optimizer', saved)
        self.assertFalse((self.root / 'last.pth').exists())
        self.assertFalse(list(self.root.glob('.doris-*.pth')))
        latest_state = torch.load(self.root / 'latest.pth')
        self.assertEqual(latest_state['meta']['doris_checkpoint']['completed_iter'], 14000)
        self.assertEqual(latest_state['meta']['hook_msgs']['best_miou_iter'], 14000)
        self.assertIn('optimizer', latest_state)

    def test_failed_saves_keep_files_manifest_and_in_memory_records(self):
        self.validate(1000, self.scores(.1))
        before, report = self.files(), (self.root / 'checkpoint_best.json').read_bytes()
        messages = copy.deepcopy(self.runner.meta['hook_msgs'])
        original_json = storage.atomic_json
        def fail_report(path, value):
            if path.name == 'checkpoint_best.json': raise OSError('JSON write failed')
            return original_json(path, value)
        failures = [
            patch('mmseg.utils.checkpoint_policy.shutil.disk_usage', return_value=SimpleNamespace(free=0)),
            patch.object(self.runner, 'save_checkpoint', side_effect=OSError('serializer failed')),
            patch.object(storage, 'atomic_json', side_effect=fail_report),
        ]
        for failure in failures:
            with failure, self.assertRaises(OSError):
                self.validate(2000, self.scores(.2))
            self.assertEqual(self.files(), before)
            self.assertEqual((self.root / 'checkpoint_best.json').read_bytes(), report)
            self.assertEqual(self.runner.meta['hook_msgs'], messages)
            self.assertFalse((self.root / storage.JOURNAL).exists())
            self.assertFalse(list(self.root.glob('.doris-*.pth')))

    def test_hardlink_fallback_is_one_traceable_file_and_same_iter_retry_is_safe(self):
        with patch.object(storage.os, 'link', side_effect=OSError('hard links unavailable')):
            report = self.validate(1000, self.scores(.1))
        self.assertEqual(len(self.files()), 1)
        self.assertEqual(len({report[key]['path'] for key in storage.METRICS}), 1)
        self.assertTrue(all(report[key]['link_error'] for key in storage.METRICS))
        old_goji = torch.load(report['IoU.goji']['path'])['state_dict']
        with torch.no_grad(): self.runner.model.weight.add_(1)
        report = self.validate(1000, {'mIoU': .2})
        self.assertEqual(len(self.files()), 2)
        self.assertNotEqual(report['mIoU']['path'], report['IoU.goji']['path'])
        for key, value in old_goji.items():
            torch.testing.assert_close(torch.load(report['IoU.goji']['path'])['state_dict'][key], value)
        self.assertFalse(list(self.root.glob('.doris-*.pth')))

    def test_partial_or_unreadable_temporary_file_preserves_old_best(self):
        self.validate(1000, self.scores(.1))
        before = self.files()
        report = (self.root / 'checkpoint_best.json').read_bytes()
        def partial(out_dir, filename_tmpl, **kwargs):
            (Path(out_dir) / filename_tmpl).write_bytes(b'partial checkpoint')
            raise OSError('temporary write failed')
        def unreadable(out_dir, filename_tmpl, **kwargs):
            (Path(out_dir) / filename_tmpl).write_bytes(b'not a checkpoint')
        for writer in (partial, unreadable):
            with patch.object(self.runner, 'save_checkpoint', side_effect=writer), self.assertRaises(Exception):
                self.validate(2000, self.scores(.2))
            self.assertEqual(self.files(), before)
            self.assertEqual((self.root / 'checkpoint_best.json').read_bytes(), report)
            self.assertFalse(list(self.root.glob('.doris-*.pth')))

    def test_same_iter_replacement_json_failure_restores_all_original_paths(self):
        with patch.object(storage.os, 'link', side_effect=OSError('unsupported')):
            self.validate(1000, self.scores(.1))
        before = self.files()
        report = (self.root / 'checkpoint_best.json').read_bytes()
        original = storage.atomic_json
        def fail_report(path, value):
            if path.name == 'checkpoint_best.json': raise OSError('manifest failed')
            return original(path, value)
        with patch.object(storage, 'atomic_json', side_effect=fail_report), self.assertRaises(OSError):
            self.validate(1000, {'mIoU': .2})
        self.assertEqual(self.files(), before)
        self.assertEqual((self.root / 'checkpoint_best.json').read_bytes(), report)

    def test_missing_validation_field_cannot_reuse_an_old_log_buffer_value(self):
        self.validate(1000, self.scores(.1))
        self.runner._iter = 1999
        self.runner.log_buffer.output.update(self.scores(.9))
        with patch.object(self.runner, 'save_checkpoint', wraps=self.runner.save_checkpoint) as save:
            score = self.hook.evaluate(self.runner, {})
            self.hook._save_ckpt(self.runner, score)
            self.assertEqual(save.call_count, 0)

    def test_rolling_latest_metadata_and_all_best_names_resolve_in_replay(self):
        from tools.replay_doris_mechanisms import checkpoint_iteration
        from tools.replay_hphy_mechanism import resolve_iteration
        self.validate(2000, self.scores(.1))
        storage.save_latest(self.runner)
        for path in self.root.glob('*.pth'):
            meta = torch.load(path)['meta']
            self.assertEqual(checkpoint_iteration(path, meta)['completed_iter'], 2000)
            self.assertEqual(resolve_iteration(path, meta)['completed_iter'], 2000)

    def test_best_contains_student_teacher_and_every_enabled_branch_without_optimizer(self):
        cfg = Config.fromfile(str(ROOT / 'configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py'))
        cls = dacs_component()
        model = cls(**cfg.uda, model=cfg.model, max_iters=15000)
        model.local_iter = 6000
        label = torch.arange(256).reshape(1, 16, 16) % 5
        feature = torch.rand(1, 5, 16, 16)
        model.proto_gate.update_source(feature, label)
        model.proto_gate.update_target(feature, label, torch.ones_like(label).float(),
                                      torch.ones_like(label).float(), iter=6000)
        model.struct_gate.update_source_adjacency(label)
        model.struct_gate.update_semantic_prototype(feature.softmax(1), label)
        model._update_boundary_pair_prior(label)
        runner = self.make_runner(self.root / 'full', model,
            dict(config=cfg.pretty_text, synthetic_fixture=True))
        runner._iter = 5999
        report = storage.save_bests(runner, self.scores(.2))
        saved = torch.load(report['mIoU']['path'])
        self.assertNotIn('optimizer', saved)
        self.assertEqual(set(saved['state_dict']), set(model.state_dict()))
        for prefix in ('model.', 'ema_model.', 'phy_head.', 'struct_gate.', 'proto_gate.', 'struct_edge_pair_prior'):
            self.assertTrue(any(key.startswith(prefix) for key in saved['state_dict']), prefix)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(saved['state_dict'][key], value, rtol=0, atol=0)

    def test_latest_atomic_replace_failure_keeps_the_previous_complete_file(self):
        self.validate(2000, self.scores(.1))
        self.runner.meta['fp16'] = dict(loss_scaler=dict(scale=128., growth_tracker=4))
        storage.save_latest(self.runner)
        before = (self.root / 'latest.pth').read_bytes()
        original = storage.os.replace
        def fail_latest(source, destination):
            if Path(destination).name == 'latest.pth': raise OSError('latest replace failed')
            return original(source, destination)
        self.runner._iter = 3999
        with patch.object(storage.os, 'replace', side_effect=fail_latest), self.assertRaises(OSError):
            storage.save_latest(self.runner)
        self.assertEqual((self.root / 'latest.pth').read_bytes(), before)
        self.assertEqual(torch.load(self.root / 'latest.pth')['meta']['fp16'], self.runner.meta['fp16'])
        self.assertFalse(list(self.root.glob('.doris-*.pth')))

    def test_startup_recovers_owned_pending_files_and_leaves_foreign_files(self):
        self.validate(1000, self.scores(.1))
        before = self.files()
        policy = self.runner.meta['doris_checkpoint_policy']
        journal = storage.new_journal(policy['owner'], 'best')
        journal['new_paths'] = ['best_mIoU_iter_2000.pth']
        storage.atomic_json(self.root / storage.JOURNAL, journal)
        (self.root / journal['temporary']).write_bytes(b'partial')
        (self.root / journal['new_paths'][0]).write_bytes(b'uncommitted')
        (self.root / 'checkpoint_best.json.tmp').write_bytes(b'partial JSON')
        (self.root / (storage.JOURNAL + '.tmp')).write_bytes(b'partial journal')
        foreign = self.root / 'foreign.pth'
        foreign.write_bytes(b'preserve')
        storage.initialize_policy(self.runner)
        foreign.unlink()
        self.assertEqual(self.files(), before)
        self.assertFalse((self.root / storage.JOURNAL).exists())
        self.assertFalse(list(self.root.glob('*.json.tmp')))

    def test_single_distributed_and_non_main_rank_evaluate_once(self):
        for hook_type in (EvalHook, DistEvalHook):
            root = self.root / hook_type.__name__
            runner = self.make_runner(root)
            dataset = Dataset()
            options = dict(interval=1000, save_best='mIoU', doris_best_only=True)
            if hook_type is DistEvalHook: options['broadcast_bn_buffer'] = False
            hook = hook_type(torch.utils.data.DataLoader(dataset), **options)
            hook.before_run(runner)
            runner._iter = 999
            inference = unittest.mock.Mock(return_value=self.scores(0.))
            with patch('mmseg.apis.single_gpu_test', inference), patch('mmseg.apis.multi_gpu_test', inference):
                hook._do_evaluate(runner)
            self.assertEqual(inference.call_count, 1)
            self.assertEqual(dataset.evaluations, 1)
            self.assertEqual(len(list(root.glob('best_*.pth'))), 4)
            runner._rank = 1
            with patch.object(runner, 'save_checkpoint', side_effect=AssertionError('Non-main write')):
                hook._save_ckpt(runner, .9)


class ResumeTests(unittest.TestCase):
    def test_complete_model_optimizer_lr_counters_and_module_state_continue_identically(self):
        torch.set_num_threads(1)
        cls = dacs_component()
        config_path = ROOT / 'configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py'
        cfg = Config.fromfile(str(config_path))
        configure_checkpoint_policy(cfg)
        class SmallDACS(cls):
            def train_step(self, data, optimizer, **kwargs):
                optimizer.zero_grad()
                image = torch.randn(1, 6, 16, 16) * (.5 + random.random() + float(np.random.rand()))
                label = torch.arange(256).reshape(1, 16, 16) % 5
                if self.local_iter == 0: self._init_ema_weights()
                else: self._update_ema(self.local_iter)
                logits = self.get_model()(image)
                loss = logits.square().mean()
                vi = torch.rand(1, 3, 16, 16)
                _, phy_loss = self._compute_phy_src_loss(vi, label[:, None], {})
                (loss + phy_loss).backward()
                optimizer.step()
                self.struct_gate.update_source_adjacency(label)
                self.struct_gate.update_semantic_prototype(logits.detach().softmax(1), label)
                self.proto_gate.update_source(logits.detach(), label)
                self.proto_gate.update_target(logits.detach(), label,
                    torch.ones(1, 16, 16), torch.ones(1, 16, 16), iter=self.local_iter)
                self._update_boundary_pair_prior(label)
                self.local_iter += 1
                return dict(log_vars=dict(loss=float(loss.detach())), num_samples=1)

        class CPUResumeRunner(IterBasedRunner):
            def resume(self, checkpoint, **kwargs):
                return super().resume(checkpoint, map_location='cpu')

        def make_runner(root):
            model = SmallDACS(**cfg.uda, model=cfg.model, max_iters=15000)
            optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5)
            runner = CPUResumeRunner(model, optimizer=optimizer, work_dir=str(root),
                logger=logging.getLogger('resume-test'), max_iters=15000,
                meta=dict(config=cfg.pretty_text, fp16=dict(loss_scaler=dict(scale=256., growth_tracker=3))))
            runner.register_training_hooks(copy.deepcopy(cfg.lr_config), None, None, None)
            runner.register_hook(DorisCheckpointHook(interval=2000, by_epoch=False), priority='LOW')
            return runner

        tree = ast.parse((ROOT / 'mmseg/apis/train.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'train_segmentor')
        block = next(node for node in function.body if isinstance(node, ast.If) and
                     ast.dump(node.test) == ast.dump(ast.parse('cfg.resume_from', mode='eval').body))
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            for start in (1998, 3998, 5998):
                root = Path(temporary) / str(start)
                torch.manual_seed(8)
                random.seed(8)
                np.random.seed(8)
                original = make_runner(root)
                original._iter = start
                original.model.local_iter = start
                original.call_hook('before_run')
                original.call_hook('before_epoch')
                fixed_batches = [dict(index=torch.tensor(0)) for _ in range(16)]
                loader = IterLoader(torch.utils.data.DataLoader(fixed_batches, generator=torch.Generator().manual_seed(1)))
                original.train(loader)
                original.train(loader)
                latest = root / 'latest.pth'
                saved = torch.load(latest)
                self.assertEqual(saved['meta']['doris_checkpoint']['completed_iter'], start + 2)
                expected_lrs = []
                for _ in range(3):
                    original.train(loader)
                    expected_lrs.append(original.optimizer.param_groups[0]['lr'])
                resumed = make_runner(root)
                runtime = Config.fromstring(cfg.pretty_text, '.py')
                runtime.resume_from = str(latest)
                runtime.load_from = None
                exec(compile(ast.Module(body=[block], type_ignores=[]), 'production_resume', 'exec'),
                    dict(cfg=runtime, runner=resumed, doris_checkpoints=True,
                         check_doris_resume_checkpoint=check_doris_resume_checkpoint,
                         meta={}, osp=__import__('os.path', fromlist=['']), re=__import__('re'),
                         logger=logging.getLogger('resume-test')))
                self.assertEqual(resumed.iter, start + 2)
                self.assertEqual(resumed.model.local_iter, start + 2)
                self.assertEqual(resumed.meta['fp16'], saved['meta']['fp16'])
                resumed.call_hook('before_run')
                resumed.call_hook('before_epoch')
                actual_lrs = []
                for _ in range(3):
                    resumed.train(loader)
                    actual_lrs.append(resumed.optimizer.param_groups[0]['lr'])
                self.assertEqual(actual_lrs, expected_lrs)
                for key, value in original.model.state_dict().items():
                    torch.testing.assert_close(resumed.model.state_dict()[key], value, rtol=0, atol=0)
                for key, value in original.optimizer.state_dict()['state'].items():
                    for field, tensor in value.items():
                        torch.testing.assert_close(resumed.optimizer.state_dict()['state'][key][field], tensor, rtol=0, atol=0)
                self.assertEqual(resumed.iter, original.iter)
                self.assertEqual(resumed.model.local_iter, original.model.local_iter)
                self.assertEqual(resumed.model.struct_gate.gate_strength(resumed.iter),
                                 original.model.struct_gate.gate_strength(original.iter))
                scope = cls.__init__.__globals__
                self.assertEqual(scope['phy_gate_strength'](resumed.iter, **resumed.model.phy_gate_params),
                                 scope['phy_gate_strength'](original.iter, **original.model.phy_gate_params))
                self.assertEqual(resumed.iter >= resumed.model.proto_gate.weight_start,
                                 original.iter >= original.model.proto_gate.weight_start)
                self.assertTrue(resumed.model.proto_gate.source_valid_proto.any())
                self.assertTrue(resumed.model.struct_gate.source_adj_initialized.any())
                self.assertTrue(resumed.model.struct_edge_pair_prior_ready)

    def test_best_rejects_resume_and_remains_a_load_from_weight_source(self):
        cfg = Config.fromfile(str(ROOT / 'configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi.py'))
        model = torch.nn.Linear(2, 2)
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            path = Path(temporary) / 'best_mIoU_iter_2000.pth'
            torch.save(dict(state_dict=model.state_dict(), meta=dict(config=cfg.pretty_text, iter=1999,
                doris_checkpoint=dict(completed_iter=2000, resumable=False))), path)
            cfg.resume_from = str(path)
            with self.assertRaisesRegex(ValueError, '--load-from'):
                check_doris_resume_checkpoint(cfg)
            fresh = torch.nn.Linear(2, 2)
            from mmcv.runner import load_checkpoint
            load_checkpoint(fresh, str(path), map_location='cpu', strict=True)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(fresh.state_dict()[key], value)


if __name__ == '__main__':
    unittest.main()

"""Focused CPU/static tests. Temporary CSV values are evaluator test fixtures."""
import ast
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_tau_proto_sensitivity as sweep
from tools.doris_checkpoint_plan import configure_checkpoint_policy


def static_configs():
    # Reuse the existing plain-Python config reader without importing that
    # test module's optional dependencies or creating its outputs/ directory.
    source = ROOT / 'tests/test_minimal_output.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'static_config')
    scope = dict(ROOT=ROOT, copy=copy)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), scope)
    return scope['static_config']


def hook_harness():
    """Exercise the actual hook around a minimal evaluator superclass."""
    class Evaluator:
        def __init__(self, *args, **kwargs):
            self.save_best = kwargs.get('save_best')

        def evaluate(self, runner, results):
            runner.calls += 1
            runner.log_buffer.output.update(results)
            return results.get('mIoU')

    path = ROOT / 'mmseg/core/evaluation/sensitivity_eval_hook.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    scope = dict(EvalHook=Evaluator, append_miou=sweep.append_miou)
    exec(compile(tree, str(path), 'exec'), scope)
    return scope['SensitivityEvalHook']


class SensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load = static_configs()
        cls.base = load(sweep.FULL_CONFIG)
        cls.child = load(sweep.SWEEP_CONFIG)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='pssal_tau_test_', dir=ROOT)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.csv = self.root / 'sensitivity_miou.csv'
        git = patch.object(sweep.subprocess, 'check_output', return_value='unit-test-sha\n')
        git.start()
        self.addCleanup(git.stop)

    def config(self, tau=0.25):
        cfg = copy.deepcopy(self.child)
        cfg['work_dir'] = str(self.root / ('tau_' + sweep.tau_key(tau).replace('.', 'p')))
        cfg['uda']['proto_cfg']['tau_proto'] = tau
        cfg['sensitivity_analysis'].update(tau_proto=tau, csv_path=str(self.csv))
        return cfg

    def test_static_inheritance_and_exact_difference_whitelist(self):
        for tau in sweep.TAU_VALUES:
            sweep.check_effective_config(self.base, self.config(tau))
        for path, value in [('optimizer.lr', 1.0), ('uda.proto_cfg.loss_start', 4000),
                            ('uda.proto_cfg.tau_conf_proto', .9), ('runner.max_iters', 20000),
                            ('data.val.img_dir', 'rgb_dir/test'), ('model.backbone.in_chans', 3),
                            ('uda.alpha', .9), ('uda.struct_cfg.enable', False)]:
            with self.subTest(path=path):
                cfg = self.config()
                target = cfg
                parts = path.split('.')
                for part in parts[:-1]:
                    target = target[part]
                target[parts[-1]] = value
                with self.assertRaises(ValueError):
                    sweep.check_effective_config(self.base, cfg)
        cfg = self.config()
        cfg['sensitivity_analysis']['tau_proto'] = .35
        with self.assertRaises(ValueError):
            sweep.check_effective_config(self.base, cfg)

    @unittest.skipUnless(importlib.util.find_spec('mmcv'), 'MMCV unavailable')
    def test_real_mmcv_config_preflight(self):
        with redirect_stdout(io.StringIO()):
            self.assertIsNotNone(sweep.preflight(self.root))
        import logging
        from torch.utils.data import DataLoader, Dataset
        from mmseg.core.evaluation.sensitivity_eval_hook import SensitivityEvalHook

        class ValidationDataset(Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, index):
                return index

            def evaluate(self, results, **kwargs):
                return results

        hook = SensitivityEvalHook(DataLoader(ValidationDataset()), csv_path=str(self.csv),
                                   tau_proto=.05, interval=1000, save_best=None)
        runner = SimpleNamespace(iter=999, logger=logging.getLogger('sensitivity-test'),
                                 log_buffer=SimpleNamespace(output={}, ready=False))
        hook.evaluate(runner, {'mIoU': .5739})
        self.assertEqual(sweep.read_rows(self.csv)[0]['mIoU'], '57.3900')

    def test_checkpoint_opt_out_is_idempotent_and_default_policy_unchanged(self):
        cfg = self.config()
        before = copy.deepcopy(cfg)
        for _ in range(2):
            self.assertFalse(configure_checkpoint_policy(cfg))
        self.assertEqual(cfg, before)
        full = copy.deepcopy(self.base)
        self.assertTrue(configure_checkpoint_policy(full))
        self.assertEqual(full['checkpoint_config']['type'], 'DorisCheckpointHook')
        self.assertTrue(full['evaluation']['doris_best_only'])
        self.assertEqual(full['checkpoint_config']['keep_iterations'], [2000, 4000, 6000])

    def test_evaluation_event_writes_percent_once_without_checkpoint(self):
        hook = hook_harness()(csv_path=str(self.csv), tau_proto=.05,
                              interval=1000, save_best=None)
        runner = SimpleNamespace(iter=999, calls=0, log_buffer=SimpleNamespace(output={}))
        self.assertEqual(hook.evaluate(runner, {'mIoU': .5739}), .5739)
        self.assertEqual(runner.calls, 1)
        self.assertEqual(self.csv.read_text().splitlines(),
                         ['tau_proto,iter,mIoU', '0.05,1000,57.3900'])
        with self.assertRaises(ValueError):
            hook.evaluate(runner, {'mIoU': .5739})
        runner.iter = 1999
        with self.assertRaises(ValueError):
            hook.evaluate(runner, {})
        self.assertEqual(len(sweep.read_rows(self.csv)), 1)
        self.assertFalse(list(self.root.glob('*.pth')))

    def test_invalid_metrics_and_nodes_never_write(self):
        for score in (None, True, '0.5', float('nan'), float('inf'), -.1, 57.39):
            with self.subTest(score=score), self.assertRaises(ValueError):
                sweep.append_miou(self.csv, .05, 1000, score)
        self.assertFalse(self.csv.exists())
        with self.assertRaises(ValueError):
            sweep.append_miou(self.csv, .05, 2000, .5)
        for tau in sweep.TAU_VALUES:
            for step in sweep.STEPS:
                sweep.append_miou(self.csv, tau, step, .5)
        self.assertEqual(len(sweep.read_rows(self.csv)), 75)
        with self.assertRaises(ValueError):
            sweep.append_miou(self.csv, .45, 15000, .5)

    def test_dry_run_prints_five_commands_without_creating_outputs(self):
        output = io.StringIO()
        with patch.object(sweep.subprocess, 'run') as run, redirect_stdout(output):
            sweep.main(['--output-root', str(self.root / 'dry'), '--dry-run'])
        run.assert_not_called()
        commands = [line for line in output.getvalue().splitlines() if 'tools/train.py ' in line]
        self.assertEqual(len(commands), 5)
        self.assertFalse((self.root / 'dry').exists())
        for tau, command in zip(sweep.TAU_VALUES, commands):
            self.assertIn('uda.proto_cfg.tau_proto=' + sweep.tau_key(tau), command)
            self.assertNotIn('--resume-from', command)
            self.assertNotIn('--load-from', command)

    def test_failed_group_requires_explicit_fresh_retry_and_preserves_logs(self):
        calls = []

        def execute(command, **kwargs):
            self.assertTrue(kwargs['check'])
            self.assertNotIn('shell', kwargs)
            tau = float(next(arg.split('=')[1] for arg in command if arg.startswith('uda.proto_cfg.tau_proto=')))
            directory = Path(command[command.index('--work-dir') + 1])
            directory.mkdir(parents=True)
            (directory / 'test.log').write_text('temporary test fixture')
            calls.append((tau, directory))
            for step in sweep.STEPS:
                sweep.append_miou(self.csv, tau, step, .5 + tau / 10)
                if tau == .15 and len(calls) == 2 and step == 2000:
                    raise subprocess.CalledProcessError(9, command)

        with patch.object(sweep.subprocess, 'run', side_effect=execute), redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                sweep.run_sweep(self.root, self.base)
            state = json.loads((self.root / 'sweep_state.json').read_text())
            self.assertEqual(state['groups']['0.05']['status'], 'completed')
            self.assertEqual(state['groups']['0.15']['completed_iter'], 2000)
            self.assertEqual(state['groups']['0.15']['exit_code'], 9)
            before = self.csv.read_bytes()
            with self.assertRaises(FileExistsError):
                sweep.run_sweep(self.root, self.base)
            with self.assertRaises(RuntimeError):
                sweep.run_sweep(self.root, self.base, resume=True)
            self.assertEqual(self.csv.read_bytes(), before)
            sweep.run_sweep(self.root, self.base, resume=True, retry_failed=True)
            self.assertEqual(len(sweep.read_rows(self.csv)), 75)
            self.assertEqual([tau for tau, _ in calls], [.05, .15, .15, .25, .35, .45])
            self.assertEqual(calls[2][1].name, 'retry_002')
            self.assertTrue((calls[1][1] / 'test.log').is_file())
            sweep.run_sweep(self.root, self.base, resume=True)
            self.assertEqual(len(calls), 6)

    def test_completed_state_csv_mismatch_is_rejected_before_reset(self):
        with sweep.sweep_lock(self.root):
            state = sweep.initialize_state(self.root, self.base, False)
        state['groups']['0.05'].update(status='completed', completed_iter=15000, exit_code=0)
        sweep.atomic_json(self.root / 'sweep_state.json', state)
        with self.assertRaises(ValueError), patch.object(sweep.subprocess, 'run') as run:
            sweep.run_sweep(self.root, self.base, resume=True, retry_failed=True)
        run.assert_not_called()

    def test_none_checkpoint_metadata_guard_and_diagnostic_filter(self):
        tree = ast.parse((ROOT / 'tools/train.py').read_text(encoding='utf-8'))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        guards = [node for node in main.body if isinstance(node, ast.If)
                  and ast.dump(node.test) == ast.dump(ast.parse(
                      'cfg.checkpoint_config is not None', mode='eval').body)]
        self.assertEqual(len(guards), 2)
        for guard in guards:
            scope = dict(cfg=SimpleNamespace(checkpoint_config=None), meta={})
            exec(compile(ast.Module(body=[guard], type_ignores=[]), 'meta_guard', 'exec'), scope)
        scope = dict(cfg=SimpleNamespace(checkpoint_config=SimpleNamespace(meta={'ordinary': 1})), meta={})
        exec(compile(ast.Module(body=[guards[1]], type_ignores=[]), 'ordinary_meta', 'exec'), scope)
        self.assertEqual(scope['meta'], {'ordinary': 1})
        tree = ast.parse((ROOT / 'mmseg/utils/precision_logger.py').read_text(encoding='utf-8'))
        logger = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'PrecisionLoggerHook')
        method = next(node for node in logger.body if isinstance(node, ast.FunctionDef) and node.name == '_keep_train_key')
        scope = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'logger_filter', 'exec'), scope)
        keep = scope['_keep_train_key']
        self.assertTrue(keep(SimpleNamespace(), 'phy_gate_iter', None))
        self.assertFalse(keep(SimpleNamespace(log_diagnostics=False), 'phy_gate_iter', None))
        self.assertTrue(keep(SimpleNamespace(log_diagnostics=False), 'loss_phy_src', None))

    def test_active_sweep_lock_prevents_a_second_launcher(self):
        with sweep.sweep_lock(self.root):
            with self.assertRaises(OSError):
                with sweep.sweep_lock(self.root):
                    self.fail('A second launcher acquired an active lock')


if __name__ == '__main__':
    unittest.main()

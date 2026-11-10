"""CPU/mocked output checks; no project environment installation or GPU needed.

Torch-only component tests run when torch is available. Config.fromfile checks
run when MMCV is installed. Neither replaces remote full-model validation.
"""
import ast
import base64
from collections import OrderedDict
import copy
import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
(ROOT / 'outputs').mkdir(exist_ok=True)
HAS_BASELINE = (ROOT / 'outputs/minimal_output_before.json').is_file()
sys.path.insert(0, str(ROOT))
try:
    import torch
except ImportError:
    torch = None


def original_source(name):
    for filename in ('minimal_output_before.json',
                     'minimal_output_additional_before.json'):
        baseline = ROOT / 'outputs' / filename
        if baseline.is_file():
            files = json.loads(baseline.read_text(encoding='utf-8'))['files']
            if name in files:
                return base64.b64decode(files[name]).decode('utf-8-sig')
    return (ROOT / name).read_text(encoding='utf-8-sig')


def load_methods(name, class_name, methods, namespace=None, original=False):
    tree = ast.parse(original_source(name) if original else
                     (ROOT / name).read_text(encoding='utf-8-sig'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == class_name)
    nodes = [node for node in cls.body if isinstance(node, ast.FunctionDef)
             and node.name in methods]
    scope = dict(namespace or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), name, 'exec'), scope)
    return type(class_name + 'Harness', (),
                {node.name: scope[node.name] for node in nodes})


def component(name, cls, original=False):
    scope = {}
    source = original_source(name) if original else (ROOT / name).read_text(
        encoding='utf-8-sig')
    exec(compile(source, name, 'exec'), scope)
    return scope[cls]


def dacs_component(original=False):
    """Real DACS methods and constructor with a small backbone stand-in."""
    class Base(torch.nn.Module):
        def __init__(self, **cfg):
            super().__init__()
            self.model = torch.nn.Conv2d(6, 5, 1)
            self.num_classes = 5
        def get_model(self):
            return self.model
    def build_segmentor(cfg):
        return torch.nn.Conv2d(6, 5, 1)
    scope = dict(torch=torch, F=torch.nn.functional, os=os, deepcopy=copy.deepcopy,
                 Base=Base, build_segmentor=build_segmentor, get_module=lambda x: x,
                 mmcv=SimpleNamespace(print_log=Mock()),
                 PhyReliabilityHead=lambda **cfg: torch.nn.Sequential(
                     torch.nn.Conv2d(3, 1, 1), torch.nn.Sigmoid()),
                 ProtoGate=component('mmseg/models/uda/proto_gate.py', 'ProtoGate', original),
                 StructPriorConsistencyGate=component(
                     'mmseg/models/uda/struct_prior_gate.py',
                     'StructPriorConsistencyGate', original))
    edge_tree = ast.parse((ROOT / 'mmseg/models/losses/struct_edge_loss.py').read_text())
    funcs = [n for n in edge_tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('_as_label', 'label_to_edge')]
    exec(compile(ast.Module(body=funcs, type_ignores=[]), 'edge', 'exec'), scope)
    path = 'mmseg/models/uda/dacs.py'
    tree = ast.parse(original_source(path) if original else (ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DACS')
    cls.decorator_list, cls.bases = [], [ast.Name(id='Base', ctx=ast.Load())]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 path, 'exec'), scope)
    return scope['DACS']


def official_configs():
    for direction in ('goji26RGB2goji30RGB', 'goji26IRRG2goji30RGB',
                      'goji30IRRG2goji26IRRG'):
        for suffix in ('', '_phy_gate', '_struct_gate', '_phy_struct_gate',
                       '_ps_proto', '_struct_edge', '_pbcr', '_proto_struct_edge'):
            yield 'configs/hrda/' + direction + '_hrda_3vi' + suffix + '.py'


def supervised_configs():
    for name in ('goji26IRRG', 'goji30RGB'):
        for suffix in ('', '_phy_gate', '_struct_gate', '_phy_struct_gate',
                       '_ps_proto', '_struct_edge', '_pbcr', '_proto_struct_edge'):
            yield 'configs/hrda/pssal/' + name + '_hrda_3vi' + suffix + '.py'


def static_config(path, original=False):
    """Execute this repo's plain Python configs; not a substitute for MMCV."""
    def merge(base, child):
        out = copy.deepcopy(base)
        for key, value in child.items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                value = copy.deepcopy(value)
                delete = value.pop('_delete_', False)
                out[key] = value if delete else merge(out[key], value)
            else:
                out[key] = copy.deepcopy(value)
        return out

    scope = {}
    source = original_source(path) if original else (ROOT / path).read_text(
        encoding='utf-8-sig')
    exec(compile(source, path, 'exec'), scope)
    base = {}
    for entry in scope.pop('_base_', []):
        relative = (ROOT / path).parent.joinpath(entry).resolve().relative_to(ROOT)
        base = merge(base, static_config(relative.as_posix(), original))
    return merge(base, {k: v for k, v in scope.items() if not k.startswith('__')})


def protected_config(value):
    output_keys = {'archive_source_code', 'output_mode', 'paper_process_logging',
                   'save_class_mix_debug', 'save_teacher_pseudo_debug',
                   'save_phy_debug', 'save_struct_debug', 'save_proto_debug',
                   'save_joint_debug', 'log_class_stats', 'console_keys'}
    if isinstance(value, dict):
        return {k: protected_config(v) for k, v in value.items()
                if k not in output_keys}
    if isinstance(value, list):
        return [protected_config(v) for v in value]
    return value


class OutputControlTests(unittest.TestCase):
    def test_disabled_debug_helpers_never_touch_files_or_student(self):
        names = ['_maybe_save_teacher_pseudo_debug', '_maybe_save_phy_debug',
                 '_maybe_save_struct_debug', '_maybe_save_proto_debug',
                 '_maybe_save_joint_debug', 'update_debug_state']
        harness = load_methods('mmseg/models/uda/dacs.py', 'DACS', names)
        hook = harness()
        hook.local_iter = 0
        hook.debug_img_interval = 1000
        hook.save_class_mix_debug = False
        student = SimpleNamespace(decode_head=SimpleNamespace(debug=True),
                                  generate_pseudo_label=Mock())
        teacher = SimpleNamespace(decode_head=SimpleNamespace(debug=True))
        hook.get_model = lambda: student
        hook.get_ema_model = lambda: teacher
        hook.mic = SimpleNamespace(debug=True)
        for flag in ('enable_phy', 'enable_struct', 'enable_proto',
                     'enable_joint_gate'):
            setattr(hook, flag, True)
        for flag in ('save_teacher_pseudo_debug', 'save_phy_debug',
                     'save_struct_debug', 'save_proto_debug', 'save_joint_debug'):
            setattr(hook, flag, False)
        hook.update_debug_state()
        self.assertFalse(student.decode_head.debug)
        self.assertFalse(teacher.decode_head.debug)
        self.assertFalse(hook.mic.debug)
        counts = [2, 11, 8, 2, 11]
        for method, count in zip(names, counts):
            getattr(hook, method)(*([None] * count))
        student.generate_pseudo_label.assert_not_called()
        hook.save_class_mix_debug = True
        hook.update_debug_state()
        self.assertTrue(student.decode_head.debug)
        hook.debug_img_interval = 0
        hook.update_debug_state()
        self.assertFalse(student.decode_head.debug)

    def test_classmix_plot_and_collections_have_explicit_guard(self):
        tree = ast.parse((ROOT / 'mmseg/models/uda/dacs.py').read_text())
        parents = {child: node for node in ast.walk(tree)
                   for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Subscript)
                       and isinstance(t.value, ast.Name)
                       and t.value.id == 'seg_debug' for t in node.targets):
                continue
            ancestor = parents[node]
            guards = []
            while ancestor in parents:
                if isinstance(ancestor, ast.If):
                    guards.append(ast.dump(ancestor.test))
                ancestor = parents[ancestor]
            self.assertTrue(any('save_class_mix_debug' in g for g in guards))

    def test_logger_filters_json_and_preserves_validation_precision(self):
        namespace = dict(OrderedDict=OrderedDict, datetime=datetime,
                         torch=SimpleNamespace(cuda=SimpleNamespace(
                             is_available=lambda: True)),
                         mmcv=SimpleNamespace(dump=lambda data, stream, **kw:
                                              json.dump(data, stream)))
        cls = load_methods('mmseg/utils/precision_logger.py',
                           'PrecisionLoggerHook',
                           ['_keep_train_key', '_dump_log', '_round_float',
                            'log', '_log_info'], namespace)
        hook = cls()
        hook.output_mode, hook.paper_process_logging = 'minimal', False
        hook.console_keys = hook.json_keys = None
        hook.log_memory = False
        hook.by_epoch, hook.interval = False, 50
        hook.interval_exp_name, hook.time_sec_tot, hook.start_iter = 1000, 0, 0
        hook.every_n_iters = lambda r, n: (r.iter + 1) % n == 0
        hook.get_mode = lambda r: 'train' if 'time' in r.log_buffer.output else 'val'
        hook.get_epoch = lambda r: 1
        hook.get_iter = lambda r, **kw: r.iter + 1
        messages = []
        runner = SimpleNamespace(
            rank=0, mode='train', epoch=0, iter=999, max_iters=20000,
            meta=None, logger=SimpleNamespace(info=messages.append),
            current_lr=lambda: [6e-5], log_buffer=SimpleNamespace(output={}))
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            hook.json_log_path = str(Path(temporary) / 'log.json')
            hook._dump_log({'env_info': 'kept', 'git_commit': 'kept'}, runner)
            runner.log_buffer.output = dict(
                time=1.1, data_time=0.01, loss=1.25,
                **{'decode.acc_seg': 86.251234,
                   'decode.hr.acc_seg': 84.125,
                   'mix.decode.acc_seg': 91.753456,
                   'mix.decode.hr.acc_seg': 89.625,
                   'decode.loss_seg': 0.55,
                   'mix.decode.loss_struct_edge': 0.0123456789,
                   'loss_phy_src': 0.6,
                   'proto_weight_class_0_mean': 0.42})
            hook.log(runner)
            train_message = messages[-1]
            self.assertIn('decode.acc_seg: 86.25%', train_message)
            self.assertIn('mix.decode.acc_seg: 91.75%', train_message)
            self.assertIn('decode.hr.acc_seg: 84.12%', train_message)
            self.assertIn('mix.decode.hr.acc_seg: 89.62%', train_message)
            self.assertIn('loss: 1.2500', train_message.splitlines()[0])
            self.assertIn('\n    source: decode.loss_seg: 0.5500', train_message)
            self.assertIn('\n    mix: mix.decode.loss_struct_edge:', train_message)
            self.assertIn('\n    aux: loss_phy_src: 0.6000', train_message)
            self.assertNotIn('proto_weight_class_0_mean', train_message)
            runner.log_buffer.output = dict(
                eval_iter_num=286, time=38.0, mIoU=0.361812345,
                mF1=0.491234567, **{'Veg.mIoU': 0.5})
            result = hook.log(runner)
            self.assertEqual(result['mode'], 'val')
            self.assertNotIn('time', result)
            rows = [json.loads(line) for line in Path(
                hook.json_log_path).read_text().splitlines()]
            self.assertIn('env_info', rows[0])
            self.assertNotIn('proto_weight_class_0_mean', rows[1])
            self.assertEqual(rows[1]['mix.decode.loss_struct_edge'], 0.0123456789)
            self.assertEqual(rows[1]['decode.acc_seg'], 86.251234)
            self.assertEqual(rows[1]['mix.decode.acc_seg'], 91.753456)
            self.assertEqual(rows[1]['decode.hr.acc_seg'], 84.125)
            self.assertEqual(rows[1]['mix.decode.hr.acc_seg'], 89.625)
            self.assertEqual(rows[2]['mIoU'], 0.361812345)
            self.assertEqual(rows[2]['Veg.mIoU'], 0.5)
            self.assertIn('Iter(val) [1000]', messages[-1])
        self.assertFalse(hook._keep_train_key('mparc_pseudo_weight_mean', None))
        hook.paper_process_logging = True
        self.assertTrue(hook._keep_train_key('mparc_pseudo_weight_mean', None))
        hook.output_mode = 'diagnostic'
        self.assertTrue(hook._keep_train_key('proto_weight_class_0_mean', None))

    def test_training_archive_is_opt_in_and_original_config_is_preserved(self):
        tree = ast.parse((ROOT / 'tools/train.py').read_text())
        parents = {child: node for node in ast.walk(tree)
                   for child in ast.iter_child_nodes(node)}
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name)
                 and n.func.id == 'gen_code_archive']
        self.assertEqual(len(calls), 1)
        guard = parents[parents[calls[0]]]
        self.assertIsInstance(guard, ast.If)
        self.assertEqual(ast.dump(guard.test), ast.dump(ast.parse(
            "cfg.get('archive_source_code', False)", mode='eval').body))
        self.assertIn('shutil.copyfile', (ROOT / 'tools/train.py').read_text())

    def test_structure_only_does_not_require_joint_debug_attributes(self):
        cls = load_methods('mmseg/models/uda/dacs.py', 'DACS',
                           ['_apply_struct_gate'])
        hook = cls()
        hook.save_struct_debug = hook.enable_joint_gate = False
        hook.log_debug_stats, hook.local_iter = False, 0
        hook.struct_gate = SimpleNamespace(apply=Mock(return_value=('w', {})))
        hook._log_struct_debug = Mock()
        hook._apply_struct_gate(None, None, None, None, {})
        self.assertFalse(hook.struct_gate.apply.call_args.kwargs['collect_debug'])

    def test_entry_point_arguments_remain_compatible(self):
        import argparse
        scope = dict(argparse=argparse, os=os, DictAction=argparse.Action)
        for path in ('tools/train.py', 'tools/test.py'):
            tree = ast.parse((ROOT / path).read_text())
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'parse_args')
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), scope)
            if path.endswith('train.py'):
                args = scope['parse_args'](['config.py', '--work-dir', 'fresh dir',
                                             '--resume-from', 'latest.pth'])
                self.assertEqual(args.work_dir, 'fresh dir')
                self.assertEqual(args.resume_from, 'latest.pth')
            else:
                with patch.object(sys, 'argv', ['test.py', 'config.py', 'latest.pth',
                     '--eval', 'mIoU', '--show-dir', 'fresh dir', '--hrda-out', 'LR',
                     '--save-confidence-map']):
                    args = scope['parse_args']()
                self.assertTrue(args.save_confidence_map)
                self.assertEqual(args.hrda_out, 'LR')
                self.assertEqual(args.eval, ['mIoU'])

    @unittest.skipUnless(HAS_BASELINE, 'Local pre-change snapshot unavailable')
    def test_supervised_configs_keep_algorithm_parameters(self):
        helper = 'configs.hrda.pssal.pssal_supervised_ablation_utils'
        configs = {}
        for original in (True, False):
            module = ModuleType(helper)
            path = helper.replace('.', '/') + '.py'
            exec(original_source(path) if original else
                 (ROOT / path).read_text(), module.__dict__)
            with patch.dict(sys.modules, {helper: module}):
                configs[original] = {p: static_config(p, original)
                                     for p in supervised_configs()}
        for path, cfg in configs[False].items():
            self.assertEqual(protected_config(configs[True][path]),
                             protected_config(cfg), path)
            self.assertEqual(cfg['uda']['output_mode'], 'minimal')
            self.assertFalse(cfg['uda']['proto_cfg'].get('log_class_stats', False))

    @unittest.skipUnless(HAS_BASELINE, 'Local pre-change snapshot unavailable')
    def test_static_config_values_protect_algorithms_for_all_24_variants(self):
        helper = 'configs.hrda.pssalda_26rgb2_30rgb_ablation_utils'
        configs = {}
        for original in (True, False):
            module = ModuleType(helper)
            path = helper.replace('.', '/') + '.py'
            exec(original_source(path) if original else
                 (ROOT / path).read_text(), module.__dict__)
            with patch.dict(sys.modules, {helper: module}):
                configs[original] = {p: static_config(p, original)
                                     for p in official_configs()}
        for path, cfg in configs[False].items():
            self.assertEqual(protected_config(configs[True][path]),
                             protected_config(cfg), path)
            self.assertFalse(cfg['archive_source_code'])
            self.assertEqual(cfg['uda']['output_mode'], 'minimal')
            self.assertFalse(cfg['uda']['save_teacher_pseudo_debug'])
            self.assertFalse(cfg['uda']['save_class_mix_debug'])
            for module, flag in [('phy_cfg', 'save_phy_debug'),
                                 ('struct_cfg', 'save_struct_debug'),
                                 ('proto_cfg', 'save_proto_debug'),
                                 ('joint_gate_cfg', 'save_joint_debug')]:
                self.assertFalse(cfg['uda'][module][flag])

    @unittest.skipUnless(importlib.util.find_spec('mmcv'), 'MMCV unavailable locally')
    def test_effective_configs_with_real_mmcv(self):
        from mmcv import Config
        for direction in ('goji26RGB2goji30RGB', 'goji26IRRG2goji30RGB',
                          'goji30IRRG2goji26IRRG'):
            for suffix in ('', '_phy_gate', '_struct_gate', '_phy_struct_gate',
                           '_ps_proto', '_struct_edge', '_pbcr', '_proto_struct_edge'):
                cfg = Config.fromfile(str(ROOT / 'configs/hrda' /
                    (direction + '_hrda_3vi' + suffix + '.py')))
                self.assertEqual(cfg.uda.output_mode, 'minimal')
                self.assertFalse(cfg.archive_source_code)
                self.assertFalse(cfg.uda.save_class_mix_debug)
                self.assertFalse(cfg.uda.save_teacher_pseudo_debug)
                for module, flag in [('phy_cfg', 'save_phy_debug'),
                                     ('struct_cfg', 'save_struct_debug'),
                                     ('proto_cfg', 'save_proto_debug')]:
                    self.assertFalse(cfg.uda[module][flag])
                self.assertEqual(cfg.runner.max_iters, 20000)
                self.assertEqual(cfg.data.samples_per_gpu, 2)
        for path in supervised_configs():
            cfg = Config.fromfile(str(ROOT / path))
            self.assertEqual(cfg.uda.output_mode, 'minimal')
            self.assertFalse(cfg.get('archive_source_code', False))


@unittest.skipIf(torch is None, 'Torch CPU environment unavailable')
class NumericalSafetyTests(unittest.TestCase):
    def test_actual_dacs_init_keeps_disabled_modules_and_output_flags_off(self):
        cls = dacs_component()
        for path in official_configs():
            cfg = static_config(path)
            hook = cls(**cfg['uda'], model=cfg['model'], max_iters=cfg['runner']['max_iters'])
            self.assertFalse(hook.save_class_mix_debug)
            self.assertFalse(hook.save_teacher_pseudo_debug)
            for module, enabled, flag in [('phy', 'enable_phy', 'save_phy_debug'),
                                          ('struct', 'enable_struct', 'save_struct_debug'),
                                          ('proto', 'enable_proto', 'save_proto_debug')]:
                if cfg['uda'][module + '_cfg']['enable']:
                    self.assertTrue(getattr(hook, enabled))
                    self.assertFalse(getattr(hook, flag))
                else:
                    self.assertFalse(getattr(hook, enabled))

    @unittest.skipUnless(HAS_BASELINE, 'Local pre-change snapshot unavailable')
    def test_phy_joint_boundary_and_ema_match_before_change(self):
        old_cls, new_cls = dacs_component(True), dacs_component(False)
        cfg = static_config('configs/hrda/goji26IRRG2goji30RGB_hrda_3vi_proto_struct_edge.py')
        cfg['uda']['joint_gate_cfg']['enable'] = True
        kwargs = dict(cfg['uda'], model=cfg['model'], max_iters=20000)
        torch.manual_seed(12)
        old = old_cls(**kwargs)
        torch.manual_seed(12)
        new = new_cls(**kwargs)
        labels = (torch.arange(50).reshape(2, 5, 5) % 5).long()
        img, vi = torch.randn(2, 6, 5, 5), torch.randn(2, 3, 5, 5)
        weight = torch.rand(2, 5, 5)
        probs = torch.randn(2, 5, 5, 5).softmax(1)
        for step in (0, 7000, 12000):
            values = []
            for hook in (old, new):
                hook.local_iter = step
                hook._init_ema_weights()
                with torch.no_grad():
                    hook.get_model().weight.add_(0.1)
                hook._update_ema(step + 1)
                response, phy_weight, _ = hook._apply_phy_gate(vi, labels, weight, {})
                joint, _ = hook._fuse_phy_struct_pseudo_weight(weight, phy_weight,
                                                             weight * 0.7, labels, {})
                src = hook._build_struct_edge_weight(img, labels[:, None], 0.004,
                                                     {}, 'src', is_source=True)
                tgt = hook._build_struct_edge_weight(img, labels, 0.002, {}, 'tgt',
                                                     target_probs=probs, is_target=True)
                values.append((response, phy_weight, joint, src, tgt))
            for a, b in zip(*values):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for key, value in old.state_dict().items():
                torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)

    def test_test_api_default_masks_and_explicit_confidence(self):
        tree = ast.parse((ROOT / 'mmseg/apis/test.py').read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in ('single_gpu_test', 'save_color_seg')]
        from PIL import Image
        write_color, write_conf = Mock(), Mock()
        mmcv = SimpleNamespace(ProgressBar=lambda n: SimpleNamespace(update=Mock()),
                               mkdir_or_exist=Mock(), imwrite=Mock())
        namespace = dict(torch=torch, mmcv=mmcv, np=np, Image=Image,
                         osp=os.path, save_confidence_map=write_conf,
                         tensor2imgs=Mock(side_effect=AssertionError('unneeded conversion')))
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'test_api', 'exec'),
             namespace)
        palette_func = namespace['save_color_seg']
        namespace['save_color_seg'] = write_color
        seg = np.arange(5).reshape(1, 5)
        metadata = {}
        exec((ROOT / 'mmseg/datasets/goji_meta.py').read_text(), metadata)
        palette = metadata['GOJI_PALETTE']
        model = Mock(return_value=[seg])
        model.module = SimpleNamespace(
            decode_head=SimpleNamespace(debug_output_attention=False),
            parameters=lambda: iter([torch.zeros(1)]),
            inference=Mock(return_value=torch.ones(1, 5, 1, 5) / 5))
        meta = dict(ori_filename='tile.png')
        data = dict(img=[torch.zeros(1, 6, 1, 5)],
                    img_metas=[SimpleNamespace(data=[[meta]])])
        class Dataset(list):
            PALETTE = palette
        class Loader:
            dataset = Dataset([0])
            def __iter__(self):
                return iter([data])
        test = namespace['single_gpu_test']
        results = test(model, Loader())
        np.testing.assert_array_equal(results[0], seg)
        write_color.assert_not_called()
        mmcv.mkdir_or_exist.assert_not_called()
        model.module.inference.assert_not_called()
        test(model, Loader(), out_dir='explicit')
        self.assertEqual(write_color.call_count, 1)
        write_conf.assert_not_called()
        model.module.inference.assert_not_called()
        test(model, Loader(), out_dir='explicit', save_confidence=True)
        model.module.inference.assert_called_once()
        write_conf.assert_called_once()
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as directory:
            path = str(Path(directory) / 'palette.png')
            palette_func(seg, palette, path)
            with Image.open(path) as img:
                np.testing.assert_array_equal(np.asarray(img), np.array([palette], dtype=np.uint8))

    def test_explicit_debug_exports_and_old_student_side_effects(self):
        names = ['_maybe_save_teacher_pseudo_debug', '_maybe_save_phy_debug',
                 '_maybe_save_struct_debug', '_maybe_save_proto_debug',
                 '_maybe_save_joint_debug']
        image_writer = Mock()
        mkdir = Mock()
        scope = dict(torch=torch, np=np,
                     os=SimpleNamespace(path=os.path, makedirs=mkdir),
                     Image=SimpleNamespace(fromarray=lambda *a, **kw: image_writer),
                     get_vis_image=lambda img, *a: img[:, :3])
        cls = load_methods('mmseg/models/uda/dacs.py', 'DACS', names, scope)
        hook = cls()
        hook.local_iter, hook.train_cfg = 0, {'work_dir': 'unused'}
        for module in ('phy', 'struct', 'proto', 'joint_gate'):
            setattr(hook, 'enable_' + module, True)
        for flag in ('teacher_pseudo', 'phy', 'struct', 'proto', 'joint'):
            setattr(hook, 'save_' + flag + '_debug', True)
        for prefix in ('teacher_pseudo', 'phy', 'struct', 'proto', 'joint'):
            setattr(hook, prefix + '_debug_interval', 1000)
            setattr(hook, prefix + '_debug_max_samples', 1)
            setattr(hook, prefix + '_debug_dir', prefix)
        hook.phy_cfg = hook.proto_cfg = hook.joint_gate_cfg = {}
        hook._get_teacher_debug_dir = lambda: 'teacher'
        for name in ('_save_teacher_raw_pseudo_label', '_save_teacher_confidence_map',
                     '_save_phy_map', '_save_struct_matrix'):
            setattr(hook, name, Mock())
        student = torch.nn.Sequential(torch.nn.BatchNorm2d(6), torch.nn.Dropout(0.5))
        hook.get_model = lambda: SimpleNamespace(
            generate_pseudo_label=lambda img, meta: student(img)[:, :5])
        img, vi = torch.randn(1, 6, 3, 3), torch.randn(1, 3, 3, 3)
        label, weight = torch.zeros(1, 3, 3).long(), torch.ones(1, 3, 3)
        debug = {key: weight for key in ('R_struct', 'R_continuity', 'R_adj', 'R_sem')}
        debug.update(source_adj_matrix=torch.eye(5), target_adj_matrix=torch.eye(5))
        args = [(label, weight),
                (img, [], vi, label[:, None], vi, label, weight, weight[:, None], weight, img, []),
                (img, [], label[:, None], label, weight, weight, weight, debug),
                ({'proto_weight': weight, 'proto_margin': weight}, weight),
                (img, [], vi, label, weight, weight[:, None], debug, weight, weight, weight, weight)]
        initial_rng = torch.get_rng_state().clone()
        for method, values in zip(names, args):
            getattr(hook, method)(*values)
        self.assertEqual(student[0].num_batches_tracked.item(), 3)
        self.assertFalse(torch.equal(initial_rng, torch.get_rng_state()))
        self.assertTrue(mkdir.called)
        self.assertTrue(image_writer.save.called)
        self.assertTrue(hook._save_struct_matrix.called)
        self.assertTrue(hook._save_phy_map.called)
        # With all switches off, not even BN counters or RNG may advance.
        for flag in ('teacher_pseudo', 'phy', 'struct', 'proto', 'joint'):
            setattr(hook, 'save_' + flag + '_debug', False)
        before = torch.get_rng_state().clone()
        for method, values in zip(names, args):
            getattr(hook, method)(*values)
        self.assertEqual(student[0].num_batches_tracked.item(), 3)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    @unittest.skipUnless(HAS_BASELINE, 'Local pre-change snapshot unavailable')
    def test_total_loss_reporting_does_not_change_update(self):
        path = 'mmseg/models/uda/dacs.py'
        weights = []
        for original in (True, False):
            cls = load_methods(path, 'DACS', ['train_step'], original=original)
            parameter = torch.nn.Parameter(torch.tensor(2.0))
            def forward(self, img_metas):
                loss = parameter.square()
                loss.backward()
                return {'loss': 99.0, 'decode.loss_seg': 4.0,
                        'mix.decode.loss_seg': 2.0, 'loss_phy_src': 5.0}
            cls.__call__ = forward
            hook = cls()
            hook.lambda_phy_src = 0.1
            output = hook.train_step({'img_metas': [{}]},
                                    torch.optim.SGD([parameter], lr=0.1))
            weights.append(parameter.detach().clone())
            if not original:
                self.assertEqual(output['log_vars']['loss'], 6.5)
        torch.testing.assert_close(*weights, rtol=0, atol=0)

    @unittest.skipUnless(HAS_BASELINE, 'Local pre-change snapshot unavailable')
    def test_prototype_states_weights_losses_gradients_and_updates(self):
        path = 'mmseg/models/uda/proto_gate.py'
        old_cls = component(path, 'ProtoGate', original=True)
        new_cls = component(path, 'ProtoGate')
        torch.manual_seed(7)
        feat = torch.randn(2, 4, 4, 4)
        labels = (torch.arange(32).reshape(2, 4, 4) % 5).long()
        labels[0, 0, 0] = 255
        kwargs = dict(min_pixels=1, target_update_start=0, weight_start=2,
                      loss_start=0, separate_domain_prototypes=True)
        old, new = old_cls(**kwargs), new_cls(
            **kwargs, log_stats=False, collect_debug=False)
        diagnostic = new_cls(**kwargs)
        for step in (0, 3):
            for gate in (old, new, diagnostic):
                gate.update_source(feat, labels, iter=step)
                gate.update_target(feat, labels, torch.ones_like(labels).float(),
                                   torch.ones_like(labels).float(), iter=step)
            for key, value in old.state_dict().items():
                torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
            with torch.no_grad():
                a, full = old.compute_weight(feat, labels, out_size=(8, 8), iter=step)
                b, lean = new.compute_weight(feat, labels, out_size=(8, 8), iter=step)
                _, restored = diagnostic.compute_weight(
                    feat, labels, out_size=(8, 8), iter=step)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(set(full), set(restored))
            for key in full:
                torch.testing.assert_close(full[key], restored[key], rtol=0, atol=0)
            self.assertEqual(set(lean), {'proto_weight'})
            self.assertIn('proto_margin', full)
            a_feat, b_feat = feat.clone().requires_grad_(), feat.clone().requires_grad_()
            a_loss, _ = old.loss_target(a_feat, labels, torch.ones_like(labels), iter=step)
            b_loss, _ = new.loss_target(b_feat, labels, torch.ones_like(labels), iter=step)
            a_loss.backward()
            b_loss.backward()
            torch.testing.assert_close(a_loss, b_loss, rtol=0, atol=0)
            torch.testing.assert_close(a_feat.grad, b_feat.grad, rtol=0, atol=0)
            a_optim = torch.optim.AdamW([a_feat], lr=1e-3)
            b_optim = torch.optim.AdamW([b_feat], lr=1e-3)
            a_optim.step()
            b_optim.step()
            torch.testing.assert_close(a_feat, b_feat, rtol=0, atol=0)

    @unittest.skipUnless(HAS_BASELINE, 'Local pre-change snapshot unavailable')
    def test_structure_gate_preserves_responses_state_and_gradients(self):
        path = 'mmseg/models/uda/struct_prior_gate.py'
        old_cls = component(path, 'StructPriorConsistencyGate', original=True)
        new_cls = component(path, 'StructPriorConsistencyGate')
        for enabled in (False, True):
            for step in (0, 7000, 12000):
                kwargs = dict(num_classes=5, use_target_gate=enabled, detach_gate=False)
                old, new = old_cls(**kwargs), new_cls(**kwargs)
                torch.manual_seed(11)
                a_logits = torch.randn(2, 5, 5, 5, requires_grad=True)
                b_logits = a_logits.detach().clone().requires_grad_()
                labels = torch.arange(50).reshape(2, 5, 5) % 5
                weight = torch.rand(2, 5, 5, requires_grad=True)
                other_weight = weight.detach().clone().requires_grad_()
                common = dict(pseudo_label=labels, source_gt=labels,
                              source_probs=a_logits.detach().softmax(1), iter=step)
                a, full = old.apply(target_logits=a_logits, pseudo_weight=weight, **common)
                b, lean = new.apply(target_logits=b_logits, pseudo_weight=other_weight,
                                    collect_debug=False, log_stats=False, **common)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                for name in ('R_struct', 'R_continuity', 'R_adj', 'R_sem'):
                    torch.testing.assert_close(full[name], lean[name], rtol=0, atol=0)
                for key, value in old.state_dict().items():
                    torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
                self.assertNotIn('target_adj_matrix', lean)
                a.sum().backward()
                b.sum().backward()
                torch.testing.assert_close(weight.grad, other_weight.grad, rtol=0, atol=0)
                if a_logits.grad is not None:
                    torch.testing.assert_close(a_logits.grad, b_logits.grad, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

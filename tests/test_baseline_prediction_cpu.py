"""Real shared strict checkpoint loading and Student inference with a tiny CPU model."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_minimal_output import static_config
from test_struct_probe_cpu import load_function

try:
    import torch
except ImportError:
    torch = None


class Config(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    @staticmethod
    def fromstring(text, extension):
        return attributes(ast.literal_eval(text))


def attributes(value):
    if isinstance(value, dict):
        return Config({key: attributes(child) for key, child in value.items()})
    if isinstance(value, list):
        return [attributes(child) for child in value]
    return value


@unittest.skipIf(torch is None, 'Torch is required for the production loader and inference')
class BaselineCPU(unittest.TestCase):
    def test_strict_checkpoint_student_identity_and_unchanged_state(self):
        shared = ROOT/'tools/replay_hphy_mechanism.py'
        common = ROOT/'tools/replay_doris_mechanisms.py'
        scope = dict(torch=torch, np=np, Path=Path, deepcopy=copy.deepcopy, re=re, json=json, hashlib=hashlib)
        def digest(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        scope['file_sha256'] = digest
        for name in ('state_digest', 'checkpoint_state', 'inference_signature'):
            load_function(common, name, scope)
        for name in ('verify_protocol', 'resolve_iteration', 'data_protocol', 'load_model', 'infer_model'):
            load_function(shared, name, scope)
        cfg = attributes(static_config('configs/hrda/doris/goji26RGB2goji30RGB/doris_goji26RGB2goji30RGB_hrda_3vi.py'))
        spec = importlib.util.spec_from_file_location('baseline_mechanism_cpu', ROOT/'mmseg/utils/mechanism.py')
        mechanism = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mechanism)
        calls = []
        class Student(torch.nn.Module):
            def __init__(self, label):
                super().__init__()
                self.layer = torch.nn.Conv2d(6, 5, 1)
                self.bn = torch.nn.BatchNorm2d(6)
                with torch.no_grad():
                    self.layer.weight.zero_()
                    self.layer.bias.zero_()
                    self.layer.bias[label] = 5

            def forward(self, img, img_metas, return_loss, rescale):
                calls.append(dict(eval=not self.training, gradient=torch.is_grad_enabled(),
                    inference=torch.is_inference_mode_enabled(), shape=tuple(img[0].shape),
                    rescale=rescale, return_loss=return_loss))
                return [self.layer(self.bn(img[0])).argmax(1)[0].cpu().numpy()]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model, self.ema_model = Student(4), Student(0)

            def get_model(self): return self.model
            def get_ema_model(self): raise AssertionError('Baseline inference must never call Teacher')

        modules = {'mmcv': SimpleNamespace(Config=Config),
            'mmcv.runner': SimpleNamespace(wrap_fp16_model=lambda model: model),
            'mmseg.models.builder': SimpleNamespace(build_train_model=lambda config: Model()),
            'mmseg.models.uda.phy_reliability': SimpleNamespace(validate_phy_checkpoint=None,
                restore_phy_gate_config=None, phy_gate_parameters=None),
            'mmseg.utils.mechanism': mechanism}
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temporary, patch.dict(sys.modules, modules):
            folder = Path(temporary)
            path = folder/'iter_4000.pth'
            original = Model()
            checkpoint = dict(state_dict=original.state_dict(),
                meta=dict(iter=3999, config=repr(dict(cfg)), synthetic_fixture=True))
            torch.save(checkpoint, path)
            before_file = digest(path)
            model, info = scope['load_model'](cfg, path, False, torch.device('cpu'), folder)
            self.assertEqual(info['completed_iter'], 4000)
            sample = (torch.rand(1, 6, 256, 256), torch.zeros(1, 3, 256, 256), [{}])
            arrays, state = scope['infer_model'](model, sample, {}, False)
            np.testing.assert_array_equal(arrays['baseline_prediction'], 4)
            self.assertEqual(arrays['baseline_prediction'].shape, (256, 256))
            self.assertEqual(arrays['baseline_prediction'].dtype, np.uint8)
            self.assertEqual(state, scope['state_digest'](model.state_dict()))
            self.assertEqual(before_file, digest(path))
            self.assertFalse(any(parameter.grad is not None for parameter in model.parameters()))
            self.assertEqual(calls, [dict(eval=True, gradient=False, inference=True,
                shape=(1, 6, 256, 256), rescale=True, return_loss=False)])
            for issue in ('missing', 'extra', 'shape', 'nonfinite', 'iteration', 'branch'):
                changed = copy.deepcopy(checkpoint)
                key = next(iter(changed['state_dict']))
                if issue == 'missing': changed['state_dict'].pop(key)
                if issue == 'extra': changed['state_dict']['unexpected'] = torch.zeros(1)
                if issue == 'shape': changed['state_dict'][key] = torch.zeros(1)
                if issue == 'nonfinite': changed['state_dict'][key].fill_(float('nan'))
                if issue == 'iteration': changed['meta']['iter'] = 12000
                if issue == 'branch':
                    historical = copy.deepcopy(cfg)
                    historical.uda.struct_cfg.enable = True
                    changed['meta']['config'] = repr(dict(historical))
                torch.save(changed, path)
                with self.subTest(issue=issue), self.assertRaises(ValueError):
                    scope['load_model'](cfg, path, False, torch.device('cpu'), folder)


if __name__ == '__main__':
    unittest.main()

"""Production Structural gate/probe parity using CPU stand-ins and no MMCV."""
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace[name]


@unittest.skipIf(torch is None, 'Torch CPU runtime is required')
class StructuralProbeCPU(unittest.TestCase):
    def test_production_online_capture_offline_forward_schedule_and_readonly_state(self):
        from tools.doris_struct_evidence import prepare_struct_evidence
        gate_class = runpy.run_path(str(ROOT/'mmseg/models/uda/struct_prior_gate.py'))['StructPriorConsistencyGate']
        spec = importlib.util.spec_from_file_location('struct_probe_cpu', ROOT/'mmseg/utils/mechanism.py')
        mechanism = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mechanism)
        digest = load_function(ROOT/'tools/replay_doris_mechanisms.py', 'state_digest', dict(hashlib=hashlib))
        infer = load_function(ROOT/'tools/replay_struct_mechanism.py', 'infer_model',
            dict(torch=torch, np=np, shared=SimpleNamespace(state_digest=digest)))

        class Student(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(6)
                self.dropout = torch.nn.Dropout2d(.2)
                self.layer = torch.nn.Conv2d(6, 5, 1)
                self.test_cfg = dict(mode='whole')

            def generate_pseudo_label(self, image, metas):
                return self.layer(self.dropout(self.bn(image)))

            def forward(self, img, img_metas, return_loss=False, rescale=True):
                if return_loss:
                    raise AssertionError('Training is outside this diagnostic')
                if self.training:
                    raise AssertionError('Student display prediction requires eval mode')
                return [self.generate_pseudo_label(img[0], img_metas[0]).argmax(1)[0].cpu().numpy()]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model, self.ema_model = Student(), Student()
                self.struct_gate = gate_class(5, fusion_mode='doris_confidence_guided', struct_min_weight=.65,
                    struct_filter_threshold=.32)
                self.struct_cfg = {key: getattr(self.struct_gate, key) for key in (
                    'fusion_mode', 'continuity_weight', 'adjacency_weight', 'semantic_weight',
                    'struct_min_weight', 'struct_filter_threshold', 'struct_filter_mode',
                    'struct_warmup_iters', 'struct_rampup_iters', 'struct_positive_classes',
                    'use_source_adjacency_ema', 'use_semantic_proto', 'use_target_gate', 'eps')}
                self.local_iter, self.num_classes, self.pseudo_threshold = 12000, 5, 0.
                self.enable_struct = True
                self.enable_phy = self.enable_proto = self.enable_struct_edge = False

            def get_model(self): return self.model
            def get_ema_model(self): return self.ema_model
            def _doris_target_valid_mask(self, label, spatial): return spatial.bool()

        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)
        torch.manual_seed(3)
        original = Model()
        truth = torch.randint(0, 5, (1, 1, 256, 256))
        original.struct_gate.update_source_adjacency(truth)
        original.struct_gate.update_semantic_prototype(torch.rand(1, 5, 256, 256).softmax(1), truth)
        stream = io.BytesIO()
        torch.save(original.state_dict(), stream)
        stream.seek(0)
        model = Model()
        model.load_state_dict(torch.load(stream, map_location='cpu'), strict=True)
        metas = [dict(img_norm_cfg=dict(mean=np.zeros(6), std=np.ones(6)))]
        sample = (torch.rand(1, 6, 256, 256), torch.rand(1, 3, 256, 256), metas)
        valid = torch.zeros(1, 1, 256, 256)
        valid[:, :, 30:226, 30:226] = 1
        initial = digest(model.state_dict())
        for step, mode in ((2000, 'soft'), (8000, 'soft'), (12000, 'soft'), (12000, 'hard')):
            with self.subTest(step=step, mode=mode):
                model.local_iter = step
                model.struct_gate.struct_filter_mode = mode
                model.struct_cfg['struct_filter_mode'] = mode
                metadata = dict(iter=step, rule_iter=step, sample_id='SYNTHETIC.tif',
                    modules=dict(phy=False, struct=True, proto=False, edge=False))
                online = mechanism.branch_probe_payload(model, *sample, valid, metadata)
                with patch.dict(sys.modules, {'mmseg.utils.mechanism': mechanism}):
                    offline, state = infer(model, sample, dict(margins=[30, 30, 30, 30]))
                offline['image_rgb'] = online['image_rgb'].copy()
                offline, info = prepare_struct_evidence(offline, dict(struct_cfg=model.struct_cfg, rule_iter=step))
                self.assertTrue(info['struct_validation']['passed'], info['struct_validation'])
                for key in ('R_conf', 'R_loc', 'R_adj', 'R_prob', 'R_struct', 'A_struct',
                            'A_struct_valid_mask', 'C_loc', 'C_adj', 'C_prob', 'W_before', 'W_after', 'struct_prediction'):
                    np.testing.assert_array_equal(online[key], offline[key])
                self.assertEqual(initial, state)
                self.assertEqual(initial, digest(model.state_dict()))
                self.assertTrue(model.model.training)
                self.assertTrue(model.model.bn.training)
                self.assertFalse(any(parameter.grad is not None for parameter in model.parameters()))
                description = json.loads(str(online['metadata']))
                self.assertTrue(description['struct_validation']['passed'])
                self.assertIn('A_struct', description['tensor_shapes'])
                self.assertEqual(description['student_inference']['model_identity'], 'Student')
                self.assertEqual(description['student_inference']['test_cfg']['mode'], 'whole')
                self.assertEqual(description['panel_sources']['struct_prediction'],
                    'Student test-mode inference from the selected model state')
                if step == 2000:
                    np.testing.assert_array_equal(offline['A_struct'], 0)


if __name__ == '__main__':
    unittest.main()

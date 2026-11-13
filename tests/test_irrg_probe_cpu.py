"""Real branch computations on tiny synthetic CPU networks for the IRRG configs."""
import copy
import importlib.util
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from test_minimal_output import ROOT, static_config, dacs_component, torch


@unittest.skipIf(torch is None, 'PyTorch CPU runtime is required')
class IRRGProbeCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('irrg_observation', ROOT / 'mmseg/utils/mechanism.py')
        cls.mechanism = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mechanism)
        cls.DACS = dacs_component()
        scope = cls.DACS.__init__.__globals__
        production_schedule = SimpleNamespace(phy_gate_strength=scope['phy_gate_strength'])
        schedule_patch = patch.dict('sys.modules', {'mmseg.models.uda.phy_reliability': production_schedule})
        schedule_patch.start()
        cls.addClassCleanup(schedule_patch.stop)
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_enabled_branches_export_real_composition_and_keep_model_state(self):
        class Student(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Conv2d(6, 4, 1)
                self.classifier = torch.nn.Conv2d(4, 5, 1)
            def extract_feat(self, image):
                return [self.features(image)]
            def generate_pseudo_label(self, image, metas):
                return self.classifier(self.features(image))

        suffixes = ('_phy_gate', '_phy_struct_gate', '_ps_proto', '_struct_edge', '_pbcr', '_proto_struct_edge')
        for suffix in suffixes:
            path = 'configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi' + suffix + '.py'
            cfg = static_config(path)
            model = self.DACS(**cfg['uda'], model=cfg['model'], max_iters=15000)
            model.model = Student()
            model.ema_model = copy.deepcopy(model.model)
            with torch.no_grad():
                model.ema_model.classifier.weight.zero_()
                model.ema_model.classifier.bias.fill_(-5)
                model.ema_model.classifier.bias[0] = 5
            image, vi = torch.randn(1, 6, 16, 16), torch.rand(1, 3, 16, 16)
            label = torch.arange(256).reshape(1, 16, 16) % 5
            if model.enable_proto:
                model.proto_gate.update_source(torch.randn(1, 4, 16, 16), label)
            if model.enable_struct:
                model.struct_gate.update_source_adjacency(label)
                model.struct_gate.update_semantic_prototype(torch.rand(1, 5, 16, 16).softmax(1), label)
                model.struct_gate.update_source_adjacency = lambda *a: self.fail('Probe updated source adjacency')
                model.struct_gate.update_semantic_prototype = lambda *a: self.fail('Probe updated probability prototypes')
            metas = [dict(filename='SYNTHETIC.tif', img_norm_cfg=dict(mean=[0] * 6, std=[1] * 6))]
            valid = torch.ones(1, 1, 16, 16)
            for iteration in (0, 4000, 10000):
                model.local_iter = iteration
                before = {key: value.clone() for key, value in model.state_dict().items()}
                flags = self.mechanism.module_flags(model)
                metadata = dict(modules=flags, iter=iteration, sample_id='SYNTHETIC.tif',
                    synthetic_fixture=True, capture_mode='synthetic_cpu', config_sha256='synthetic')
                with self.subTest(suffix=suffix, iteration=iteration):
                    payload = self.mechanism.branch_probe_payload(model, image, vi, metas, valid, metadata)
                    self.assertIn('w_final', payload)
                    expected = payload['w_base'].copy()
                    if model.enable_phy:
                        expected *= payload['g_phy']
                        np.testing.assert_allclose(payload['w_t_phy'], expected)
                        self.assertTrue({'r_veg', 'q_goji_given_veg', 'p_goji', 'p_other_veg',
                            's_phy_class_agreement', 'g_phy', 'a_phy'}.issubset(payload))
                    if model.enable_struct:
                        np.testing.assert_allclose(payload['w_before_struct'], expected)
                        expected = payload['w_after_struct']
                        self.assertTrue({'r_conf', 'r_loc', 'r_adj', 'r_prob', 'r_struct',
                            'adjacency_available', 'semantic_available', 'evidence_count'}.issubset(payload))
                        meta = json.loads(str(payload['metadata']))
                        self.assertEqual(meta['struct_cfg']['fusion_mode'], 'struct_distinct_v1')
                        strength = .10 + .90 * min(iteration / 4000., 1.)
                        self.assertAlmostEqual(meta['struct_gate_strength'], strength)
                    if model.enable_proto:
                        np.testing.assert_allclose(payload['w_pre_proto'], expected)
                        expected = expected * payload['proto_weight']
                        self.assertIn('prototype', json.loads(str(payload['metadata'])))
                    if model.enable_struct_edge:
                        self.assertTrue({'e_prob', 'e_img', 'r_pair', 'r_edge', 'boundary_weight',
                            'teacher_pseudo_boundary', 'edge_supervision'}.issubset(payload))
                    np.testing.assert_allclose(payload['w_final'], expected, atol=1e-7)
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
                    self.assertFalse(any(parameter.grad is not None for parameter in model.parameters()))

    def test_legacy_boundary_minimum_is_bypassed_in_reliable_mode(self):
        cfg = static_config('configs/hrda/doris/goji26IRRG2goji30RGB/doris_goji26IRRG2goji30RGB_hrda_3vi_struct_edge.py')
        model = self.DACS(**cfg['uda'], model=cfg['model'], max_iters=15000)
        model.local_iter = 10000
        model.struct_edge_pair_prior_ready.fill_(True)
        model.struct_edge_pair_prior.fill_(.2)
        label = (torch.arange(256).reshape(1, 16, 16) % 5).long()
        image = torch.randn(1, 6, 16, 16)
        probabilities = torch.rand(1, 5, 16, 16).softmax(1)
        model.struct_edge_pair_min_weight = .01
        first = model.doris_edge_responses(image, label, probabilities)
        model.struct_edge_pair_min_weight = .99
        second = model.doris_edge_responses(image, label, probabilities)
        for key in first:
            if first[key] is not None:
                torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
